from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.repositories.hybrid_inspection_results_repository import (
        HybridInspectionResultsRepository,
    )

logger = logging.getLogger(__name__)

_MAX_RETRY_COUNT = 5
_DEAD_LETTER_STATUS = "dead_letter"
_BACKOFF_BASE_SECONDS = 10       # 10s, 20s, 40s, 80s, 160s
_WORKER_INTERVAL_SECONDS = 30    # poll every 30 s
_BATCH_SIZE = 50


class PushWorker:
    """Background thread that drains `pending`/`failed` inspection results
    to the SQL Server mirror.

    Lifecycle:
        pending  → sent          (success)
        pending  → failed        (first failure; will retry)
        failed   → sent          (retry success)
        failed   → dead_letter   (after MAX_RETRY_COUNT failures)

    The worker uses exponential back-off based on `retry_count`; it will
    skip a record whose next-attempt time has not arrived yet.
    """

    def __init__(
        self,
        results_repo: "HybridInspectionResultsRepository",
        *,
        interval_seconds: int = _WORKER_INTERVAL_SECONDS,
        batch_size: int = _BATCH_SIZE,
        max_retry_count: int = _MAX_RETRY_COUNT,
        backoff_base: int = _BACKOFF_BASE_SECONDS,
    ) -> None:
        self._repo = results_repo
        self._interval = max(5, int(interval_seconds))
        self._batch_size = max(1, int(batch_size))
        self._max_retry = max(1, int(max_retry_count))
        self._backoff_base = max(1, int(backoff_base))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="qc-push-worker",
            daemon=True,
        )
        self._thread.start()
        logger.info("[push-worker] started (interval=%ds, max_retry=%d)", self._interval, self._max_retry)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("[push-worker] stopped")

    def run_once(self) -> dict:
        """Process one batch synchronously. Used in tests and admin triggers."""
        return self._process_batch()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                stats = self._process_batch()
                if stats["processed"]:
                    logger.info("[push-worker] batch: %s", stats)
            except Exception as exc:  # noqa: BLE001
                logger.error("[push-worker] unexpected error: %s", exc, exc_info=True)
            self._stop_event.wait(timeout=self._interval)

    def _process_batch(self) -> dict:
        if self._repo._sql_mirror_repo is None:  # noqa: SLF001
            return {"processed": 0, "sent": 0, "dead_lettered": 0, "skipped": 0}

        records = self._repo._local_repo.list_results(  # noqa: SLF001
            push_status="pending", limit=self._batch_size, offset=0
        )
        failed_records = self._repo._local_repo.list_results(  # noqa: SLF001
            push_status="failed", limit=self._batch_size, offset=0
        )

        processed = sent = dead_lettered = skipped = 0

        for record in [*records, *failed_records]:
            if self._stop_event.is_set():
                break

            retry_count = int(record.get("retry_count") or 0)

            # Dead-letter records that exceeded retry limit
            if retry_count >= self._max_retry:
                self._repo._local_repo.update_result(  # noqa: SLF001
                    int(record["id"]),
                    {"push_status": _DEAD_LETTER_STATUS},
                )
                dead_lettered += 1
                processed += 1
                logger.warning(
                    "[push-worker] dead-letter result #%s after %d retries",
                    record["id"], retry_count,
                )
                continue

            # Exponential back-off: skip if last attempt too recent
            if not self._is_due(record):
                skipped += 1
                continue

            try:
                result = self._repo._apply_mirror_result(record)  # noqa: SLF001
                processed += 1
                if result.get("push_status") == "sent":
                    sent += 1
                else:
                    logger.debug("[push-worker] push still failed for result #%s", record["id"])
            except Exception as exc:  # noqa: BLE001
                logger.error("[push-worker] error processing result #%s: %s", record["id"], exc)
                processed += 1

        return {
            "processed": processed,
            "sent": sent,
            "dead_lettered": dead_lettered,
            "skipped": skipped,
        }

    def _is_due(self, record: dict) -> bool:
        """True if enough time has passed since the last push attempt."""
        from datetime import UTC, datetime

        last_attempt = record.get("last_push_attempt_at")
        if not last_attempt:
            return True
        try:
            last_dt = datetime.fromisoformat(str(last_attempt).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return True
        retry_count = max(0, int(record.get("retry_count") or 0))
        wait_seconds = self._backoff_base * (2 ** max(0, retry_count - 1))
        return (datetime.now(UTC) - last_dt).total_seconds() >= wait_seconds
