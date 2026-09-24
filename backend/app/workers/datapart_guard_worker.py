from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 10


class DatapartGuardWorker:
    """Background thread that periodically asks the datapart guard to
    re-check whether a locked batch has been confirmed (DatapartID filled)
    by the downstream MES. Cheap no-op while unlocked."""

    def __init__(self, guard_service, *, interval_seconds: int = _POLL_INTERVAL_SECONDS) -> None:
        self._guard = guard_service
        self._interval = max(2, int(interval_seconds))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="qc-datapart-guard", daemon=True)
        self._thread.start()
        logger.info("[datapart-guard] worker started (interval=%ds)", self._interval)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("[datapart-guard] worker stopped")

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._guard.poll_once()
            except Exception as exc:  # noqa: BLE001
                logger.error("[datapart-guard] worker error: %s", exc, exc_info=True)
            self._stop_event.wait(timeout=self._interval)
