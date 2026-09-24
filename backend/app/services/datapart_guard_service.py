"""Datapart guard — locks further judgement commits every N accepted+pushed
results until a downstream MES process has filled DatapartID for all of
them on the SQL mirror table.

State lives in DatapartGuardRepository (JSON file), not just in memory: a
backend restart must not silently forget a batch that was never actually
confirmed. `record_pushed` is the only write path that grows the pending
batch; `poll_once` (background worker) and `override` (admin) are the only
ways to clear it.
"""
from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

DATAPART_GUARD_BATCH_SIZE = 5


class DatapartGuardService:
    def __init__(self, repo, mirror_repo) -> None:
        self._repo = repo
        self._mirror_repo = mirror_repo
        self._lock = threading.RLock()
        state = self._repo.load_state()
        self._pending_ids: list[int] = [int(v) for v in (state.get("pending_ids") or [])]
        self._locked: bool = bool(state.get("locked", False))
        self._last_override_at: str | None = state.get("last_override_at")
        self._last_override_by: str | None = state.get("last_override_by")

    def is_locked(self) -> bool:
        with self._lock:
            return self._locked

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "locked": self._locked,
                "pending_ids": list(self._pending_ids),
                "batch_size": DATAPART_GUARD_BATCH_SIZE,
                "last_override_at": self._last_override_at,
                "last_override_by": self._last_override_by,
            }

    def record_pushed(self, mirror_id: int | None) -> None:
        """Call once per ACCEPT commit that was successfully pushed to the
        SQL mirror (has a sql_mirror_id). Locks once the pending batch
        reaches DATAPART_GUARD_BATCH_SIZE."""
        if mirror_id is None or self._mirror_repo is None:
            return
        with self._lock:
            if self._locked:
                # Already locked and awaiting confirmation — don't grow the
                # batch further; the next batch starts once this one clears.
                return
            self._pending_ids.append(int(mirror_id))
            if len(self._pending_ids) >= DATAPART_GUARD_BATCH_SIZE:
                self._locked = True
                logger.warning(
                    "[datapart-guard] locked after %d pushes awaiting DatapartID: %s",
                    len(self._pending_ids), self._pending_ids,
                )
            self._persist()

    def poll_once(self) -> None:
        """Background check (called by DatapartGuardWorker): while locked,
        ask the mirror table which pending ids still have no DatapartID.
        Unlocks once none remain."""
        with self._lock:
            if not self._locked or not self._pending_ids or self._mirror_repo is None:
                return
            pending = list(self._pending_ids)
        try:
            unfilled = self._mirror_repo.unfilled_datapart_ids(pending)
        except Exception as exc:  # noqa: BLE001
            logger.error("[datapart-guard] poll failed: %s", exc)
            return
        if unfilled:
            return
        with self._lock:
            # Only clear if the batch we checked is still the current one
            # (nothing else — e.g. a concurrent override — changed it meanwhile).
            if self._pending_ids == pending:
                self._pending_ids = []
                self._locked = False
                self._persist()
                logger.info("[datapart-guard] unlocked — all %d DatapartID confirmed", len(pending))

    def override(self, by: str) -> None:
        """Admin escape hatch: force-clear a stuck lock (e.g. downstream MES
        is down). Always logged so it shows up in the audit trail."""
        with self._lock:
            self._pending_ids = []
            self._locked = False
            self._last_override_at = datetime.now(UTC).isoformat()
            self._last_override_by = str(by or "").strip() or None
            self._persist()
            logger.warning("[datapart-guard] manually overridden by %s", self._last_override_by)

    def _persist(self) -> None:
        self._repo.save_state(
            {
                "pending_ids": list(self._pending_ids),
                "locked": self._locked,
                "locked_at": datetime.now(UTC).isoformat() if self._locked else None,
                "last_override_at": self._last_override_at,
                "last_override_by": self._last_override_by,
            }
        )
