"""NG JSONL logger — daily rolling log for rejected inspections (no images)."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class NgCacheLogger:
    """Append-only JSONL logger for rejected component-count inspections.

    - One file per day: ng_log_YYYY-MM-DD.jsonl
    - Auto-purge files older than `retention_days`
    - Thread-safe via a module-level lock
    """

    _lock = threading.Lock()

    def __init__(self, log_dir: str | Path, *, retention_days: int = 30) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = max(1, int(retention_days))

    def _today_path(self) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"ng_log_{ts}.jsonl"

    def log_ng(
        self,
        *,
        session_id: str,
        event_id: str | None,
        mode: str,
        part_type: str | None,
        reject_reason: str | None,
        per_roi: list[dict[str, Any]],
        extra: dict[str, Any] | None = None,
    ) -> str | None:
        """Append one NG entry. Returns the file path written, or None on failure."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session": session_id,
            "event_id": event_id,
            "mode": mode,
            "part_type": part_type,
            "reject_reason": reject_reason,
            "per_roi": per_roi,
        }
        if extra:
            entry["extra"] = extra

        path = self._today_path()
        with self._lock:
            try:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                self._maybe_purge()
                return str(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[ng_logger] write failed: %s", exc)
                return None

    def _maybe_purge(self) -> None:
        """Remove JSONL files older than retention_days."""
        try:
            cutoff = time.time() - (self.retention_days * 86400)
            for f in self.log_dir.glob("ng_log_*.jsonl"):
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                except OSError:
                    pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ng_logger] purge failed: %s", exc)
