from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.core.config import JSON_STORE_DIR


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class RejectLogRepository:
    """Append-only reject log backed by a local JSON lines file."""

    _FILENAME = "reject_log.jsonl"

    def __init__(self, store_dir: Path | None = None) -> None:
        self._path = (store_dir or JSON_STORE_DIR) / self._FILENAME
        self._lock = threading.Lock()

    def log_reject(self, payload: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "session_id": str(payload.get("session_id") or ""),
            "event_id": str(payload.get("event_id") or ""),
            "template_version_id": int(payload.get("template_version_id") or 0),
            "line_id": payload.get("line_id"),
            "station_id": payload.get("station_id"),
            "part_name": payload.get("part_name"),
            "decision_code": str(payload.get("decision_code") or "REJECT"),
            "reject_reason_code": str(payload.get("reject_reason_code") or "ERROR"),
            "operator_user_id": int(payload["operator_user_id"]) if payload.get("operator_user_id") is not None else None,
            "mp_check": payload.get("mp_check"),
            "validation_details": payload.get("validation_details"),
            "part_ready": payload.get("part_ready"),
            "sticker_detection": payload.get("sticker_detection"),
            "part_ready_roi_meta": payload.get("part_ready_roi_meta"),
            "sticker_roi_meta": payload.get("sticker_roi_meta"),
            "created_at": _utcnow_iso(),
        }
        line = json.dumps(entry, ensure_ascii=True)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return entry

    def list_recent(self, *, limit: int = 200) -> list[dict[str, Any]]:
        try:
            with self._lock:
                raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []

        entries: list[dict[str, Any]] = []
        for line in reversed(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            entries.append(entry)
            if len(entries) >= limit:
                break
        return entries
