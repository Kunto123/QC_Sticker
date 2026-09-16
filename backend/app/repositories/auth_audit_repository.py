from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from backend.app.core.config import JSON_STORE_DIR


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class AuthAuditRepository:
    """Append-only auth audit log backed by a local JSON lines file.

    Each call to :meth:`log` atomically appends one JSON line.
    :meth:`list_recent` reads the file and returns entries newest-first.
    Survives backend restart because entries are written to disk.
    """

    _FILENAME = "auth_audit.jsonl"

    def __init__(self, store_dir: Path | None = None) -> None:
        self._path = (store_dir or JSON_STORE_DIR) / self._FILENAME
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log(
        self,
        event_type: str,
        *,
        user_id: int | None = None,
        username: str | None = None,
        session_id: str | None = None,
        actor_id: int | None = None,
        actor_username: str | None = None,
        ip_address: str | None = None,
        client_name: str | None = None,
        details: str | None = None,
    ) -> None:
        entry = {
            "event_type": str(event_type),
            "user_id": int(user_id) if user_id is not None else None,
            "username": str(username) if username is not None else None,
            "session_id": str(session_id) if session_id is not None else None,
            "actor_id": int(actor_id) if actor_id is not None else None,
            "actor_username": str(actor_username) if actor_username is not None else None,
            "ip_address": str(ip_address) if ip_address is not None else None,
            "client_name": str(client_name) if client_name is not None else None,
            "details": str(details) if details is not None else None,
            "created_at": _utcnow_iso(),
        }
        line = json.dumps(entry, ensure_ascii=True)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def list_recent(self, *, limit: int = 200, user_id: int | None = None) -> list[dict]:
        """Return up to *limit* recent events, newest first.

        If *user_id* is given, only events for that user are returned.
        """
        try:
            with self._lock:
                raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []

        entries: list[dict] = []
        for line in reversed(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if user_id is not None and entry.get("user_id") != user_id:
                continue
            entries.append(entry)
            if len(entries) >= limit:
                break
        return entries
