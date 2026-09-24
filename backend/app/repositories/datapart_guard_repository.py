"""Datapart guard state — `data/json_store/datapart_guard.json`.

Durable on purpose: every DATAPART_GUARD_BATCH_SIZE accepted judgements that
get pushed to the SQL mirror, further commits lock until the downstream MES
has filled DatapartID for all of them (services/datapart_guard_service.py).
Keeping this in memory only would silently drop the lock/pending-batch state
on a backend restart, letting judgements past a batch that was never
actually confirmed — see SESSION_LOG for the equivalent bug already fixed
once in inspection_results_repository.py's read-modify-write pattern.
"""
from __future__ import annotations

from typing import Any

from backend.app.repositories.base_json import JsonRepository

_DEFAULT_STATE: dict[str, Any] = {
    "pending_ids": [],
    "locked": False,
    "locked_at": None,
    "last_override_at": None,
    "last_override_by": None,
}


class DatapartGuardRepository(JsonRepository):
    def __init__(self) -> None:
        super().__init__("datapart_guard.json", dict(_DEFAULT_STATE))

    def load_state(self) -> dict[str, Any]:
        raw = self.load()
        return {**_DEFAULT_STATE, **raw}

    def save_state(self, state: dict[str, Any]) -> None:
        self.save({**_DEFAULT_STATE, **state})
