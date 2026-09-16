from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.app.repositories.base_json import JsonRepository


class WorkstationRegistryRepository(JsonRepository):
    """Keeps a lightweight registry of known workstations based on heartbeat pings."""

    def __init__(self) -> None:
        super().__init__("workstations.json", {"workstations": []})

    def _payload(self) -> dict[str, Any]:
        return self.load()

    def list_workstations(self) -> list[dict[str, Any]]:
        return self._payload()["workstations"]

    def delete_workstation(self, machine_id: str) -> bool:
        normalized = str(machine_id or "").strip()
        if not normalized:
            return False
        store = self._payload()
        workstations = store["workstations"]
        before = len(workstations)
        store["workstations"] = [
            item for item in workstations if str(item.get("machine_id") or "").strip() != normalized
        ]
        if len(store["workstations"]) == before:
            return False
        self.save(store)
        return True

    def heartbeat(
        self,
        *,
        machine_id: str,
        client_version: str | None = None,
        line_id: str | None = None,
        station_id: str | None = None,
        ip_address: str | None = None,
    ) -> dict[str, Any]:
        store = self._payload()
        workstations = store["workstations"]
        now = datetime.now(UTC).isoformat()

        existing = next(
            (w for w in workstations if w.get("machine_id") == machine_id),
            None,
        )
        if existing is not None:
            existing["last_seen_at"] = now
            if client_version is not None:
                existing["client_version"] = client_version
            if line_id is not None:
                existing["line_id"] = line_id
            if station_id is not None:
                existing["station_id"] = station_id
            if ip_address is not None:
                existing["ip_address"] = ip_address
            self.save(store)
            return dict(existing)

        record: dict[str, Any] = {
            "machine_id": machine_id,
            "client_version": client_version,
            "line_id": line_id,
            "station_id": station_id,
            "ip_address": ip_address,
            "first_seen_at": now,
            "last_seen_at": now,
        }
        workstations.append(record)
        self.save(store)
        return dict(record)

    def get_stale(self, *, stale_after_seconds: int = 120) -> list[dict[str, Any]]:
        """Return workstations not seen within stale_after_seconds."""
        now = datetime.now(UTC)
        result = []
        for w in self.list_workstations():
            last_seen_str = w.get("last_seen_at")
            if not last_seen_str:
                result.append(w)
                continue
            try:
                last_seen = datetime.fromisoformat(last_seen_str.replace("Z", "+00:00"))
                elapsed = (now - last_seen).total_seconds()
                if elapsed > stale_after_seconds:
                    result.append(w)
            except (ValueError, TypeError):
                result.append(w)
        return result
