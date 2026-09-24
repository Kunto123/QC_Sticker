"""Read-only local mirror of the plant MES's `PI_OkDataPart` table.

In production (postgresql/sqlserver backends) this table is populated
upstream (at production/molding time) and this app never writes to it — see
`postgres/ok_data_part_repository.py` / `sqlserver/ok_data_part_repository.py`.
This local, JSON-backed variant exists only for dev machines without the MES
DB and for tests, which use `seed_row` to populate fixture rows.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from backend.app.repositories.base_json import JsonRepository


class OkDataPartRepository(JsonRepository):
    def __init__(self) -> None:
        super().__init__("ok_data_part.json", [])

    def seed_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Insert or replace a fixture row by IDDataPart. Local/dev/test only —
        this table is upstream-populated and read-only in production."""
        rows = self.load()
        id_data_part = str(row.get("IDDataPart") or "")
        rows = [r for r in rows if str(r.get("IDDataPart") or "") != id_data_part]
        rows.append(dict(row))
        self.save(rows)
        return dict(row)

    def get_by_id_data_part(self, id_data_part: str) -> dict[str, Any] | None:
        normalized = str(id_data_part or "").strip()
        if not normalized:
            return None
        record = next(
            (r for r in self.load() if str(r.get("IDDataPart") or "") == normalized),
            None,
        )
        if record is None:
            return None
        record = dict(record)
        prod_date = record.get("ProdDate")
        if isinstance(prod_date, str):
            record["ProdDate"] = datetime.fromisoformat(prod_date)
        return record
