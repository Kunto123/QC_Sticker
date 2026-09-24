"""Local JSON-backed mirror of the plant MES's `PI_BoxLuggage` table.

In production (postgresql/sqlserver backends) this is a real, fixed-schema
MES table — see `postgres/box_luggage_repository.py` /
`sqlserver/box_luggage_repository.py`. This local variant exists for dev
machines without the MES DB and for tests.

Row model: one row is inserted per unit of the box's goal quantity at
Datapart 1 (all empty on Data1/Data2/DateCheckMC/DateSendDB); each accepted
OK fills exactly one still-empty row; Datapart 2 closes all of a box's rows
at once. There is no dedicated box/batch-id column on this table — rows
belonging to the same box are grouped by sharing an identical `DatapartPI`
value (the raw Datapart 1 scan string), which is unique per physical box tag.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from backend.app.repositories.base_json import JsonRepository

ABANDONED_SENTINEL = "ABANDONED"


def _utcnow_iso() -> str:
    from datetime import UTC

    return datetime.now(UTC).isoformat()


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class BoxLuggageRepository(JsonRepository):
    def __init__(self) -> None:
        super().__init__("box_luggage.json", [])

    def open_box(
        self,
        *,
        part_name: str,
        mp_check: str,
        datapart_pi: str,
        datapart_pi_date: Any,
        mpid_data_part: str,
        line: str,
        qty: int,
    ) -> list[dict[str, Any]]:
        rows = self.load()
        base = self.next_id(rows, "No") - 1 if rows else 0
        new_rows = []
        for offset in range(1, qty + 1):
            row = {
                "No": base + offset,
                "PartName": part_name,
                "DateCheckMC": None,
                "MPCheck": mp_check,
                "DateSendDB": None,
                "DatapartID": None,
                "DatapartDate": None,
                "MPIDDataPart": mpid_data_part,
                "Data1": None,
                "Data2": None,
                "Line": line,
                "DatapartPI": datapart_pi,
                "DatapartPIDate": _iso(datapart_pi_date),
            }
            rows.append(row)
            new_rows.append(row)
        self.save(rows)
        return new_rows

    def record_ok(self, *, datapart_pi: str, data1: str, data2: str, mp_check: str) -> dict[str, Any] | None:
        rows = self.load()
        candidates = [
            r for r in rows
            if r.get("DatapartPI") == datapart_pi and r.get("DatapartID") is None and r.get("Data1") is None
        ]
        if not candidates:
            return None
        target = min(candidates, key=lambda r: r["No"])
        for row in rows:
            if row["No"] == target["No"]:
                row["Data1"] = data1
                row["Data2"] = data2
                row["MPCheck"] = mp_check
                row["DateCheckMC"] = _utcnow_iso()
                row["DateSendDB"] = _utcnow_iso()
                self.save(rows)
                return dict(row)
        return None

    def close_box(self, *, datapart_pi: str, datapart_id: str, datapart_date: Any) -> int:
        rows = self.load()
        count = 0
        for row in rows:
            if row.get("DatapartPI") == datapart_pi and row.get("DatapartID") is None:
                row["DatapartID"] = datapart_id
                row["DatapartDate"] = _iso(datapart_date)
                count += 1
        if count:
            self.save(rows)
        return count

    def abandon_box(self, *, datapart_pi: str) -> int:
        rows = self.load()
        count = 0
        for row in rows:
            if row.get("DatapartPI") == datapart_pi and row.get("DatapartID") is None:
                row["DatapartID"] = ABANDONED_SENTINEL
                row["DatapartDate"] = _utcnow_iso()
                count += 1
        if count:
            self.save(rows)
        return count

    def find_open_box(self, *, part_name: str) -> dict[str, Any] | None:
        rows = [
            r for r in self.load()
            if r.get("PartName") == part_name and r.get("DatapartID") is None
        ]
        if not rows:
            return None
        datapart_pi = rows[0]["DatapartPI"]
        group = [r for r in rows if r.get("DatapartPI") == datapart_pi]
        remaining = [r for r in group if r.get("Data1") is None]
        return {
            "datapart_pi": datapart_pi,
            "qty_goal": len(group),
            "qty_remaining": len(remaining),
        }
