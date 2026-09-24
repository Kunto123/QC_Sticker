"""Box-level datapart traceability: Datapart 1 (open) -> per-OK fill ->
Datapart 2 (close), writing into the plant MES's `PI_BoxLuggage` table
(read-only lookups against `PI_OkDataPart`).

Scan strings are pipe-delimited, produced by a USB keyboard-wedge scanner,
e.g.:

    8125B-K2S -H3XX|1867490|5|B09SB1100748292049|QRTag
    part_number      |company_id|qty_box|id_data_part           |tag_marker

`DatapartPI` on `PI_BoxLuggage` stores this raw string verbatim (there is no
dedicated PartNumber/CompanyID/QtyBox column on that table) — it is re-parsed
whenever those values are needed again, e.g. for the Datapart 2 cross-check.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("backend.box_tracking")

_SCAN_FIELD_COUNT = 5
_SCAN_FIELDS = ("part_number", "company_id", "qty_box", "id_data_part", "tag_marker")
_CROSS_CHECK_FIELDS = ("part_number", "company_id", "qty_box")


class BoxTrackingError(Exception):
    """Raised for any box-tracking failure that must fail closed at the route
    layer (age gate, unknown datapart, Datapart-2 mismatch, box not found)."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def parse_datapart_scan(raw: str) -> dict[str, str]:
    """Parse a pipe-delimited Datapart 1/2 scan string. Raises BoxTrackingError
    on anything that doesn't look like a real scan — never guesses."""
    fields = str(raw or "").split("|")
    if len(fields) != _SCAN_FIELD_COUNT:
        raise BoxTrackingError(
            "INVALID_SCAN",
            f"Invalid scan format: expected {_SCAN_FIELD_COUNT} pipe-delimited fields, got {len(fields)}.",
        )
    parsed = {name: value.strip() for name, value in zip(_SCAN_FIELDS, fields)}
    if not parsed["part_number"] or not parsed["company_id"] or not parsed["id_data_part"]:
        raise BoxTrackingError("INVALID_SCAN", "Invalid scan: part number, company id and data part id are required.")
    # Validates qty_box is a real positive integer; raises on failure.
    to_int_qty(parsed["qty_box"])
    return parsed


def to_int_qty(raw_qty: str) -> int:
    try:
        qty = int(str(raw_qty or "").strip())
    except (TypeError, ValueError) as exc:
        raise BoxTrackingError("INVALID_SCAN", f"Invalid scan: qty_box {raw_qty!r} is not an integer.") from exc
    if qty <= 0:
        raise BoxTrackingError("INVALID_SCAN", f"Invalid scan: qty_box {qty} must be positive.")
    return qty


def _utcnow() -> datetime:
    return datetime.now(UTC)


class BoxTrackingService:
    def __init__(self, box_repo: Any, ok_data_part_repo: Any, app_config: Any, machine_settings: Any) -> None:
        self._box_repo = box_repo
        self._ok_data_part_repo = ok_data_part_repo
        self._app_config = app_config
        self._machine_settings = machine_settings

    def start_box(self, *, template: Any, raw_scan: str, username: str) -> dict[str, Any]:
        scan = parse_datapart_scan(raw_scan)
        ok_row = self._ok_data_part_repo.get_by_id_data_part(scan["id_data_part"])
        if ok_row is None:
            raise BoxTrackingError("NOT_FOUND", f"Data part {scan['id_data_part']!r} not found in PI_OkDataPart.")
        prod_date = ok_row["ProdDate"]
        age_hours = (_utcnow() - prod_date).total_seconds() / 3600.0
        min_age_hours = float(template.box_tracking.min_age_hours)
        if age_hours < min_age_hours:
            raise BoxTrackingError(
                "PART_NOT_AGED",
                f"Part not aged enough: {age_hours:.1f}h since ProdDate, minimum is {min_age_hours:.1f}h.",
            )
        qty = to_int_qty(scan["qty_box"])
        line = str(getattr(self._machine_settings.identity, "line", "") or "")
        self._box_repo.open_box(
            part_name=template.name,
            mp_check=username,
            datapart_pi=raw_scan,
            datapart_pi_date=prod_date,
            mpid_data_part=ok_row["MPID"],
            line=line,
            qty=qty,
        )
        return {"datapart_pi": raw_scan, "qty_goal": qty, "qty_remaining": qty}

    def record_ok(self, *, datapart_pi: str, part_ready_ratio: float, sticker_confidence: float, mp_check: str) -> dict[str, Any] | None:
        data1 = f"{float(part_ready_ratio or 0.0) * 100:.2f}"
        data2 = f"{float(sticker_confidence or 0.0) * 100:.2f}"
        return self._box_repo.record_ok(datapart_pi=datapart_pi, data1=data1, data2=data2, mp_check=mp_check)

    def close_box(self, *, datapart_pi: str, raw_scan2: str) -> dict[str, Any]:
        scan1 = parse_datapart_scan(datapart_pi)
        scan2 = parse_datapart_scan(raw_scan2)
        mismatches = [field for field in _CROSS_CHECK_FIELDS if scan1[field] != scan2[field]]
        if mismatches:
            raise BoxTrackingError("MISMATCH", f"Datapart 2 mismatch on: {', '.join(mismatches)}.")
        count = self._box_repo.close_box(datapart_pi=datapart_pi, datapart_id=scan2["id_data_part"], datapart_date=_utcnow())
        if count == 0:
            raise BoxTrackingError("NOT_FOUND", "No open box found for this datapart.")
        return {"closed_rows": count}

    def abandon_box(self, *, datapart_pi: str) -> dict[str, Any]:
        count = self._box_repo.abandon_box(datapart_pi=datapart_pi)
        if count == 0:
            raise BoxTrackingError("NOT_FOUND", "No open box found for this datapart.")
        return {"abandoned_rows": count}
