from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from backend.app.core.config import AppConfig


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _quote_ident(name: str) -> str:
    return ".".join(f"[{part}]" for part in str(name).split("."))


_MEMBER_ID_CODE_PATTERN = re.compile(r"\d{4}")


def _mp_check_code(mp_check: Any) -> str | None:
    """Member_ID is formatted like "ID 9101 PUTRA" — only the 4-digit code
    in the middle gets pushed to MPCheck, not the full string. Falls back to
    the raw value when no 4-digit run is found, so a differently-shaped
    Member_ID still pushes something rather than nothing."""
    text = str(mp_check or "")
    match = _MEMBER_ID_CODE_PATTERN.search(text)
    return match.group(0) if match else (text or None)


def _line_last_char(line_id: Any) -> str | None:
    """identity.line is a line code like "GB3" or "A01" — only its last
    character is pushed to the Line column."""
    text = str(line_id or "").strip()
    return text[-1] if text else None


def _as_percentage(ratio: Any) -> str | None:
    """Data1/Data2 are stored locally as 0..1 confidence ratios but pushed
    to SQL as a whole-number percentage string, e.g. 0.9326 -> "93%"."""
    if ratio is None:
        return None
    try:
        return f"{round(float(ratio) * 100.0)}%"
    except (TypeError, ValueError):
        return None


class SqlServerInspectionMirrorRepository:
    """Pushes inspection results to an external push table owned by the
    plant MES / reporting system, not this app — table and column names are
    deployment-specific (`QC_SUITE_INSPECTION_*` in core/config.py, default
    `qc_inspection_push`). This repository never creates or alters it."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._table = _quote_ident(config.inspection_push_table)
        self._col_id = _quote_ident(config.inspection_push_col_id)
        self._col_datapart_id = _quote_ident(config.inspection_push_col_datapart_id)
        # Each logical field may map to more than one physical column (a
        # target table that duplicates the same value across redundant
        # columns) — every column in the list gets the same value on insert.
        self._field_columns: list[tuple[str, list[str]]] = [
            ("PartName", [_quote_ident(c) for c in config.inspection_push_col_part_name]),
            ("DateCheckMC", [_quote_ident(c) for c in config.inspection_push_col_date_check_mc]),
            ("MPCheck", [_quote_ident(c) for c in config.inspection_push_col_mp_check]),
            ("Data1", [_quote_ident(c) for c in config.inspection_push_col_data1]),
            ("Data2", [_quote_ident(c) for c in config.inspection_push_col_data2]),
            ("Line", [_quote_ident(c) for c in config.inspection_push_col_line]),
        ]

    def _connect(self):
        import pyodbc

        connection_string = (
            f"DRIVER={{{self._config.sql_driver}}};"
            f"SERVER={self._config.sql_server};"
            f"DATABASE={self._config.sql_database};"
            f"UID={self._config.sql_username};"
            f"PWD={self._config.sql_password};"
            "Encrypt=yes;"
            "TrustServerCertificate=yes;"
        )
        return pyodbc.connect(connection_string, timeout=5)

    @staticmethod
    def build_sql_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Map a local inspection result to the agreed SQL push contract.

        Contract (fixed — do not change without downstream coordination):
            PartName    <- template_name (the template's own "Preset Name" in
                           Admin -> Templates; fallback: expected_class, part_name)
            DateCheckMC <- inspected_at
            MPCheck     <- mp_check  (operator username)
            Data1       <- part_ready_match_ratio, as a whole-number percentage string (e.g. "93%")
            Data2       <- sticker_confidence, as a whole-number percentage string (e.g. "81%")
            Line        <- line_id

        All other fields are stored locally only (as raw 0..1 ratios).
        Actual destination column names on the target table are
        configurable, and a field may fan out to more than one physical
        column — see `QC_SUITE_INSPECTION_COL_*` in core/config.py. MPCheck,
        Line, Data1 and Data2 are further transformed (see `_mp_check_code`
        / `_line_last_char` / `_as_percentage`) before being pushed.
        """
        return {
            "PartName": payload.get("template_name") or payload.get("expected_class") or payload.get("part_name"),
            "DateCheckMC": payload.get("inspected_at") or _utcnow_iso(),
            "MPCheck": _mp_check_code(payload.get("mp_check")),
            "Data1": _as_percentage(payload.get("part_ready_match_ratio")),   # confidence part ready
            "Data2": _as_percentage(payload.get("sticker_confidence")),        # confidence sticker
            "Line": _line_last_char(payload.get("line_id")),
        }

    def _build_insert_plan(self, record: dict[str, Any]) -> tuple[list[str], list[Any]]:
        """Flatten `self._field_columns` against `record` into parallel
        (quoted column, value) lists — duplicating a field's value into every
        physical column configured for it. Kept separate from `create_result`
        so it's unit-testable without a live DB connection."""
        columns: list[str] = []
        values: list[Any] = []
        for logical_key, quoted_columns in self._field_columns:
            value = record.get(logical_key)
            for column in quoted_columns:
                columns.append(column)
                values.append(value)
        return columns, values

    def create_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = self.build_sql_payload(payload)
        columns, values = self._build_insert_plan(record)
        columns_sql = ", ".join(columns)
        placeholders_sql = ", ".join(["?"] * len(values))
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                INSERT INTO {self._table} (
                    {columns_sql}
                )
                OUTPUT INSERTED.{self._col_id}
                VALUES ({placeholders_sql})
                """,
                *values,
            )
            inserted_id = int(cursor.fetchone()[0])
            conn.commit()
        return {"id": inserted_id, **record}

    def unfilled_datapart_ids(self, mirror_ids: list[int]) -> list[int]:
        """Read-only: which of `mirror_ids` still have an empty DatapartID
        column — i.e. the downstream MES hasn't confirmed them yet. Used by
        the datapart guard; never writes to this table."""
        if not mirror_ids:
            return []
        placeholders_sql = ", ".join(["?"] * len(mirror_ids))
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT {self._col_id} FROM {self._table}
                WHERE {self._col_id} IN ({placeholders_sql})
                AND ({self._col_datapart_id} IS NULL OR CAST({self._col_datapart_id} AS VARCHAR(64)) = '')
                """,
                *[int(v) for v in mirror_ids],
            )
            rows = cursor.fetchall()
        return [int(row[0]) for row in rows]

    def delete_result(self, mirror_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"DELETE FROM {self._table} WHERE {self._col_id} = ?",
                int(mirror_id),
            )
            deleted = int(cursor.rowcount or 0) > 0
            conn.commit()
        return deleted
