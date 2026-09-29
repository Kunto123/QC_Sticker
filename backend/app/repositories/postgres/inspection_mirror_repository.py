from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from backend.app.core.config import AppConfig
from backend.app.repositories.postgres._base import PostgresRepositoryBase


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _quote_ident(name: str) -> str:
    return ".".join(f'"{part}"' for part in str(name).split("."))


_MEMBER_ID_CODE_PATTERN = re.compile(r"\d{4}")


def _mp_check_code(mp_check: Any) -> str | None:
    """Member_ID formatnya seperti "ID 9101 PUTRA" — cuma kode 4-digit di
    tengahnya yang di-push ke MPCheck, bukan string lengkapnya. Fallback ke
    nilai aslinya kalau tidak ketemu pola 4-digit, supaya Member_ID dengan
    bentuk berbeda tetap push sesuatu, bukan kosong."""
    text = str(mp_check or "")
    match = _MEMBER_ID_CODE_PATTERN.search(text)
    return match.group(0) if match else (text or None)


def _line_last_char(line_id: Any) -> str | None:
    """identity.line itu kode line seperti "GB3" atau "A01" — cuma karakter
    terakhirnya yang di-push ke kolom Line."""
    text = str(line_id or "").strip()
    return text[-1] if text else None


def _as_percentage(ratio: Any) -> str | None:
    """Data1/Data2 disimpan lokal sebagai rasio confidence 0..1 tapi di-push
    ke SQL sebagai string persentase bilangan bulat, mis. 0.9326 -> "93%"."""
    if ratio is None:
        return None
    try:
        return f"{round(float(ratio) * 100.0)}%"
    except (TypeError, ValueError):
        return None


class PostgresInspectionMirrorRepository(PostgresRepositoryBase):
    """Push hasil inspeksi ke tabel push eksternal milik MES pabrik /
    sistem reporting, bukan app ini — nama tabel dan kolom spesifik per
    deployment (`QC_SUITE_INSPECTION_*` di core/config.py, default
    `qc_inspection_push`). Repository ini tidak pernah membuat atau mengubahnya."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._table = _quote_ident(config.inspection_push_table)
        self._col_id = _quote_ident(config.inspection_push_col_id)
        # Tiap field logis boleh dipetakan ke lebih dari satu kolom fisik
        # (tabel tujuan yang menduplikasi nilai yang sama di beberapa
        # kolom) — tiap kolom di list dapat nilai yang sama saat insert.
        self._field_columns: list[tuple[str, list[str]]] = [
            ("PartName", [_quote_ident(c) for c in config.inspection_push_col_part_name]),
            ("DateCheckMC", [_quote_ident(c) for c in config.inspection_push_col_date_check_mc]),
            ("MPCheck", [_quote_ident(c) for c in config.inspection_push_col_mp_check]),
            ("Data1", [_quote_ident(c) for c in config.inspection_push_col_data1]),
            ("Data2", [_quote_ident(c) for c in config.inspection_push_col_data2]),
            ("Line", [_quote_ident(c) for c in config.inspection_push_col_line]),
        ]

    @staticmethod
    def build_sql_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            # Nama template itu sendiri ("Preset Name" di Admin -> Templates),
            # bukan label ML expected_class — admin/expected_class cuma
            # fallback kalau nama template entah kenapa kosong.
            "PartName": payload.get("template_name") or payload.get("expected_class") or payload.get("part_name"),
            "DateCheckMC": payload.get("inspected_at") or _utcnow(),
            "MPCheck": _mp_check_code(payload.get("mp_check")),
            "Data1": _as_percentage(payload.get("part_ready_match_ratio")),
            "Data2": _as_percentage(payload.get("sticker_confidence")),
            "Line": _line_last_char(payload.get("line_id")),
        }

    def _build_insert_plan(self, record: dict[str, Any]) -> tuple[list[str], list[Any]]:
        """Ratakan `self._field_columns` terhadap `record` jadi list paralel
        (kolom ter-quote, nilai) — menduplikasi nilai satu field ke tiap
        kolom fisik yang dikonfigurasi untuknya. Dipisah dari `create_result`
        supaya bisa di-unit-test tanpa koneksi DB nyata."""
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
        placeholders_sql = ", ".join(["%s"] * len(values))
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {self._table} (
                        {columns_sql}
                    )
                    VALUES ({placeholders_sql})
                    RETURNING {self._col_id} AS id
                    """,
                    tuple(values),
                )
                inserted_id = int(cursor.fetchone()["id"])
            conn.commit()
        return {"id": inserted_id, **record}

    def delete_result(self, mirror_id: int) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self._table} WHERE {self._col_id} = %s",
                    (int(mirror_id),),
                )
                deleted = int(cursor.rowcount or 0) > 0
            conn.commit()
        return deleted
