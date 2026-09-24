"""Read-only access to the plant MES's `PI_OkDataPart` table (Postgres).

Owned by another system, populated upstream (at production/molding time).
This app never writes to it — no auto-create, no seeding, fails loudly if
the table doesn't exist."""
from __future__ import annotations

from typing import Any

from backend.app.core.config import AppConfig
from backend.app.repositories.postgres._base import PostgresRepositoryBase


def _quote_ident(name: str) -> str:
    return ".".join(f'"{part}"' for part in str(name).split("."))


class PostgresOkDataPartRepository(PostgresRepositoryBase):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._table = _quote_ident(config.okdatapart_table)
        self._col_no = _quote_ident(config.okdatapart_col_no)
        self._col_nama_part = _quote_ident(config.okdatapart_col_nama_part)
        self._col_prod_date = _quote_ident(config.okdatapart_col_prod_date)
        self._col_id_data_part = _quote_ident(config.okdatapart_col_id_data_part)
        self._col_qty_box = _quote_ident(config.okdatapart_col_qty_box)
        self._col_mpid = _quote_ident(config.okdatapart_col_mpid)
        self._col_mcid = _quote_ident(config.okdatapart_col_mcid)
        self._col_station = _quote_ident(config.okdatapart_col_station)

    def _select_columns(self) -> str:
        return (
            f"{self._col_no} AS no_pk, "
            f"{self._col_nama_part} AS nama_part, "
            f"{self._col_prod_date} AS prod_date, "
            f"{self._col_id_data_part} AS id_data_part, "
            f"{self._col_qty_box} AS qty_box, "
            f"{self._col_mpid} AS mpid, "
            f"{self._col_mcid} AS mcid, "
            f"{self._col_station} AS station"
        )

    def _row_to_record(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        return {
            "No": int(row["no_pk"]),
            "NamaPart": str(row.get("nama_part") or ""),
            "ProdDate": row.get("prod_date"),
            "IDDataPart": str(row.get("id_data_part") or ""),
            "QtyBox": row.get("qty_box"),
            "MPID": str(row.get("mpid") or ""),
            "MCID": str(row.get("mcid") or ""),
            "Station": str(row.get("station") or ""),
        }

    def get_by_id_data_part(self, id_data_part: str) -> dict[str, Any] | None:
        normalized = str(id_data_part or "").strip()
        if not normalized:
            return None
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"SELECT {self._select_columns()} FROM {self._table} "
                    f"WHERE {self._col_id_data_part} = %s LIMIT 1",
                    (normalized,),
                )
                row = cursor.fetchone()
        return self._row_to_record(row)
