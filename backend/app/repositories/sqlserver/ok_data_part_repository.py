"""Read-only access to the plant MES's `PI_OkDataPart` table (SQL Server).

See the Postgres sibling (`postgres/ok_data_part_repository.py`) for the full
rationale — same contract, pyodbc instead of psycopg."""
from __future__ import annotations

from typing import Any

from backend.app.core.config import AppConfig


def _quote_ident(name: str) -> str:
    return ".".join(f"[{part}]" for part in str(name).split("."))


class SqlServerOkDataPartRepository:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._table = _quote_ident(config.okdatapart_table)
        self._col_no = _quote_ident(config.okdatapart_col_no)
        self._col_nama_part = _quote_ident(config.okdatapart_col_nama_part)
        self._col_prod_date = _quote_ident(config.okdatapart_col_prod_date)
        self._col_id_data_part = _quote_ident(config.okdatapart_col_id_data_part)
        self._col_qty_box = _quote_ident(config.okdatapart_col_qty_box)
        self._col_mpid = _quote_ident(config.okdatapart_col_mpid)
        self._col_mcid = _quote_ident(config.okdatapart_col_mcid)
        self._col_station = _quote_ident(config.okdatapart_col_station)

    def _connect(self):
        import pyodbc

        connection_string = (
            f"DRIVER={{{self._config.sql_driver}}};"
            f"SERVER={self._config.sql_server};"
            f"DATABASE={self._config.sql_database};"
            f"UID={self._config.sql_username};"
            f"PWD={self._config.sql_password};"
            "TrustServerCertificate=yes;"
        )
        return pyodbc.connect(connection_string, timeout=5)

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

    def _row_to_record(self, row) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "No": int(row.no_pk),
            "NamaPart": str(row.nama_part) if row.nama_part is not None else "",
            "ProdDate": row.prod_date,
            "IDDataPart": str(row.id_data_part) if row.id_data_part is not None else "",
            "QtyBox": row.qty_box,
            "MPID": str(row.mpid) if row.mpid is not None else "",
            "MCID": str(row.mcid) if row.mcid is not None else "",
            "Station": str(row.station) if row.station is not None else "",
        }

    def get_by_id_data_part(self, id_data_part: str) -> dict[str, Any] | None:
        normalized = str(id_data_part or "").strip()
        if not normalized:
            return None
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT TOP 1 {self._select_columns()} FROM {self._table} "
                f"WHERE {self._col_id_data_part} = ?",
                normalized,
            )
            row = cursor.fetchone()
        return self._row_to_record(row)
