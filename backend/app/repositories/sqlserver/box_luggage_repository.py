"""Writes box-level datapart traceability into the plant MES's
`PI_BoxLuggage` table (SQL Server). See the Postgres sibling
(`postgres/box_luggage_repository.py`) for the full rationale — same
contract, pyodbc instead of psycopg."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.app.core.config import AppConfig

ABANDONED_SENTINEL = "ABANDONED"


def _quote_ident(name: str) -> str:
    return ".".join(f"[{part}]" for part in str(name).split("."))


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SqlServerBoxLuggageRepository:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._table = _quote_ident(config.boxluggage_table)
        self._col_no = _quote_ident(config.boxluggage_col_no)
        self._col_part_name = _quote_ident(config.boxluggage_col_part_name)
        self._col_date_check_mc = _quote_ident(config.boxluggage_col_date_check_mc)
        self._col_mp_check = _quote_ident(config.boxluggage_col_mp_check)
        self._col_date_send_db = _quote_ident(config.boxluggage_col_date_send_db)
        self._col_datapart_id = _quote_ident(config.boxluggage_col_datapart_id)
        self._col_datapart_date = _quote_ident(config.boxluggage_col_datapart_date)
        self._col_mpid_datapart = _quote_ident(config.boxluggage_col_mpid_datapart)
        self._col_data1 = _quote_ident(config.boxluggage_col_data1)
        self._col_data2 = _quote_ident(config.boxluggage_col_data2)
        self._col_line = _quote_ident(config.boxluggage_col_line)
        self._col_datapart_pi = _quote_ident(config.boxluggage_col_datapart_pi)
        self._col_datapart_pi_date = _quote_ident(config.boxluggage_col_datapart_pi_date)

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
        import pyodbc

        inserted: list[dict[str, Any]] | None = None
        for _attempt in range(5):
            try:
                with self._connect() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT ISNULL(MAX({self._col_no}), 0) FROM {self._table} WITH (UPDLOCK, HOLDLOCK)")
                    base = int(cursor.fetchone()[0])
                    new_nos = list(range(base + 1, base + qty + 1))
                    for no in new_nos:
                        cursor.execute(
                            f"""
                            INSERT INTO {self._table} (
                                {self._col_no}, {self._col_part_name}, {self._col_mp_check},
                                {self._col_mpid_datapart}, {self._col_line},
                                {self._col_datapart_pi}, {self._col_datapart_pi_date}
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            no, part_name, mp_check, mpid_data_part, line, datapart_pi, datapart_pi_date,
                        )
                    conn.commit()
                inserted = [{"No": no} for no in new_nos]
                break
            except pyodbc.IntegrityError:
                inserted = None
                continue
        if inserted is None:
            raise ValueError("Failed to open box: could not allocate a free No block after several attempts.")
        return inserted

    def record_ok(self, *, datapart_pi: str, data1: str, data2: str, mp_check: str) -> dict[str, Any] | None:
        now = _utcnow()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                UPDATE TOP (1) {self._table} WITH (ROWLOCK, READPAST)
                SET {self._col_data1} = ?, {self._col_data2} = ?, {self._col_mp_check} = ?,
                    {self._col_date_check_mc} = ?, {self._col_date_send_db} = ?
                OUTPUT INSERTED.{self._col_no}
                WHERE {self._col_datapart_pi} = ? AND {self._col_datapart_id} IS NULL AND {self._col_data1} IS NULL
                """,
                data1, data2, mp_check, now, now, datapart_pi,
            )
            row = cursor.fetchone()
            conn.commit()
        return {"No": int(row[0])} if row else None

    def close_box(self, *, datapart_pi: str, datapart_id: str, datapart_date: Any) -> int:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                UPDATE {self._table}
                SET {self._col_datapart_id} = ?, {self._col_datapart_date} = ?
                WHERE {self._col_datapart_pi} = ? AND {self._col_datapart_id} IS NULL
                """,
                datapart_id, datapart_date, datapart_pi,
            )
            count = int(cursor.rowcount or 0)
            conn.commit()
        return count

    def abandon_box(self, *, datapart_pi: str) -> int:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                UPDATE {self._table}
                SET {self._col_datapart_id} = ?, {self._col_datapart_date} = ?
                WHERE {self._col_datapart_pi} = ? AND {self._col_datapart_id} IS NULL
                """,
                ABANDONED_SENTINEL, _utcnow(), datapart_pi,
            )
            count = int(cursor.rowcount or 0)
            conn.commit()
        return count

    def find_open_box(self, *, part_name: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT TOP 1 {self._col_datapart_pi},
                       COUNT(*) OVER (PARTITION BY {self._col_datapart_pi}) AS qty_goal,
                       SUM(CASE WHEN {self._col_data1} IS NULL THEN 1 ELSE 0 END)
                           OVER (PARTITION BY {self._col_datapart_pi}) AS qty_remaining
                FROM {self._table}
                WHERE {self._col_part_name} = ? AND {self._col_datapart_id} IS NULL
                """,
                part_name,
            )
            row = cursor.fetchone()
        if not row:
            return None
        return {"datapart_pi": row[0], "qty_goal": int(row[1]), "qty_remaining": int(row[2])}
