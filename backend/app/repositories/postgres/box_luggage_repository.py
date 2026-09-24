"""Writes box-level datapart traceability into the plant MES's
`PI_BoxLuggage` table (Postgres). Owned by another system, never
created/altered by this app.

Row model: one row is inserted per unit of the box's goal quantity at
Datapart 1 (all empty on Data1/Data2/DateCheckMC/DateSendDB); each accepted
OK fills exactly one still-empty row (claimed safely under concurrency via
`FOR UPDATE SKIP LOCKED`); Datapart 2 closes all of a box's rows at once.
Rows belonging to the same box are grouped by sharing an identical
`DatapartPI` value (the raw Datapart 1 scan string).
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.app.core.config import AppConfig
from backend.app.repositories.postgres._base import PostgresRepositoryBase, _format_datetime

ABANDONED_SENTINEL = "ABANDONED"


def _quote_ident(name: str) -> str:
    return ".".join(f'"{part}"' for part in str(name).split("."))


def _utcnow() -> datetime:
    return datetime.now(UTC)


class PostgresBoxLuggageRepository(PostgresRepositoryBase):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
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
        from psycopg.errors import UniqueViolation

        inserted: list[dict[str, Any]] | None = None
        for _attempt in range(5):
            try:
                with self._connect() as conn:
                    with conn.cursor() as cursor:
                        # Reserve a contiguous block of `No` and insert all `qty`
                        # rows in a single statement — a partial failure must not
                        # leave an incomplete, un-recoverable box. PostgreSQL
                        # forbids `FOR UPDATE` on an aggregate SELECT, so the base
                        # is computed in a CTE and the retry-on-UniqueViolation
                        # loop below (same pattern as create_user) handles races
                        # instead of an explicit row lock.
                        cursor.execute(
                            f"""
                            WITH base AS (
                                SELECT COALESCE(MAX({self._col_no}), 0) AS base_no FROM {self._table}
                            )
                            INSERT INTO {self._table} (
                                {self._col_no}, {self._col_part_name}, {self._col_mp_check},
                                {self._col_mpid_datapart}, {self._col_line},
                                {self._col_datapart_pi}, {self._col_datapart_pi_date}
                            )
                            SELECT gs, %s, %s, %s, %s, %s, %s
                            FROM base, generate_series(base.base_no + 1, base.base_no + %s) AS gs
                            RETURNING {self._col_no} AS no_val
                            """,
                            (
                                part_name,
                                mp_check,
                                mpid_data_part,
                                line,
                                datapart_pi,
                                _format_datetime(datapart_pi_date),
                                qty,
                            ),
                        )
                        rows = cursor.fetchall()
                    conn.commit()
                inserted = [{"No": int(r["no_val"])} for r in rows]
                break
            except UniqueViolation:
                inserted = None
                continue
        if inserted is None:
            raise ValueError("Failed to open box: could not allocate a free No block after several attempts.")
        return inserted

    def record_ok(self, *, datapart_pi: str, data1: str, data2: str, mp_check: str) -> dict[str, Any] | None:
        now = _format_datetime(_utcnow())
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self._table}
                    SET {self._col_data1} = %s, {self._col_data2} = %s, {self._col_mp_check} = %s,
                        {self._col_date_check_mc} = %s, {self._col_date_send_db} = %s
                    WHERE {self._col_no} = (
                        SELECT {self._col_no} FROM {self._table}
                        WHERE {self._col_datapart_pi} = %s AND {self._col_datapart_id} IS NULL AND {self._col_data1} IS NULL
                        ORDER BY {self._col_no} ASC LIMIT 1 FOR UPDATE SKIP LOCKED
                    )
                    RETURNING {self._col_no} AS no_val
                    """,
                    (data1, data2, mp_check, now, now, datapart_pi),
                )
                row = cursor.fetchone()
            conn.commit()
        return {"No": int(row["no_val"])} if row else None

    def close_box(self, *, datapart_pi: str, datapart_id: str, datapart_date: Any) -> int:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self._table}
                    SET {self._col_datapart_id} = %s, {self._col_datapart_date} = %s
                    WHERE {self._col_datapart_pi} = %s AND {self._col_datapart_id} IS NULL
                    RETURNING {self._col_no}
                    """,
                    (datapart_id, _format_datetime(datapart_date), datapart_pi),
                )
                rows = cursor.fetchall()
            conn.commit()
        return len(rows)

    def abandon_box(self, *, datapart_pi: str) -> int:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self._table}
                    SET {self._col_datapart_id} = %s, {self._col_datapart_date} = %s
                    WHERE {self._col_datapart_pi} = %s AND {self._col_datapart_id} IS NULL
                    RETURNING {self._col_no}
                    """,
                    (ABANDONED_SENTINEL, _format_datetime(_utcnow()), datapart_pi),
                )
                rows = cursor.fetchall()
            conn.commit()
        return len(rows)

    def find_open_box(self, *, part_name: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT {self._col_datapart_pi} AS datapart_pi,
                           COUNT(*) AS qty_goal,
                           COUNT(*) FILTER (WHERE {self._col_data1} IS NULL) AS qty_remaining
                    FROM {self._table}
                    WHERE {self._col_part_name} = %s AND {self._col_datapart_id} IS NULL
                    GROUP BY {self._col_datapart_pi}
                    LIMIT 1
                    """,
                    (part_name,),
                )
                row = cursor.fetchone()
        if not row:
            return None
        return {
            "datapart_pi": row["datapart_pi"],
            "qty_goal": int(row["qty_goal"]),
            "qty_remaining": int(row["qty_remaining"]),
        }
