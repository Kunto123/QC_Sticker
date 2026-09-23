from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.app.core.config import AppConfig


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class SqlServerInspectionMirrorRepository:
    """Pushes inspection results to `dbo.qc_inspection_push`, which must
    already exist — this repository never creates or alters it."""

    TABLE_NAME = "dbo.qc_inspection_push"

    def __init__(self, config: AppConfig) -> None:
        self._config = config

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
        """Map a local inspection result to the agreed SQL Server push contract.

        Contract (fixed — do not change without downstream coordination):
            PartName    <- expected_class (fallback: part_name)
            DateCheckMC <- inspected_at
            MPCheck     <- mp_check  (operator username)
            Data1       <- part_ready_match_ratio  (confidence part ready)
            Data2       <- sticker_confidence       (confidence sticker)
            Line        <- line_id

        All other fields are stored locally only.
        """
        return {
            "PartName": payload.get("expected_class") or payload.get("part_name"),
            "DateCheckMC": payload.get("inspected_at") or _utcnow_iso(),
            "MPCheck": payload.get("mp_check"),
            "Data1": payload.get("part_ready_match_ratio"),   # confidence part ready
            "Data2": payload.get("sticker_confidence"),        # confidence sticker
            "Line": payload.get("line_id"),
        }

    def create_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = self.build_sql_payload(payload)
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                INSERT INTO {self.TABLE_NAME} (
                    PartName,
                    DateCheckMC,
                    MPCheck,
                    Data1,
                    Data2,
                    Line
                )
                OUTPUT INSERTED.id
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                record.get("PartName"),
                record.get("DateCheckMC"),
                record.get("MPCheck"),
                record.get("Data1"),
                record.get("Data2"),
                record.get("Line"),
            )
            inserted_id = int(cursor.fetchone()[0])
            conn.commit()
        return {"id": inserted_id, **record}

    def delete_result(self, mirror_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"DELETE FROM {self.TABLE_NAME} WHERE id = ?",
                int(mirror_id),
            )
            deleted = int(cursor.rowcount or 0) > 0
            conn.commit()
        return deleted
