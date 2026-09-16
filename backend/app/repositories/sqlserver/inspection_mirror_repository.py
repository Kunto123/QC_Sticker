from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.app.core.config import AppConfig


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class SqlServerInspectionMirrorRepository:
    TABLE_NAME = "dbo.qc_inspection_push"

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._ensure_schema()

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

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                IF OBJECT_ID('{self.TABLE_NAME}', 'U') IS NULL
                BEGIN
                    CREATE TABLE {self.TABLE_NAME} (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        PartName NVARCHAR(150) NULL,
                        DateCheckMC DATETIMEOFFSET NOT NULL CONSTRAINT DF_qc_inspection_push_DateCheckMC DEFAULT SYSUTCDATETIME(),
                        MPCheck NVARCHAR(100) NULL,
                        Data1 FLOAT NULL,
                        Data2 FLOAT NULL,
                        Line NVARCHAR(100) NULL
                    )
                END
                """
            )
            for column_name, sql_type in (
                ("PartName", "NVARCHAR(150) NULL"),
                ("DateCheckMC", "DATETIMEOFFSET NOT NULL CONSTRAINT DF_qc_inspection_results_DateCheckMC DEFAULT SYSUTCDATETIME()"),
                ("MPCheck", "NVARCHAR(100) NULL"),
                ("Data1", "FLOAT NULL"),
                ("Data2", "FLOAT NULL"),
                ("Line", "NVARCHAR(100) NULL"),
            ):
                cursor.execute(
                    f"""
                    IF COL_LENGTH('{self.TABLE_NAME}', '{column_name}') IS NULL
                    BEGIN
                        ALTER TABLE {self.TABLE_NAME} ADD {column_name} {sql_type}
                    END
                    """
                )
            conn.commit()

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
