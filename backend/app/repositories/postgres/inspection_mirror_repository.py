from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.app.core.config import AppConfig
from backend.app.repositories.postgres._base import PostgresRepositoryBase, _utcnow_iso


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class PostgresInspectionMirrorRepository(PostgresRepositoryBase):
    TABLE_NAME = "qc_inspection_push"

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                        id BIGSERIAL PRIMARY KEY,
                        PartName TEXT NULL,
                        DateCheckMC TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        MPCheck TEXT NULL,
                        Data1 DOUBLE PRECISION NULL,
                        Data2 DOUBLE PRECISION NULL,
                        Line TEXT NULL
                    )
                    """
                )
            conn.commit()

    @staticmethod
    def build_sql_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "PartName": payload.get("expected_class") or payload.get("part_name"),
            "DateCheckMC": payload.get("inspected_at") or _utcnow(),
            "MPCheck": payload.get("mp_check"),
            "Data1": payload.get("part_ready_match_ratio"),
            "Data2": payload.get("sticker_confidence"),
            "Line": payload.get("line_id"),
        }

    def create_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = self.build_sql_payload(payload)
        with self._connect() as conn:
            with conn.cursor() as cursor:
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
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        record.get("PartName"),
                        record.get("DateCheckMC"),
                        record.get("MPCheck"),
                        record.get("Data1"),
                        record.get("Data2"),
                        record.get("Line"),
                    ),
                )
                inserted_id = int(cursor.fetchone()["id"])
            conn.commit()
        return {"id": inserted_id, **record}

    def delete_result(self, mirror_id: int) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self.TABLE_NAME} WHERE id = %s",
                    (int(mirror_id),),
                )
                deleted = int(cursor.rowcount or 0) > 0
            conn.commit()
        return deleted
