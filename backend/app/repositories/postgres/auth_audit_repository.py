from __future__ import annotations

from backend.app.core.config import AppConfig
from backend.app.repositories.postgres._base import PostgresRepositoryBase, _format_datetime, _utcnow_iso


class PostgresAuthAuditRepository(PostgresRepositoryBase):
    TABLE_NAME = "qc_auth_audit"

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
                        event_type TEXT NOT NULL,
                        user_id INT NULL,
                        username TEXT NULL,
                        session_id TEXT NULL,
                        actor_id INT NULL,
                        actor_username TEXT NULL,
                        ip_address TEXT NULL,
                        client_name TEXT NULL,
                        details TEXT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS IX_{self.TABLE_NAME}_user_id
                    ON {self.TABLE_NAME} (user_id)
                    """
                )
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS IX_{self.TABLE_NAME}_created_at
                    ON {self.TABLE_NAME} (created_at DESC)
                    """
                )
            conn.commit()

    def log(
        self,
        event_type: str,
        *,
        user_id: int | None = None,
        username: str | None = None,
        session_id: str | None = None,
        actor_id: int | None = None,
        actor_username: str | None = None,
        ip_address: str | None = None,
        client_name: str | None = None,
        details: str | None = None,
    ) -> None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {self.TABLE_NAME} (
                        event_type,
                        user_id,
                        username,
                        session_id,
                        actor_id,
                        actor_username,
                        ip_address,
                        client_name,
                        details,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(event_type),
                        int(user_id) if user_id is not None else None,
                        str(username) if username is not None else None,
                        str(session_id) if session_id is not None else None,
                        int(actor_id) if actor_id is not None else None,
                        str(actor_username) if actor_username is not None else None,
                        str(ip_address) if ip_address is not None else None,
                        str(client_name) if client_name is not None else None,
                        str(details) if details is not None else None,
                        _utcnow_iso(),
                    ),
                )
            conn.commit()

    def list_recent(self, *, limit: int = 200, user_id: int | None = None) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                if user_id is not None:
                    cursor.execute(
                        f"""
                        SELECT
                            id,
                            event_type,
                            user_id,
                            username,
                            session_id,
                            actor_id,
                            actor_username,
                            ip_address,
                            client_name,
                            details,
                            created_at
                        FROM {self.TABLE_NAME}
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                            (int(user_id), int(limit)),
                    )
                else:
                    cursor.execute(
                        f"""
                        SELECT
                            id,
                            event_type,
                            user_id,
                            username,
                            session_id,
                            actor_id,
                            actor_username,
                            ip_address,
                            client_name,
                            details,
                            created_at
                        FROM {self.TABLE_NAME}
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                            (int(limit),),
                    )
                rows = cursor.fetchall()
        return [
            {
                "id": int(row["id"]),
                "event_type": str(row["event_type"]),
                "user_id": int(row["user_id"]) if row["user_id"] is not None else None,
                "username": str(row["username"]) if row["username"] is not None else None,
                "session_id": str(row["session_id"]) if row["session_id"] is not None else None,
                "actor_id": int(row["actor_id"]) if row["actor_id"] is not None else None,
                "actor_username": str(row["actor_username"]) if row["actor_username"] is not None else None,
                "ip_address": str(row["ip_address"]) if row["ip_address"] is not None else None,
                "client_name": str(row["client_name"]) if row["client_name"] is not None else None,
                "details": str(row["details"]) if row["details"] is not None else None,
                "created_at": _format_datetime(row["created_at"]),
            }
            for row in rows
        ]