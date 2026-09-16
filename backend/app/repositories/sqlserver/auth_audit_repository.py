from __future__ import annotations

from datetime import UTC, datetime

from backend.app.core.config import AppConfig


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class SqlServerAuthAuditRepository:
    """Auth audit log backed by dbo.qc_auth_audit in SQL Server.

    Schema is created / migrated on startup via :meth:`_ensure_schema`.
    Writes are fire-and-forget in the sense that callers never need to
    wait – but errors ARE raised so the container can decide whether to
    fall back to the local repo.
    """

    TABLE_NAME = "dbo.qc_auth_audit"

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                IF OBJECT_ID('{self.TABLE_NAME}', 'U') IS NULL
                BEGIN
                    CREATE TABLE {self.TABLE_NAME} (
                        id          INT IDENTITY(1,1) PRIMARY KEY,
                        event_type  NVARCHAR(64)  NOT NULL,
                        user_id     INT           NULL,
                        username    NVARCHAR(100) NULL,
                        session_id  NVARCHAR(64)  NULL,
                        actor_id    INT           NULL,
                        actor_username NVARCHAR(100) NULL,
                        ip_address  NVARCHAR(64)  NULL,
                        client_name NVARCHAR(120) NULL,
                        details     NVARCHAR(512) NULL,
                        created_at  DATETIMEOFFSET NOT NULL
                            CONSTRAINT DF_qc_auth_audit_created_at
                            DEFAULT SYSUTCDATETIME()
                    )
                END
                """
            )
            # Additive column migration so existing deployments are safe
            for col_name, col_def in (
                ("actor_id",       "INT NULL"),
                ("actor_username", "NVARCHAR(100) NULL"),
            ):
                cursor.execute(
                    f"""
                    IF COL_LENGTH('{self.TABLE_NAME}', '{col_name}') IS NULL
                    BEGIN
                        ALTER TABLE {self.TABLE_NAME} ADD {col_name} {col_def}
                    END
                    """
                )
            # Indexes
            for idx_name, idx_ddl in (
                (
                    "IX_qc_auth_audit_user_id",
                    f"CREATE INDEX IX_qc_auth_audit_user_id ON {self.TABLE_NAME} (user_id)",
                ),
                (
                    "IX_qc_auth_audit_created_at",
                    f"CREATE INDEX IX_qc_auth_audit_created_at ON {self.TABLE_NAME} (created_at DESC)",
                ),
            ):
                cursor.execute(
                    f"""
                    IF NOT EXISTS (
                        SELECT 1 FROM sys.indexes
                        WHERE name = '{idx_name}'
                          AND object_id = OBJECT_ID('{self.TABLE_NAME}')
                    )
                    BEGIN
                        {idx_ddl}
                    END
                    """
                )
            conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

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
            cursor = conn.cursor()
            cursor.execute(
                f"""
                INSERT INTO {self.TABLE_NAME} (
                    event_type, user_id, username, session_id,
                    actor_id, actor_username,
                    ip_address, client_name, details, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
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
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def list_recent(self, *, limit: int = 200, user_id: int | None = None) -> list[dict]:
        """Return up to *limit* recent events, newest first."""
        with self._connect() as conn:
            cursor = conn.cursor()
            if user_id is not None:
                cursor.execute(
                    f"""
                    SELECT TOP (?)
                        id, event_type, user_id, username, session_id,
                        actor_id, actor_username,
                        ip_address, client_name, details,
                        CONVERT(NVARCHAR(33), created_at, 127) AS created_at
                    FROM {self.TABLE_NAME}
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    """,
                    int(limit),
                    int(user_id),
                )
            else:
                cursor.execute(
                    f"""
                    SELECT TOP (?)
                        id, event_type, user_id, username, session_id,
                        actor_id, actor_username,
                        ip_address, client_name, details,
                        CONVERT(NVARCHAR(33), created_at, 127) AS created_at
                    FROM {self.TABLE_NAME}
                    ORDER BY created_at DESC
                    """,
                    int(limit),
                )
            rows = cursor.fetchall()
        return [
            {
                "id": int(row.id),
                "event_type": str(row.event_type),
                "user_id": int(row.user_id) if row.user_id is not None else None,
                "username": str(row.username) if row.username is not None else None,
                "session_id": str(row.session_id) if row.session_id is not None else None,
                "actor_id": int(row.actor_id) if row.actor_id is not None else None,
                "actor_username": str(row.actor_username) if row.actor_username is not None else None,
                "ip_address": str(row.ip_address) if row.ip_address is not None else None,
                "client_name": str(row.client_name) if row.client_name is not None else None,
                "details": str(row.details) if row.details is not None else None,
                "created_at": str(row.created_at),
            }
            for row in rows
        ]
