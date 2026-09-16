from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

from backend.app.core.config import AppConfig
from backend.app.core.security import TokenRecord
from shared.contracts.auth import UserInfo
from shared.contracts.enums import UserRole


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_dt(value: str | datetime | None) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            pass
    return _utcnow()


class SqlServerTokenStore:
    TABLE_NAME = "dbo.qc_user_sessions"

    def __init__(self, config: AppConfig, ttl_seconds: int = 86_400) -> None:
        self._config = config
        self.ttl_seconds = max(60, int(ttl_seconds or 86_400))
        self._ensure_schema()

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
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        session_id NVARCHAR(64) NOT NULL,
                        token_hash NVARCHAR(64) NOT NULL,
                        user_id INT NOT NULL,
                        username NVARCHAR(100) NOT NULL,
                        role NVARCHAR(32) NOT NULL,
                        issued_at DATETIMEOFFSET NOT NULL,
                        expires_at DATETIMEOFFSET NOT NULL,
                        last_seen_at DATETIMEOFFSET NOT NULL,
                        ip_address NVARCHAR(64) NULL,
                        client_name NVARCHAR(120) NULL,
                        user_agent NVARCHAR(512) NULL
                    )
                END
                """
            )
            for column_name, sql_type in (
                ("session_id", "NVARCHAR(64) NOT NULL"),
                ("token_hash", "NVARCHAR(64) NOT NULL"),
                ("user_id", "INT NOT NULL"),
                ("username", "NVARCHAR(100) NOT NULL"),
                ("role", "NVARCHAR(32) NOT NULL"),
                ("issued_at", "DATETIMEOFFSET NOT NULL"),
                ("expires_at", "DATETIMEOFFSET NOT NULL"),
                ("last_seen_at", "DATETIMEOFFSET NOT NULL"),
                ("ip_address", "NVARCHAR(64) NULL"),
                ("client_name", "NVARCHAR(120) NULL"),
                ("user_agent", "NVARCHAR(512) NULL"),
            ):
                cursor.execute(
                    f"""
                    IF COL_LENGTH('{self.TABLE_NAME}', '{column_name}') IS NULL
                    BEGIN
                        ALTER TABLE {self.TABLE_NAME} ADD {column_name} {sql_type}
                    END
                    """
                )
            cursor.execute(
                f"""
                IF NOT EXISTS (
                    SELECT 1
                    FROM sys.indexes
                    WHERE name = 'UX_qc_user_sessions_session_id'
                      AND object_id = OBJECT_ID('{self.TABLE_NAME}')
                )
                BEGIN
                    CREATE UNIQUE INDEX UX_qc_user_sessions_session_id
                    ON {self.TABLE_NAME} (session_id)
                END
                """
            )
            cursor.execute(
                f"""
                IF NOT EXISTS (
                    SELECT 1
                    FROM sys.indexes
                    WHERE name = 'UX_qc_user_sessions_token_hash'
                      AND object_id = OBJECT_ID('{self.TABLE_NAME}')
                )
                BEGIN
                    CREATE UNIQUE INDEX UX_qc_user_sessions_token_hash
                    ON {self.TABLE_NAME} (token_hash)
                END
                """
            )
            cursor.execute(
                f"""
                IF NOT EXISTS (
                    SELECT 1
                    FROM sys.indexes
                    WHERE name = 'IX_qc_user_sessions_user_id'
                      AND object_id = OBJECT_ID('{self.TABLE_NAME}')
                )
                BEGIN
                    CREATE INDEX IX_qc_user_sessions_user_id
                    ON {self.TABLE_NAME} (user_id)
                END
                """
            )
            conn.commit()

    def _token_hash(self, token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _purge_expired_with_cursor(self, cursor, *, now: datetime | None = None) -> int:
        current = (now or _utcnow()).isoformat()
        cursor.execute(f"SELECT COUNT(1) FROM {self.TABLE_NAME} WHERE expires_at <= ?", current)
        count = int(cursor.fetchone()[0] or 0)
        if count:
            cursor.execute(f"DELETE FROM {self.TABLE_NAME} WHERE expires_at <= ?", current)
        return count

    def _row_to_record(self, row, *, token: str = "") -> TokenRecord:
        return TokenRecord(
            session_id=str(row.session_id),
            token=token,
            user=UserInfo(
                id=int(row.user_id),
                username=str(row.username),
                role=UserRole(str(row.role)),
            ),
            issued_at=_parse_dt(row.issued_at),
            expires_at=_parse_dt(row.expires_at),
            last_seen_at=_parse_dt(row.last_seen_at),
            ip_address=str(row.ip_address) if row.ip_address is not None else None,
            client_name=str(row.client_name) if row.client_name is not None else None,
            user_agent=str(row.user_agent) if row.user_agent is not None else None,
        )

    def issue(
        self,
        user: UserInfo,
        *,
        ttl_seconds: int | None = None,
        ip_address: str | None = None,
        client_name: str | None = None,
        user_agent: str | None = None,
    ) -> TokenRecord:
        token = secrets.token_urlsafe(32)
        session_id = secrets.token_hex(8)
        now = _utcnow()
        ttl = max(60, int(ttl_seconds or self.ttl_seconds))
        record = TokenRecord(
            session_id=session_id,
            token=token,
            user=user,
            issued_at=now,
            expires_at=now + timedelta(seconds=ttl),
            last_seen_at=now,
            ip_address=(ip_address or "").strip() or None,
            client_name=(client_name or "").strip() or None,
            user_agent=(user_agent or "").strip() or None,
        )
        with self._connect() as conn:
            cursor = conn.cursor()
            self._purge_expired_with_cursor(cursor, now=now)
            cursor.execute(
                f"""
                INSERT INTO {self.TABLE_NAME} (
                    session_id,
                    token_hash,
                    user_id,
                    username,
                    role,
                    issued_at,
                    expires_at,
                    last_seen_at,
                    ip_address,
                    client_name,
                    user_agent
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                record.session_id,
                self._token_hash(token),
                int(user.id),
                user.username,
                user.role.value,
                record.issued_at.isoformat(),
                record.expires_at.isoformat(),
                record.last_seen_at.isoformat(),
                record.ip_address,
                record.client_name,
                record.user_agent,
            )
            conn.commit()
        return record

    def get(self, token: str):
        record = self.get_record(token)
        return record.user if record else None

    def get_record(self, token: str, *, touch: bool = True) -> TokenRecord | None:
        normalized = str(token or "").strip()
        if not normalized:
            return None
        now = _utcnow()
        with self._connect() as conn:
            cursor = conn.cursor()
            self._purge_expired_with_cursor(cursor, now=now)
            cursor.execute(
                f"""
                SELECT TOP 1
                    session_id,
                    user_id,
                    username,
                    role,
                    CONVERT(NVARCHAR(33), issued_at, 127) AS issued_at,
                    CONVERT(NVARCHAR(33), expires_at, 127) AS expires_at,
                    CONVERT(NVARCHAR(33), last_seen_at, 127) AS last_seen_at,
                    ip_address,
                    client_name,
                    user_agent
                FROM {self.TABLE_NAME}
                WHERE token_hash = ?
                """,
                self._token_hash(normalized),
            )
            row = cursor.fetchone()
            if row is None:
                conn.commit()
                return None
            if touch:
                cursor.execute(
                    f"""
                    UPDATE {self.TABLE_NAME}
                    SET last_seen_at = ?
                    WHERE token_hash = ?
                    """,
                    now.isoformat(),
                    self._token_hash(normalized),
                )
            conn.commit()
        record = self._row_to_record(row, token=normalized)
        if touch:
            record.last_seen_at = now
        return record

    def revoke(self, token: str) -> bool:
        normalized = str(token or "").strip()
        if not normalized:
            return False
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(1) FROM {self.TABLE_NAME} WHERE token_hash = ?", self._token_hash(normalized))
            count = int(cursor.fetchone()[0] or 0)
            if count:
                cursor.execute(f"DELETE FROM {self.TABLE_NAME} WHERE token_hash = ?", self._token_hash(normalized))
            conn.commit()
        return bool(count)

    def revoke_user(self, user_id: int, *, exclude_token: str | None = None) -> int:
        exclude_hash = self._token_hash(exclude_token.strip()) if exclude_token and exclude_token.strip() else None
        with self._connect() as conn:
            cursor = conn.cursor()
            self._purge_expired_with_cursor(cursor)
            if exclude_hash:
                cursor.execute(
                    f"SELECT COUNT(1) FROM {self.TABLE_NAME} WHERE user_id = ? AND token_hash <> ?",
                    int(user_id),
                    exclude_hash,
                )
                count = int(cursor.fetchone()[0] or 0)
                if count:
                    cursor.execute(
                        f"DELETE FROM {self.TABLE_NAME} WHERE user_id = ? AND token_hash <> ?",
                        int(user_id),
                        exclude_hash,
                    )
            else:
                cursor.execute(f"SELECT COUNT(1) FROM {self.TABLE_NAME} WHERE user_id = ?", int(user_id))
                count = int(cursor.fetchone()[0] or 0)
                if count:
                    cursor.execute(f"DELETE FROM {self.TABLE_NAME} WHERE user_id = ?", int(user_id))
            conn.commit()
        return count

    def list_user_sessions(self, user_id: int) -> list[TokenRecord]:
        with self._connect() as conn:
            cursor = conn.cursor()
            self._purge_expired_with_cursor(cursor)
            cursor.execute(
                f"""
                SELECT
                    session_id,
                    user_id,
                    username,
                    role,
                    CONVERT(NVARCHAR(33), issued_at, 127) AS issued_at,
                    CONVERT(NVARCHAR(33), expires_at, 127) AS expires_at,
                    CONVERT(NVARCHAR(33), last_seen_at, 127) AS last_seen_at,
                    ip_address,
                    client_name,
                    user_agent
                FROM {self.TABLE_NAME}
                WHERE user_id = ?
                ORDER BY issued_at DESC
                """,
                int(user_id),
            )
            rows = cursor.fetchall()
            conn.commit()
        return [self._row_to_record(row) for row in rows]

    def purge_expired(self) -> int:
        with self._connect() as conn:
            cursor = conn.cursor()
            count = self._purge_expired_with_cursor(cursor)
            conn.commit()
        return count
