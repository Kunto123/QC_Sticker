from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.app.core.config import AppConfig


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _format_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    text = str(value).strip()
    return text or None


class PostgresRepositoryBase:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._validate_config()

    def _validate_config(self) -> None:
        missing = []
        if not self._config.postgresql_host:
            missing.append("POSTGRESQL_HOST")
        if not self._config.postgresql_database:
            missing.append("POSTGRESQL_DATABASE")
        if not self._config.postgresql_username:
            missing.append("POSTGRESQL_USERNAME")
        if not self._config.postgresql_password:
            missing.append("POSTGRESQL_PASSWORD")
        if missing:
            raise ValueError("PostgreSQL configuration is incomplete: " + ", ".join(missing))

    def _connect(self):
        from psycopg import connect
        from psycopg.rows import dict_row

        return connect(
            host=self._config.postgresql_host,
            port=self._config.postgresql_port,
            dbname=self._config.postgresql_database,
            user=self._config.postgresql_username,
            password=self._config.postgresql_password,
            connect_timeout=5,
            row_factory=dict_row,
            sslmode=self._config.postgresql_sslmode,
            options=f"-c search_path={self._config.postgresql_schema}",
        )