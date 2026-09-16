#!/usr/bin/env python3
"""Drop legacy relational auth audit/session tables.

By default this is a dry run. Pass ``--execute`` to actually drop tables.
The application no longer uses these tables; auth audit is local JSONL and
auth sessions are in-memory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=False)

from backend.app.core.config import AppConfig  # noqa: E402


SQLSERVER_TABLES = ("dbo.qc_auth_audit", "dbo.qc_user_sessions")
POSTGRES_TABLES = ("qc_auth_audit", "qc_user_sessions")


def _resolve_backend(config: AppConfig, requested: str) -> str:
    if requested != "auto":
        return requested
    backend = config.database_backend
    if backend not in {"sqlserver", "postgresql"}:
        raise SystemExit(
            "No relational backend is active. Pass --backend sqlserver|postgresql "
            "if you want to target one explicitly."
        )
    return backend


def _drop_sqlserver(config: AppConfig, *, execute: bool) -> None:
    print("[drop-auth-tables] target backend: sqlserver")
    print(f"[drop-auth-tables] mode: {'execute' if execute else 'dry-run'}")
    if not execute:
        for table_name in SQLSERVER_TABLES:
            print(f"[drop-auth-tables] would drop {table_name}")
        return

    import pyodbc

    connection_string = (
        f"DRIVER={{{config.sql_driver}}};"
        f"SERVER={config.sql_server};"
        f"DATABASE={config.sql_database};"
        f"UID={config.sql_username};"
        f"PWD={config.sql_password};"
        "TrustServerCertificate=yes;"
    )
    with pyodbc.connect(connection_string, timeout=5) as conn:
        cursor = conn.cursor()
        for table_name in SQLSERVER_TABLES:
            print(f"[drop-auth-tables] dropping {table_name}")
            cursor.execute(
                f"""
                IF OBJECT_ID('{table_name}', 'U') IS NOT NULL
                BEGIN
                    DROP TABLE {table_name}
                END
                """
            )
        conn.commit()


def _drop_postgresql(config: AppConfig, *, execute: bool) -> None:
    schema = config.postgresql_schema or "public"
    print("[drop-auth-tables] target backend: postgresql")
    print(f"[drop-auth-tables] schema: {schema}")
    print(f"[drop-auth-tables] mode: {'execute' if execute else 'dry-run'}")
    if not execute:
        for table_name in POSTGRES_TABLES:
            print(f"[drop-auth-tables] would drop {schema}.{table_name}")
        return

    from psycopg import connect, sql

    with connect(
        host=config.postgresql_host,
        port=config.postgresql_port,
        dbname=config.postgresql_database,
        user=config.postgresql_username,
        password=config.postgresql_password,
        connect_timeout=5,
        sslmode=config.postgresql_sslmode,
    ) as conn:
        with conn.cursor() as cursor:
            for table_name in POSTGRES_TABLES:
                qualified = sql.Identifier(schema, table_name)
                print(f"[drop-auth-tables] dropping {schema}.{table_name}")
                cursor.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(qualified)
                )
        conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Drop legacy qc_auth_audit and qc_user_sessions tables."
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "sqlserver", "postgresql"),
        default="auto",
        help="Relational backend to target. Default: auto from environment.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually drop the tables. Without this flag the script is dry-run only.",
    )
    args = parser.parse_args()

    config = AppConfig()
    backend = _resolve_backend(config, args.backend)
    if backend == "sqlserver":
        _drop_sqlserver(config, execute=bool(args.execute))
    else:
        _drop_postgresql(config, execute=bool(args.execute))

    if not args.execute:
        print("[drop-auth-tables] dry-run only. Re-run with --execute to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
