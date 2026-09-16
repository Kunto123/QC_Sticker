"""Standalone utility: create a new admin user directly in Postgres.

Does NOT touch any existing application code/routes. Reuses the same
password hashing function the backend uses (backend.app.core.security)
so the resulting account can log in normally via /auth/login.

Usage:
    python scripts/create_postgres_admin.py <username> <password>

    # or run with no args and it will prompt interactively:
    python scripts/create_postgres_admin.py
"""
from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

import psycopg  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402

from backend.app.core.security import hash_password  # noqa: E402

TABLE_NAME = "qc_user_accounts"


def main() -> int:
    args = sys.argv[1:]
    if len(args) >= 2:
        username, password = args[0], args[1]
    else:
        username = input("New admin username: ").strip()
        password = getpass.getpass("New admin password: ").strip()

    username = username.strip()
    password = password.strip()
    if not username or not password:
        print("Username and password are required.", file=sys.stderr)
        return 1
    if len(password) < 6:
        print("Password must be at least 6 characters.", file=sys.stderr)
        return 1

    conn = psycopg.connect(
        host=os.getenv("POSTGRESQL_HOST", "127.0.0.1"),
        port=int(os.getenv("POSTGRESQL_PORT", "5432")),
        dbname=os.getenv("POSTGRESQL_DATABASE", ""),
        user=os.getenv("POSTGRESQL_USERNAME", ""),
        password=os.getenv("POSTGRESQL_PASSWORD", ""),
        row_factory=dict_row,
        sslmode=os.getenv("POSTGRESQL_SSLMODE", "prefer"),
        options=f"-c search_path={os.getenv('POSTGRESQL_SCHEMA', 'public')}",
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                f"SELECT id FROM {TABLE_NAME} WHERE LOWER(username) = LOWER(%s)",
                (username,),
            )
            if cursor.fetchone():
                print(f"Username '{username}' already exists.", file=sys.stderr)
                return 1

            password_hash = hash_password(password)
            cursor.execute(
                f"""
                INSERT INTO {TABLE_NAME} (username, password_hash, role, is_active)
                VALUES (%s, %s, 'admin', TRUE)
                RETURNING id
                """,
                (username, password_hash),
            )
            new_id = cursor.fetchone()["id"]
        conn.commit()
    finally:
        conn.close()

    print(f"Created admin user '{username}' (id={new_id}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
