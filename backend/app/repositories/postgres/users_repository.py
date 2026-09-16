from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.app.core.config import AppConfig
from backend.app.core.security import hash_password, verify_password
from backend.app.repositories.postgres._base import PostgresRepositoryBase, _format_datetime, _utcnow_iso
from shared.contracts.auth import UserInfo
from shared.contracts.enums import UserRole


def _seed_users() -> list[dict[str, Any]]:
    now = _utcnow_iso()
    return [
        {
            "id": 1,
            "username": "admin",
            "password_hash": hash_password("admin123"),
            "role": UserRole.ADMIN.value,
            "is_active": True,
            "created_at": now,
            "updated_at": now,
            "last_login_at": None,
            "rfid_uid_hash": None,
            "rfid_uid_last4": None,
            "rfid_bound_at": None,
        },
        {
            "id": 2,
            "username": "operator",
            "password_hash": hash_password("operator123"),
            "role": UserRole.OPERATOR.value,
            "is_active": True,
            "created_at": now,
            "updated_at": now,
            "last_login_at": None,
            "rfid_uid_hash": None,
            "rfid_uid_last4": None,
            "rfid_bound_at": None,
        },
    ]


class PostgresUsersRepository(PostgresRepositoryBase):
    TABLE_NAME = "qc_user_accounts"

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._ensure_schema()
        self._seed_defaults_if_empty()
        self.migrate_legacy_roles()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                        id BIGSERIAL PRIMARY KEY,
                        username TEXT NOT NULL,
                        password_hash TEXT NOT NULL,
                        role TEXT NOT NULL,
                        is_active BOOLEAN NOT NULL DEFAULT TRUE,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        last_login_at TIMESTAMPTZ NULL
                    )
                    """
                )
                cursor.execute(
                    f"""
                    CREATE UNIQUE INDEX IF NOT EXISTS UX_{self.TABLE_NAME}_username
                    ON {self.TABLE_NAME} (LOWER(username))
                    """
                )
                cursor.execute(
                    f"""
                    ALTER TABLE {self.TABLE_NAME}
                    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    """
                )
                cursor.execute(
                    f"""
                    ALTER TABLE {self.TABLE_NAME}
                    ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMPTZ NULL
                    """
                )
                cursor.execute(
                    f"""
                    ALTER TABLE {self.TABLE_NAME}
                    ADD COLUMN IF NOT EXISTS rfid_uid_hash TEXT NULL
                    """
                )
                cursor.execute(
                    f"""
                    ALTER TABLE {self.TABLE_NAME}
                    ADD COLUMN IF NOT EXISTS rfid_uid_last4 TEXT NULL
                    """
                )
                cursor.execute(
                    f"""
                    ALTER TABLE {self.TABLE_NAME}
                    ADD COLUMN IF NOT EXISTS rfid_bound_at TIMESTAMPTZ NULL
                    """
                )
                cursor.execute(
                    f"""
                    CREATE UNIQUE INDEX IF NOT EXISTS UX_{self.TABLE_NAME}_rfid_uid_hash
                    ON {self.TABLE_NAME} (rfid_uid_hash)
                    WHERE rfid_uid_hash IS NOT NULL
                    """
                )
            conn.commit()

    def _seed_defaults_if_empty(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(1) AS count FROM {self.TABLE_NAME}")
                count = int(cursor.fetchone()["count"] or 0)
                if count > 0:
                    return
                for record in _seed_users():
                    cursor.execute(
                        f"""
                        INSERT INTO {self.TABLE_NAME} (
                            username,
                            password_hash,
                            role,
                            is_active,
                            created_at,
                            updated_at,
                            last_login_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            record["username"],
                            record["password_hash"],
                            record["role"],
                            bool(record["is_active"]),
                            record["created_at"],
                            record["updated_at"],
                            record["last_login_at"],
                        ),
                    )
            conn.commit()

    def _row_to_record(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        return {
            "id": int(row["id"]),
            "username": str(row["username"]),
            "password_hash": str(row["password_hash"]),
            "role": str(row["role"]),
            "is_active": bool(row.get("is_active", True)),
            "created_at": _format_datetime(row.get("created_at")),
            "updated_at": _format_datetime(row.get("updated_at")),
            "last_login_at": _format_datetime(row.get("last_login_at")),
            "rfid_uid_hash": row.get("rfid_uid_hash"),
            "rfid_uid_last4": row.get("rfid_uid_last4"),
            "rfid_bound_at": _format_datetime(row.get("rfid_bound_at")),
        }

    def _public_record(self, record: dict[str, Any] | None) -> dict[str, Any] | None:
        if not record:
            return None
        return {
            "id": int(record["id"]),
            "username": str(record["username"]),
            "role": str(record["role"]),
            "is_active": bool(record.get("is_active", True)),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
            "last_login_at": record.get("last_login_at"),
            "rfid_uid_last4": record.get("rfid_uid_last4"),
            "rfid_bound_at": record.get("rfid_bound_at"),
            "rfid_bound": bool(record.get("rfid_uid_hash")),
        }

    def migrate_legacy_roles(self) -> int:
        now = _utcnow_iso()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.TABLE_NAME}
                    SET role = %s, updated_at = %s
                    WHERE LOWER(role) = %s
                    """,
                    (
                        UserRole.ADMIN.value,
                        now,
                        "engineer",
                    ),
                )
                updated = int(cursor.rowcount or 0)
            conn.commit()
        return max(0, updated)

    def list_users(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        id,
                        username,
                        password_hash,
                        role,
                        is_active,
                        created_at,
                        updated_at,
                        last_login_at,
                        rfid_uid_hash,
                        rfid_uid_last4,
                        rfid_bound_at
                    FROM {self.TABLE_NAME}
                    ORDER BY id ASC
                    """
                )
                rows = cursor.fetchall()
        return [self._public_record(self._row_to_record(row)) for row in rows]

    def get_by_username(self, username: str) -> dict[str, Any] | None:
        normalized = username.strip().lower()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        id,
                        username,
                        password_hash,
                        role,
                        is_active,
                        created_at,
                        updated_at,
                        last_login_at,
                        rfid_uid_hash,
                        rfid_uid_last4,
                        rfid_bound_at
                    FROM {self.TABLE_NAME}
                    WHERE LOWER(username) = %s
                    LIMIT 1
                    """,
                    (normalized,),
                )
                row = cursor.fetchone()
        return self._row_to_record(row)

    def get_by_id(self, user_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        id,
                        username,
                        password_hash,
                        role,
                        is_active,
                        created_at,
                        updated_at,
                        last_login_at,
                        rfid_uid_hash,
                        rfid_uid_last4,
                        rfid_bound_at
                    FROM {self.TABLE_NAME}
                    WHERE id = %s
                    LIMIT 1
                    """,
                    (int(user_id),),
                )
                row = cursor.fetchone()
        return self._row_to_record(row)

    def get_by_rfid_uid_hash(self, rfid_uid_hash: str) -> dict[str, Any] | None:
        normalized_hash = str(rfid_uid_hash or "").strip()
        if not normalized_hash:
            return None
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        id,
                        username,
                        password_hash,
                        role,
                        is_active,
                        created_at,
                        updated_at,
                        last_login_at,
                        rfid_uid_hash,
                        rfid_uid_last4,
                        rfid_bound_at
                    FROM {self.TABLE_NAME}
                    WHERE rfid_uid_hash = %s
                    LIMIT 1
                    """,
                    (normalized_hash,),
                )
                row = cursor.fetchone()
        return self._row_to_record(row)

    def to_user_info(self, record: dict[str, Any] | None) -> UserInfo | None:
        if not record:
            return None
        return UserInfo(
            id=int(record["id"]),
            username=str(record["username"]),
            role=UserRole(str(record["role"])),
            is_active=bool(record.get("is_active", True)),
        )

    def get_user_info(self, user_id: int) -> UserInfo | None:
        return self.to_user_info(self.get_by_id(user_id))

    def authenticate(self, username: str, password: str) -> UserInfo | None:
        record = self.get_by_username(username)
        if not record or not record.get("is_active"):
            return None
        if not verify_password(password, str(record.get("password_hash") or "")):
            return None
        now = _utcnow_iso()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.TABLE_NAME}
                    SET last_login_at = %s, updated_at = %s
                    WHERE id = %s
                    """,
                    (now, now, int(record["id"])),
                )
            conn.commit()
        record["last_login_at"] = now
        record["updated_at"] = now
        return self.to_user_info(record)

    def authenticate_rfid_hash(self, rfid_uid_hash: str) -> UserInfo | None:
        record = self.get_by_rfid_uid_hash(rfid_uid_hash)
        if not record or not record.get("is_active"):
            return None
        now = _utcnow_iso()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.TABLE_NAME}
                    SET last_login_at = %s, updated_at = %s
                    WHERE id = %s
                    """,
                    (now, now, int(record["id"])),
                )
            conn.commit()
        record["last_login_at"] = now
        record["updated_at"] = now
        return self.to_user_info(record)

    def create_user(self, username: str, password: str, role: str) -> dict[str, Any]:
        normalized_username = username.strip()
        if not normalized_username:
            raise ValueError("Username is required.")
        if self.get_by_username(normalized_username):
            raise ValueError("Username already exists.")
        role_enum = UserRole(role)
        now = _utcnow_iso()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {self.TABLE_NAME} (
                        username,
                        password_hash,
                        role,
                        is_active,
                        created_at,
                        updated_at,
                        last_login_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id,
                              username,
                              password_hash,
                              role,
                              is_active,
                              created_at,
                              updated_at,
                              last_login_at,
                              rfid_uid_hash,
                              rfid_uid_last4,
                              rfid_bound_at
                    """,
                    (
                        normalized_username,
                        hash_password(password),
                        role_enum.value,
                        True,
                        now,
                        now,
                        None,
                    ),
                )
                row = cursor.fetchone()
            conn.commit()
        record = self._public_record(self._row_to_record(row))
        if record is None:
            raise ValueError("Failed to create user.")
        return record

    def set_active(self, user_id: int, is_active: bool) -> dict[str, Any]:
        if self.get_by_id(user_id) is None:
            raise ValueError("User not found.")
        now = _utcnow_iso()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.TABLE_NAME}
                    SET is_active = %s, updated_at = %s
                    WHERE id = %s
                    RETURNING id,
                              username,
                              password_hash,
                              role,
                              is_active,
                              created_at,
                              updated_at,
                              last_login_at,
                              rfid_uid_hash,
                              rfid_uid_last4,
                              rfid_bound_at
                    """,
                    (bool(is_active), now, int(user_id)),
                )
                row = cursor.fetchone()
            conn.commit()
        record = self._public_record(self._row_to_record(row))
        if record is None:
            raise ValueError("User not found.")
        return record

    def set_role(self, user_id: int, role: str) -> dict[str, Any]:
        role_enum = UserRole(role)
        if self.get_by_id(user_id) is None:
            raise ValueError("User not found.")
        now = _utcnow_iso()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.TABLE_NAME}
                    SET role = %s, updated_at = %s
                    WHERE id = %s
                    RETURNING id,
                              username,
                              password_hash,
                              role,
                              is_active,
                              created_at,
                              updated_at,
                              last_login_at,
                              rfid_uid_hash,
                              rfid_uid_last4,
                              rfid_bound_at
                    """,
                    (role_enum.value, now, int(user_id)),
                )
                row = cursor.fetchone()
            conn.commit()
        record = self._public_record(self._row_to_record(row))
        if record is None:
            raise ValueError("User not found.")
        return record

    def delete_user(self, user_id: int) -> dict[str, Any]:
        record = self.get_by_id(user_id)
        if record is None:
            raise ValueError("User not found.")
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"DELETE FROM {self.TABLE_NAME} WHERE id = %s", (int(user_id),))
                conn.commit()
        return record

    def set_password(self, user_id: int, new_password: str) -> dict[str, Any]:
        if self.get_by_id(user_id) is None:
            raise ValueError("User not found.")
        now = _utcnow_iso()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.TABLE_NAME}
                    SET password_hash = %s, updated_at = %s
                    WHERE id = %s
                    RETURNING id,
                              username,
                              password_hash,
                              role,
                              is_active,
                              created_at,
                              updated_at,
                              last_login_at,
                              rfid_uid_hash,
                              rfid_uid_last4,
                              rfid_bound_at
                    """,
                    (hash_password(new_password), now, int(user_id)),
                )
                row = cursor.fetchone()
            conn.commit()
        record = self._public_record(self._row_to_record(row))
        if record is None:
            raise ValueError("User not found.")
        return record

    def set_rfid_uid_hash(self, user_id: int, rfid_uid_hash: str, rfid_uid_last4: str) -> dict[str, Any]:
        normalized_hash = str(rfid_uid_hash or "").strip()
        if not normalized_hash:
            raise ValueError("RFID UID is required.")
        existing = self.get_by_rfid_uid_hash(normalized_hash)
        if existing is not None and int(existing["id"]) != int(user_id):
            raise ValueError("RFID card is already bound to another user.")
        if self.get_by_id(user_id) is None:
            raise ValueError("User not found.")
        now = _utcnow_iso()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.TABLE_NAME}
                    SET rfid_uid_hash = %s,
                        rfid_uid_last4 = %s,
                        rfid_bound_at = %s,
                        updated_at = %s
                    WHERE id = %s
                    RETURNING id,
                              username,
                              password_hash,
                              role,
                              is_active,
                              created_at,
                              updated_at,
                              last_login_at,
                              rfid_uid_hash,
                              rfid_uid_last4,
                              rfid_bound_at
                    """,
                    (normalized_hash, str(rfid_uid_last4 or "")[-4:] or None, now, now, int(user_id)),
                )
                row = cursor.fetchone()
            conn.commit()
        record = self._public_record(self._row_to_record(row))
        if record is None:
            raise ValueError("User not found.")
        return record

    def clear_rfid_uid(self, user_id: int) -> dict[str, Any]:
        if self.get_by_id(user_id) is None:
            raise ValueError("User not found.")
        now = _utcnow_iso()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {self.TABLE_NAME}
                    SET rfid_uid_hash = NULL,
                        rfid_uid_last4 = NULL,
                        rfid_bound_at = NULL,
                        updated_at = %s
                    WHERE id = %s
                    RETURNING id,
                              username,
                              password_hash,
                              role,
                              is_active,
                              created_at,
                              updated_at,
                              last_login_at,
                              rfid_uid_hash,
                              rfid_uid_last4,
                              rfid_bound_at
                    """,
                    (now, int(user_id)),
                )
                row = cursor.fetchone()
            conn.commit()
        record = self._public_record(self._row_to_record(row))
        if record is None:
            raise ValueError("User not found.")
        return record
