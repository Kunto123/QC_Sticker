"""Authentication backed by the factory's own `operator` table.

See the Postgres sibling (`postgres/users_repository.py`) for the full
rationale — same contract, pyodbc instead of psycopg: no auto-create, no
seeding, MC_ID is plain per-row data (not a query scope), and No_RFID
doubles as the login password.
"""
from __future__ import annotations

from typing import Any

from backend.app.core.config import AppConfig
from shared.contracts.auth import UserInfo
from shared.contracts.enums import UserRole


def _quote_ident(name: str) -> str:
    return ".".join(f"[{part}]" for part in str(name).split("."))


class SqlServerUsersRepository:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._table = _quote_ident(config.operator_table)
        self._col_no = _quote_ident(config.operator_col_no)
        self._col_mc_id = _quote_ident(config.operator_col_mc_id)
        self._col_rfid = _quote_ident(config.operator_col_rfid)
        self._col_member_id = _quote_ident(config.operator_col_member_id)
        self._col_status = _quote_ident(config.operator_col_status)

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

    def _select_columns(self) -> str:
        return (
            f"{self._col_no} AS no_pk, "
            f"{self._col_member_id} AS member_id, "
            f"{self._col_rfid} AS rfid, "
            f"{self._col_status} AS status_mp, "
            f"{self._col_mc_id} AS mc_id"
        )

    def _row_to_record(self, row) -> dict[str, Any] | None:
        if row is None:
            return None
        rfid = str(row.rfid).strip() if row.rfid is not None else ""
        return {
            "id": int(row.no_pk),
            "username": str(row.member_id) if row.member_id is not None else "",
            "role": str(row.status_mp) if row.status_mp is not None else "",
            "mc_id": str(row.mc_id) if row.mc_id is not None else "",
            "rfid": rfid,
            "is_active": True,
            "created_at": None,
            "updated_at": None,
            "last_login_at": None,
            "rfid_uid_last4": rfid[-4:] if rfid else None,
            "rfid_bound_at": None,
        }

    def _public_record(self, record: dict[str, Any] | None) -> dict[str, Any] | None:
        if not record:
            return None
        return {
            "id": record["id"],
            "username": record["username"],
            "role": record["role"],
            "mc_id": record.get("mc_id") or "",
            "is_active": True,
            "created_at": None,
            "updated_at": None,
            "last_login_at": None,
            "rfid_uid_last4": record.get("rfid_uid_last4"),
            "rfid_bound_at": None,
            "rfid_bound": bool(record.get("rfid")),
        }

    def list_users(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT {self._select_columns()} FROM {self._table} ORDER BY {self._col_no} ASC")
            rows = cursor.fetchall()
        return [self._public_record(self._row_to_record(row)) for row in rows]

    def get_by_username(self, username: str) -> dict[str, Any] | None:
        normalized = username.strip()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT TOP 1 {self._select_columns()} FROM {self._table} "
                f"WHERE UPPER({self._col_member_id}) = UPPER(?)",
                normalized,
            )
            row = cursor.fetchone()
        return self._row_to_record(row)

    def get_by_id(self, user_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT TOP 1 {self._select_columns()} FROM {self._table} WHERE {self._col_no} = ?",
                int(user_id),
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
            is_active=True,
        )

    def get_user_info(self, user_id: int) -> UserInfo | None:
        return self.to_user_info(self.get_by_id(user_id))

    def authenticate(self, username: str, password: str) -> UserInfo | None:
        record = self.get_by_username(username)
        if not record:
            return None
        if str(password or "").strip().upper() != str(record.get("rfid") or "").strip().upper():
            return None
        return self.to_user_info(record)

    def authenticate_rfid(self, *, normalized_uid: str, rfid_uid_hash: str) -> UserInfo | None:
        normalized = str(normalized_uid or "").strip().upper()
        if not normalized:
            return None
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT TOP 1 {self._select_columns()} FROM {self._table} "
                f"WHERE UPPER({self._col_rfid}) = ?",
                normalized,
            )
            row = cursor.fetchone()
        record = self._row_to_record(row)
        return self.to_user_info(record) if record else None

    def create_user(self, username: str, password: str, role: str, mc_id: str = "") -> dict[str, Any]:
        normalized_username = username.strip()
        if not normalized_username:
            raise ValueError("Username is required.")
        if self.get_by_username(normalized_username):
            raise ValueError("Username already exists.")
        role_enum = UserRole(role)
        rfid_value = str(password or "").strip().upper()
        mc_id_value = str(mc_id or "").strip()
        import pyodbc

        inserted_id = None
        for _attempt in range(5):
            try:
                with self._connect() as conn:
                    cursor = conn.cursor()
                    # Compute No = MAX(No) + 1 ourselves rather than trusting a DB
                    # identity to stay in sync (the operator table can be written to
                    # by other systems too). If another writer takes the same number
                    # first, the PK on No rejects it and we retry.
                    cursor.execute(
                        f"INSERT INTO {self._table} "
                        f"({self._col_member_id}, {self._col_rfid}, {self._col_status}, "
                        f"{self._col_mc_id}, {self._col_no}) "
                        f"OUTPUT INSERTED.{self._col_no} "
                        f"SELECT ?, ?, ?, ?, ISNULL(MAX({self._col_no}), 0) + 1 FROM {self._table}",
                        normalized_username,
                        rfid_value,
                        role_enum.value,
                        mc_id_value,
                    )
                    inserted_id = int(cursor.fetchone()[0])
                    conn.commit()
                break
            except pyodbc.IntegrityError:
                inserted_id = None
                continue
        if inserted_id is None:
            raise ValueError("Failed to create user: could not allocate a free No after several attempts.")
        record = self.get_by_id(inserted_id)
        if record is None:
            raise ValueError("Failed to create user.")
        return self._public_record(record)

    def set_active(self, user_id: int, is_active: bool) -> dict[str, Any]:
        # No is_active column on the operator table — enabling/disabling
        # accounts isn't supported on this backend, only add/edit/delete.
        record = self.get_by_id(user_id)
        if record is None:
            raise ValueError("User not found.")
        return self._public_record(record)

    def set_role(self, user_id: int, role: str) -> dict[str, Any]:
        role_enum = UserRole(role)
        if self.get_by_id(user_id) is None:
            raise ValueError("User not found.")
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE {self._table} SET {self._col_status} = ? WHERE {self._col_no} = ?",
                role_enum.value,
                int(user_id),
            )
            conn.commit()
        record = self.get_by_id(user_id)
        if record is None:
            raise ValueError("User not found.")
        return self._public_record(record)

    def set_mc_id(self, user_id: int, mc_id: str) -> dict[str, Any]:
        if self.get_by_id(user_id) is None:
            raise ValueError("User not found.")
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE {self._table} SET {self._col_mc_id} = ? WHERE {self._col_no} = ?",
                str(mc_id or "").strip(),
                int(user_id),
            )
            conn.commit()
        record = self.get_by_id(user_id)
        if record is None:
            raise ValueError("User not found.")
        return self._public_record(record)

    def delete_user(self, user_id: int) -> dict[str, Any]:
        record = self.get_by_id(user_id)
        if record is None:
            raise ValueError("User not found.")
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self._table} WHERE {self._col_no} = ?", int(user_id))
            conn.commit()
        return self._public_record(record)

    def set_password(self, user_id: int, new_password: str) -> dict[str, Any]:
        return self.set_rfid_uid(user_id, normalized_uid=new_password, rfid_uid_hash="", rfid_uid_last4="")

    def set_rfid_uid(self, user_id: int, *, normalized_uid: str, rfid_uid_hash: str, rfid_uid_last4: str) -> dict[str, Any]:
        rfid_value = str(normalized_uid or "").strip().upper()
        if not rfid_value:
            raise ValueError("RFID UID is required.")
        if self.get_by_id(user_id) is None:
            raise ValueError("User not found.")
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE {self._table} SET {self._col_rfid} = ? WHERE {self._col_no} = ?",
                rfid_value,
                int(user_id),
            )
            conn.commit()
        record = self.get_by_id(user_id)
        if record is None:
            raise ValueError("User not found.")
        return self._public_record(record)

    def clear_rfid_uid(self, user_id: int) -> dict[str, Any]:
        if self.get_by_id(user_id) is None:
            raise ValueError("User not found.")
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE {self._table} SET {self._col_rfid} = NULL WHERE {self._col_no} = ?",
                int(user_id),
            )
            conn.commit()
        record = self.get_by_id(user_id)
        if record is None:
            raise ValueError("User not found.")
        return self._public_record(record)
