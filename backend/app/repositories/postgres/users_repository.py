"""Autentikasi berbasis tabel `operator` milik pabrik.

Tabel itu (kolom: No/MC_ID/No_RFID/Member_ID/StatusMP secara default — lihat
`QC_SUITE_OPERATOR_*` di core/config.py) dimiliki sistem lain (MES pabrik),
bukan app ini:

- Tidak ada auto-create / auto-migrate: tabelnya harus sudah ada. Kalau
  belum ada, query gagal keras alih-alih diam-diam membuat schema bayangan.
- Tidak ada seeding akun default: akun dikelola di tabel bersama itu.
- MC_ID cuma data per-baris biasa, bukan query scope: menunjukkan mesin/
  station tempat operator itu ditugaskan (di-set/diedit dari form Admin ->
  Operators), tidak memfilter baris mana yang bisa dilihat app ini.
- Tidak ada kolom password_hash: nilai No_RFID sekaligus jadi password login
  (mengetiknya, atau tap kartu fisik, keduanya dibandingkan ke kolom yang
  sama). Juga tidak ada kolom is_active / created_at / updated_at /
  last_login_at, jadi konsep-konsep itu diusahakan sebaik mungkin (lihat set_active).
"""
from __future__ import annotations

from typing import Any

from backend.app.core.config import AppConfig
from backend.app.repositories.postgres._base import PostgresRepositoryBase
from shared.contracts.auth import UserInfo
from shared.contracts.enums import UserRole


def _quote_ident(name: str) -> str:
    return ".".join(f'"{part}"' for part in str(name).split("."))


class PostgresUsersRepository(PostgresRepositoryBase):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._table = _quote_ident(config.operator_table)
        self._col_no = _quote_ident(config.operator_col_no)
        self._col_mc_id = _quote_ident(config.operator_col_mc_id)
        self._col_rfid = _quote_ident(config.operator_col_rfid)
        self._col_member_id = _quote_ident(config.operator_col_member_id)
        self._col_status = _quote_ident(config.operator_col_status)

    def _select_columns(self) -> str:
        return (
            f"{self._col_no} AS no_pk, "
            f"{self._col_member_id} AS member_id, "
            f"{self._col_rfid} AS rfid, "
            f"{self._col_status} AS status_mp, "
            f"{self._col_mc_id} AS mc_id"
        )

    def _row_to_record(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        rfid = str(row.get("rfid") or "").strip()
        return {
            "id": int(row["no_pk"]),
            "username": str(row.get("member_id") or ""),
            "role": str(row.get("status_mp") or ""),
            "mc_id": str(row.get("mc_id") or ""),
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
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT {self._select_columns()} FROM {self._table} ORDER BY {self._col_no} ASC")
                rows = cursor.fetchall()
        return [self._public_record(self._row_to_record(row)) for row in rows]

    def get_by_username(self, username: str) -> dict[str, Any] | None:
        normalized = username.strip()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"SELECT {self._select_columns()} FROM {self._table} "
                    f"WHERE UPPER({self._col_member_id}) = UPPER(%s) LIMIT 1",
                    (normalized,),
                )
                row = cursor.fetchone()
        return self._row_to_record(row)

    def get_by_id(self, user_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"SELECT {self._select_columns()} FROM {self._table} WHERE {self._col_no} = %s LIMIT 1",
                    (int(user_id),),
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
            with conn.cursor() as cursor:
                cursor.execute(
                    f"SELECT {self._select_columns()} FROM {self._table} "
                    f"WHERE UPPER({self._col_rfid}) = %s LIMIT 1",
                    (normalized,),
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
        from psycopg.errors import UniqueViolation

        row = None
        for _attempt in range(5):
            try:
                with self._connect() as conn:
                    with conn.cursor() as cursor:
                        # Hitung No = MAX(No) + 1 sendiri, tidak percaya identity/sequence
                        # DB supaya tetap sinkron (tabel operator bisa juga ditulis sistem
                        # lain). Kalau writer lain ambil angka yang sama duluan, PK asli
                        # di No akan menolaknya dan kita coba lagi.
                        cursor.execute(
                            f"INSERT INTO {self._table} "
                            f"({self._col_member_id}, {self._col_rfid}, {self._col_status}, "
                            f"{self._col_mc_id}, {self._col_no}) "
                            f"SELECT %s, %s, %s, %s, COALESCE(MAX({self._col_no}), 0) + 1 FROM {self._table} "
                            f"RETURNING {self._select_columns()}",
                            (normalized_username, rfid_value, role_enum.value, mc_id_value),
                        )
                        row = cursor.fetchone()
                    conn.commit()
                break
            except UniqueViolation:
                row = None
                continue
        if row is None:
            raise ValueError("Gagal membuat user: tidak berhasil mengalokasikan No yang bebas setelah beberapa percobaan.")
        record = self._public_record(self._row_to_record(row))
        if record is None:
            raise ValueError("Gagal membuat user.")
        return record

    def set_active(self, user_id: int, is_active: bool) -> dict[str, Any]:
        # Tidak ada kolom is_active di tabel operator — enable/disable akun
        # tidak didukung di backend ini, cuma add/edit/delete.
        record = self.get_by_id(user_id)
        if record is None:
            raise ValueError("User not found.")
        return self._public_record(record)

    def set_role(self, user_id: int, role: str) -> dict[str, Any]:
        role_enum = UserRole(role)
        if self.get_by_id(user_id) is None:
            raise ValueError("User not found.")
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"UPDATE {self._table} SET {self._col_status} = %s "
                    f"WHERE {self._col_no} = %s RETURNING {self._select_columns()}",
                    (role_enum.value, int(user_id)),
                )
                row = cursor.fetchone()
            conn.commit()
        record = self._public_record(self._row_to_record(row))
        if record is None:
            raise ValueError("User not found.")
        return record

    def set_mc_id(self, user_id: int, mc_id: str) -> dict[str, Any]:
        if self.get_by_id(user_id) is None:
            raise ValueError("User not found.")
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"UPDATE {self._table} SET {self._col_mc_id} = %s "
                    f"WHERE {self._col_no} = %s RETURNING {self._select_columns()}",
                    (str(mc_id or "").strip(), int(user_id)),
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
                cursor.execute(f"DELETE FROM {self._table} WHERE {self._col_no} = %s", (int(user_id),))
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
            with conn.cursor() as cursor:
                cursor.execute(
                    f"UPDATE {self._table} SET {self._col_rfid} = %s "
                    f"WHERE {self._col_no} = %s RETURNING {self._select_columns()}",
                    (rfid_value, int(user_id)),
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
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"UPDATE {self._table} SET {self._col_rfid} = NULL "
                    f"WHERE {self._col_no} = %s RETURNING {self._select_columns()}",
                    (int(user_id),),
                )
                row = cursor.fetchone()
            conn.commit()
        record = self._public_record(self._row_to_record(row))
        if record is None:
            raise ValueError("User not found.")
        return record
