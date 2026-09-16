from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from shared.contracts.auth import UserInfo


_RFID_ALLOWED_PATTERN = re.compile(r"^[A-Z0-9]{4,64}$")


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    ).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, _ = stored_hash.split("$", 1)
    except ValueError:
        return False
    candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, stored_hash)


def normalize_rfid_uid(raw_uid: object) -> str:
    """Normalize RFID keyboard-wedge output into a stable UID string."""
    value = str(raw_uid or "").strip()
    normalized = re.sub(r"[\s:\-]", "", value).upper()
    if not _RFID_ALLOWED_PATTERN.fullmatch(normalized):
        raise ValueError("RFID UID must be 4-64 alphanumeric characters.")
    return normalized


def hash_rfid_uid(normalized_uid: str, secret_key: str) -> str:
    secret = str(secret_key or "").encode("utf-8")
    payload = str(normalized_uid or "").encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def rfid_uid_last4(normalized_uid: str) -> str:
    return str(normalized_uid or "")[-4:]


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class TokenRecord:
    session_id: str
    token: str
    user: UserInfo
    issued_at: datetime
    expires_at: datetime
    last_seen_at: datetime
    ip_address: str | None = None
    client_name: str | None = None
    user_agent: str | None = None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user.id,
            "username": self.user.username,
            "role": self.user.role.value,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "ip_address": self.ip_address,
            "client_name": self.client_name,
            "user_agent": self.user_agent,
        }

    def remaining_seconds(self, *, now: datetime | None = None) -> int:
        current = now or _utcnow()
        return max(0, int((self.expires_at - current).total_seconds()))


class TokenStore:
    def __init__(self, ttl_seconds: int = 86_400) -> None:
        self.ttl_seconds = max(60, int(ttl_seconds or 86_400))
        self._items: dict[str, TokenRecord] = {}
        self._lock = threading.Lock()

    def _purge_expired_locked(self, now: datetime) -> int:
        expired_tokens = [token for token, record in self._items.items() if record.expires_at <= now]
        for token in expired_tokens:
            self._items.pop(token, None)
        return len(expired_tokens)

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
        now = _utcnow()
        ttl = max(60, int(ttl_seconds or self.ttl_seconds))
        record = TokenRecord(
            session_id=secrets.token_hex(8),
            token=token,
            user=user,
            issued_at=now,
            expires_at=now + timedelta(seconds=ttl),
            last_seen_at=now,
            ip_address=(ip_address or "").strip() or None,
            client_name=(client_name or "").strip() or None,
            user_agent=(user_agent or "").strip() or None,
        )
        with self._lock:
            self._purge_expired_locked(now)
            self._items[token] = record
        return record

    def get(self, token: str) -> UserInfo | None:
        record = self.get_record(token)
        return record.user if record else None

    def get_record(self, token: str, *, touch: bool = True) -> TokenRecord | None:
        if not token:
            return None
        now = _utcnow()
        with self._lock:
            self._purge_expired_locked(now)
            record = self._items.get(token)
            if record is None:
                return None
            if touch:
                record.last_seen_at = now
        return record

    def revoke(self, token: str) -> bool:
        with self._lock:
            return self._items.pop(token, None) is not None

    def revoke_user(self, user_id: int, *, exclude_token: str | None = None) -> int:
        revoked = 0
        with self._lock:
            for token, record in list(self._items.items()):
                if int(record.user.id) != int(user_id):
                    continue
                if exclude_token and token == exclude_token:
                    continue
                self._items.pop(token, None)
                revoked += 1
        return revoked

    def list_user_sessions(self, user_id: int) -> list[TokenRecord]:
        now = _utcnow()
        with self._lock:
            self._purge_expired_locked(now)
            items = [record for record in self._items.values() if int(record.user.id) == int(user_id)]
        return sorted(items, key=lambda item: item.issued_at, reverse=True)

    def purge_expired(self) -> int:
        now = _utcnow()
        with self._lock:
            return self._purge_expired_locked(now)
