from __future__ import annotations

from dataclasses import asdict, dataclass

from shared.contracts.enums import UserRole


@dataclass(slots=True)
class UserInfo:
    id: int
    username: str
    role: UserRole
    is_active: bool = True

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["role"] = self.role.value
        return payload

