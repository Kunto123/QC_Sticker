from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SessionState:
    base_url: str
    token: str | None = None
    user: dict[str, Any] | None = None
    active_session: dict[str, Any] | None = None
    active_deployment: dict[str, Any] | None = None
    latest_result: dict[str, Any] | None = None
    latest_error: str | None = None
    cache: dict[str, Any] = field(default_factory=dict)

