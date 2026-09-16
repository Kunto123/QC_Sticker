from __future__ import annotations

from functools import wraps
from typing import Callable

from flask import g, jsonify, request

from backend.app.core.container import token_store, users_repo
from shared.contracts.enums import UserRole


def _extract_token() -> str:
    auth = str(request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return str(request.headers.get("X-Auth-Token") or "").strip()


def current_user():
    token = _extract_token()
    if not token:
        return None
    session = token_store.get_record(token)
    if session is None:
        return None
    user = users_repo.get_user_info(session.user.id)
    if user is None or not user.is_active:
        token_store.revoke(token)
        return None
    g.current_token = token
    g.current_session = session
    return user


def require_auth(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = current_user()
        if user is None:
            return jsonify({"error": "Unauthorized"}), 401
        g.current_user = user
        return fn(*args, **kwargs)

    return wrapper


def require_roles(*allowed_roles: UserRole):
    def decorator(fn: Callable):
        @wraps(fn)
        @require_auth
        def wrapper(*args, **kwargs):
            user = g.current_user
            if user.role not in allowed_roles:
                return jsonify({"error": "Forbidden"}), 403
            return fn(*args, **kwargs)

        return wrapper

    return decorator
