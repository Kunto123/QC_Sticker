from __future__ import annotations

import sys

from flask import Blueprint, g, jsonify, request

from backend.app.core.container import app_config, audit_repo, token_store, users_repo
from backend.app.core.http import require_auth, require_roles
from backend.app.core.security import hash_rfid_uid, normalize_rfid_uid, rfid_uid_last4
from shared.contracts.enums import UserRole


auth_blueprint = Blueprint("auth", __name__, url_prefix="/auth")

ALLOWED_MANAGED_ROLES = {UserRole.ADMIN.value, UserRole.OPERATOR.value}


def _normalize_managed_role(raw_role: object, *, field_name: str = "role") -> str:
    role = str(raw_role or "").strip().lower()
    if not role:
        raise ValueError(f"{field_name} is required")
    if role not in ALLOWED_MANAGED_ROLES:
        allowed = ", ".join(sorted(ALLOWED_MANAGED_ROLES))
        raise ValueError(f"{field_name} must be one of: {allowed}")
    return role


def _client_ip() -> str:
    forwarded_for = str(request.headers.get("X-Forwarded-For") or "").strip()
    return forwarded_for.split(",", 1)[0].strip() if forwarded_for else (request.remote_addr or "")


def _client_name(payload: dict) -> str | None:
    value = str(payload.get("client_name") or request.headers.get("X-Client-Name") or "").strip()
    return value or None


def _try_audit(event_type: str, **kwargs) -> None:
    """Fire-and-forget audit log. Never raises so auth ops are not blocked."""
    try:
        audit_repo.log(event_type, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[audit] WARNING: failed to write audit event '{event_type}': {exc}", file=sys.stderr)


def _rfid_hash_from_payload(payload: dict) -> tuple[str, str]:
    normalized_uid = normalize_rfid_uid(payload.get("rfid_uid"))
    return hash_rfid_uid(normalized_uid, app_config.secret_key), rfid_uid_last4(normalized_uid)


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@auth_blueprint.post("/login")
def login():
    payload = request.get_json(force=True) or {}
    username_input = str(payload.get("username") or "").strip()
    password_input = str(payload.get("password") or "").strip()
    rfid_uid_input = str(payload.get("rfid_uid") or "").strip()
    ip = _client_ip()
    client = _client_name(payload)
    ua = request.user_agent.string or None

    user = None
    credential_label = ""
    rfid_last4 = ""
    if rfid_uid_input:
        try:
            rfid_uid_hash, _uid_last4 = _rfid_hash_from_payload(payload)
        except ValueError as exc:
            _try_audit(
                "login_failure",
                username=None,
                ip_address=ip,
                client_name=client,
                details=str(exc),
            )
            return jsonify({"error": "Invalid RFID card"}), 401
        rfid_last4 = _uid_last4
        user = users_repo.authenticate_rfid_hash(rfid_uid_hash)
        credential_label = "rfid"
    else:
        if not username_input or not password_input:
            _try_audit(
                "login_failure",
                username=username_input or None,
                ip_address=ip,
                client_name=client,
                details="Missing username or password",
            )
            return jsonify({"error": "Username and password are required"}), 400
        user = users_repo.authenticate(username_input, password_input)
        credential_label = "password"

    if user is None:
        _try_audit(
            "login_failure",
            username=username_input if credential_label == "password" else None,
            ip_address=ip,
            client_name=client,
            details=f"Invalid {credential_label} credentials"
                     + (f"; rfid_last4={rfid_last4}" if rfid_last4 else ""),
        )
        return jsonify({"error": "Invalid credentials"}), 401

    if user.role.value not in ALLOWED_MANAGED_ROLES:
        _try_audit(
            "login_failure",
            user_id=user.id,
            username=user.username,
            ip_address=ip,
            client_name=client,
            details=f"Unsupported role for {credential_label} login",
        )
        return jsonify({"error": "Invalid credentials"}), 401

    session = token_store.issue(user, ip_address=ip, client_name=client, user_agent=ua)
    _try_audit(
        "login_success",
        user_id=user.id,
        username=user.username,
        session_id=session.session_id,
        ip_address=ip,
        client_name=client,
        details=f"credential={credential_label}",
    )
    return jsonify(
        {
            "token": session.token,
            "user": user.to_dict(),
            "expires_in": session.remaining_seconds(now=session.issued_at),
            "expires_at": session.expires_at.isoformat(),
            "session": session.to_dict(),
        }
    )


@auth_blueprint.get("/me")
@require_auth
def me():
    return jsonify(g.current_user.to_dict())


@auth_blueprint.post("/logout")
@require_auth
def logout():
    token = getattr(g, "current_token", "")
    session = getattr(g, "current_session", None)
    revoked = token_store.revoke(token)
    _try_audit(
        "logout",
        user_id=g.current_user.id,
        username=g.current_user.username,
        session_id=session.session_id if session else None,
        ip_address=_client_ip(),
    )
    return jsonify({"revoked": revoked})


@auth_blueprint.post("/logout-all")
@require_auth
def logout_all():
    revoked = token_store.revoke_user(g.current_user.id)
    _try_audit(
        "logout_all",
        user_id=g.current_user.id,
        username=g.current_user.username,
        ip_address=_client_ip(),
        details=f"revoked {revoked} session(s)",
    )
    return jsonify({"revoked": revoked, "user_id": g.current_user.id})


@auth_blueprint.get("/sessions")
@require_auth
def list_sessions():
    current_session = getattr(g, "current_session", None)
    items = []
    for session in token_store.list_user_sessions(g.current_user.id):
        record = session.to_dict()
        record["is_current"] = bool(current_session and session.session_id == current_session.session_id)
        items.append(record)
    return jsonify(items)


# ---------------------------------------------------------------------------
# User management (admin only)
# ---------------------------------------------------------------------------

@auth_blueprint.get("/users")
@require_roles(UserRole.ADMIN)
def list_users():
    return jsonify(users_repo.list_users())


@auth_blueprint.post("/users")
@require_roles(UserRole.ADMIN)
def create_user():
    payload = request.get_json(force=True) or {}
    try:
        role = _normalize_managed_role(payload.get("role") or UserRole.OPERATOR.value)
        record = users_repo.create_user(
            username=str(payload.get("username") or "").strip(),
            password=str(payload.get("password") or "").strip(),
            role=role,
        )
    except (ValueError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400
    _try_audit(
        "user_created",
        user_id=record.get("id"),
        username=record.get("username"),
        actor_id=g.current_user.id,
        actor_username=g.current_user.username,
        ip_address=_client_ip(),
        details=f"role={record.get('role')}",
    )
    return jsonify(record), 201


@auth_blueprint.put("/users/<int:user_id>")
@require_roles(UserRole.ADMIN)
def update_user(user_id: int):
    payload = request.get_json(force=True) or {}
    is_active = bool(payload.get("is_active", True))
    try:
        record = users_repo.set_active(user_id, is_active)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    if not is_active:
        token_store.revoke_user(user_id)
    _try_audit(
        "user_disabled" if not is_active else "user_enabled",
        user_id=user_id,
        username=record.get("username"),
        actor_id=g.current_user.id,
        actor_username=g.current_user.username,
        ip_address=_client_ip(),
    )
    return jsonify(record)


@auth_blueprint.post("/users/<int:user_id>/role")
@require_roles(UserRole.ADMIN)
def change_user_role(user_id: int):
    payload = request.get_json(force=True) or {}
    try:
        new_role = _normalize_managed_role(payload.get("role"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        record = users_repo.set_role(user_id, new_role)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    _try_audit(
        "role_changed",
        user_id=user_id,
        username=record.get("username"),
        actor_id=g.current_user.id,
        actor_username=g.current_user.username,
        ip_address=_client_ip(),
        details=f"new_role={new_role}",
    )
    return jsonify(record)


@auth_blueprint.post("/users/<int:user_id>/reset-password")
@require_roles(UserRole.ADMIN)
def reset_user_password(user_id: int):
    payload = request.get_json(force=True) or {}
    new_password = str(payload.get("password") or "").strip()
    if len(new_password) < 6:
        return jsonify({"error": "password must be at least 6 characters"}), 400
    try:
        record = users_repo.set_password(user_id, new_password)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    # Revoke existing sessions so the user must re-authenticate
    token_store.revoke_user(user_id)
    _try_audit(
        "password_reset",
        user_id=user_id,
        username=record.get("username"),
        actor_id=g.current_user.id,
        actor_username=g.current_user.username,
        ip_address=_client_ip(),
    )
    return jsonify({"ok": True, "user_id": user_id, "sessions_revoked": True})


@auth_blueprint.post("/users/<int:user_id>/rfid")
@require_roles(UserRole.ADMIN)
def bind_user_rfid(user_id: int):
    payload = request.get_json(force=True) or {}
    try:
        rfid_uid_hash, last4 = _rfid_hash_from_payload(payload)
        record = users_repo.set_rfid_uid_hash(user_id, rfid_uid_hash, last4)
    except ValueError as exc:
        message = str(exc)
        if "already bound" in message.lower():
            return jsonify({"error": message}), 409
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    revoked = token_store.revoke_user(user_id)
    _try_audit(
        "rfid_bound",
        user_id=user_id,
        username=record.get("username"),
        actor_id=g.current_user.id,
        actor_username=g.current_user.username,
        ip_address=_client_ip(),
        details=f"last4={record.get('rfid_uid_last4')}; revoked={revoked}",
    )
    return jsonify({"user": record, "sessions_revoked": revoked})


@auth_blueprint.delete("/users/<int:user_id>/rfid")
@require_roles(UserRole.ADMIN)
def clear_user_rfid(user_id: int):
    try:
        record = users_repo.clear_rfid_uid(user_id)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    revoked = token_store.revoke_user(user_id)
    _try_audit(
        "rfid_cleared",
        user_id=user_id,
        username=record.get("username"),
        actor_id=g.current_user.id,
        actor_username=g.current_user.username,
        ip_address=_client_ip(),
        details=f"revoked={revoked}",
    )
    return jsonify({"user": record, "sessions_revoked": revoked})


@auth_blueprint.post("/users/<int:user_id>/revoke-sessions")
@require_roles(UserRole.ADMIN)
def revoke_user_sessions(user_id: int):
    target = users_repo.get_by_id(user_id)
    if target is None:
        return jsonify({"error": "User not found."}), 404
    revoked = token_store.revoke_user(user_id)
    _try_audit(
        "session_revoked",
        user_id=user_id,
        username=target.get("username"),
        actor_id=g.current_user.id,
        actor_username=g.current_user.username,
        ip_address=_client_ip(),
        details=f"revoked {revoked} session(s)",
    )
    return jsonify({"user_id": user_id, "revoked": revoked})


@auth_blueprint.delete("/users/<int:user_id>")
@require_roles(UserRole.ADMIN)
def delete_user(user_id: int):
    target = users_repo.get_by_id(user_id)
    if target is None:
        return jsonify({"error": "User not found."}), 404
    # Prevent self-deletion
    if user_id == g.current_user.id:
        return jsonify({"error": "Cannot delete your own account."}), 400
    removed = users_repo.delete_user(user_id)
    token_store.revoke_user(user_id)
    _try_audit(
        "user_deleted",
        user_id=user_id,
        username=removed.get("username"),
        actor_id=g.current_user.id,
        actor_username=g.current_user.username,
        ip_address=_client_ip(),
    )
    return jsonify({"ok": True, "deleted": removed.get("id"), "username": removed.get("username")})


# ---------------------------------------------------------------------------
# Audit log (admin only)
# ---------------------------------------------------------------------------

@auth_blueprint.get("/audit-log")
@require_roles(UserRole.ADMIN)
def get_audit_log():
    """Return recent auth audit events.

    Query params:
    - ``limit``   max entries to return (default 100, max 500)
    - ``user_id`` optional filter by user
    """
    try:
        limit = min(500, max(1, int(request.args.get("limit", 100))))
    except (ValueError, TypeError):
        limit = 100
    user_id_raw = request.args.get("user_id")
    user_id: int | None = None
    if user_id_raw is not None:
        try:
            user_id = int(user_id_raw)
        except (ValueError, TypeError):
            return jsonify({"error": "user_id must be an integer"}), 400
    return jsonify(audit_repo.list_recent(limit=limit, user_id=user_id))
