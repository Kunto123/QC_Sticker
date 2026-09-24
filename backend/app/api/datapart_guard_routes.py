"""Datapart guard API.

GET  /datapart-guard/status              -> current lock state (any authenticated user)
POST /datapart-guard/override            -> admin-only escape hatch when the downstream
                                             MES never fills DatapartID and the line is
                                             stuck waiting.
POST /datapart-guard/override-with-rfid  -> same escape hatch, authorized by scanning a
                                             LEADERPI's RFID card instead of switching to
                                             an admin session — for the operator-screen
                                             lock popup's "Bypass" flow.
"""
from __future__ import annotations

import logging

from flask import Blueprint, g, jsonify, request

from backend.app.core.container import app_config, audit_repo, datapart_guard_service, users_repo
from backend.app.core.http import require_auth, require_roles
from backend.app.core.security import hash_rfid_uid, normalize_rfid_uid
from shared.contracts.enums import UserRole

logger = logging.getLogger(__name__)

datapart_guard_blueprint = Blueprint("datapart_guard", __name__, url_prefix="/datapart-guard")


def _try_audit(event_type: str, **kwargs) -> None:
    try:
        audit_repo.log(event_type, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[datapart-guard] failed to write audit event '%s': %s", event_type, exc)


@datapart_guard_blueprint.get("/status")
@require_auth
def get_status():
    return jsonify(datapart_guard_service.status())


@datapart_guard_blueprint.post("/override")
@require_roles(UserRole.ADMIN)
def override():
    datapart_guard_service.override(by=g.current_user.username)
    _try_audit(
        "datapart_guard_override",
        user_id=g.current_user.id,
        username=g.current_user.username,
        details="override via admin session",
    )
    logger.info("[datapart-guard] override requested by %s", g.current_user.username)
    return jsonify(datapart_guard_service.status())


@datapart_guard_blueprint.post("/override-with-rfid")
@require_auth
def override_with_rfid():
    """Bypass the lock by scanning a LEADERPI's RFID card — the caller's own
    session (typically an operator, mid-inspection) does not need admin
    rights; authorization comes entirely from the scanned card's role."""
    payload = request.get_json(force=True) or {}
    raw_uid = payload.get("rfid_uid")
    try:
        normalized_uid = normalize_rfid_uid(raw_uid)
    except ValueError as exc:
        _try_audit(
            "datapart_guard_override_denied",
            user_id=g.current_user.id,
            username=g.current_user.username,
            details=f"invalid RFID: {exc}",
        )
        return jsonify({"error": "Kartu RFID tidak valid."}), 400

    rfid_uid_hash = hash_rfid_uid(normalized_uid, app_config.secret_key)
    scanned_user = users_repo.authenticate_rfid(normalized_uid=normalized_uid, rfid_uid_hash=rfid_uid_hash)

    if scanned_user is None:
        _try_audit(
            "datapart_guard_override_denied",
            user_id=g.current_user.id,
            username=g.current_user.username,
            details="RFID not recognized",
        )
        return jsonify({"error": "Kartu RFID tidak dikenali."}), 401

    if scanned_user.role != UserRole.ADMIN:
        _try_audit(
            "datapart_guard_override_denied",
            user_id=g.current_user.id,
            username=g.current_user.username,
            details=f"RFID belongs to {scanned_user.username} (role={scanned_user.role.value}), not LEADERPI",
        )
        return jsonify({"error": "Kartu ini bukan LEADERPI."}), 403

    datapart_guard_service.override(by=scanned_user.username)
    _try_audit(
        "datapart_guard_override",
        user_id=g.current_user.id,
        username=g.current_user.username,
        details=f"override via RFID scan, authorized by LEADERPI {scanned_user.username}",
    )
    logger.info(
        "[datapart-guard] override via RFID by LEADERPI %s (requested from session of %s)",
        scanned_user.username, g.current_user.username,
    )
    return jsonify(datapart_guard_service.status())
