"""Machine Settings API — CRUD + seed + diagnostics.

GET  /machine-settings          → current settings
PUT  /machine-settings          → update settings (admin)
POST /machine-settings/seed     → re-seed from env (admin, force=True)
GET  /machine-settings/plc/diagnostics  → live PLC status + input snapshot
POST /machine-settings/plc/test-coil   → pulse a coil for wiring test (admin)
"""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, g, jsonify, request

from backend.app.core.container import inspection_session_service, machine_settings_repo, plc_worker
from backend.app.core.http import require_roles
from backend.app.models.machine_settings import MachineSettings
from shared.contracts.enums import UserRole

logger = logging.getLogger(__name__)

machine_settings_blueprint = Blueprint("machine_settings", __name__, url_prefix="/machine-settings")


# ── CRUD ────────────────────────────────────────────────────────────

@machine_settings_blueprint.get("")
@require_roles(UserRole.ADMIN)
def get_machine_settings():
    """Return current machine settings."""
    settings = machine_settings_repo.load_settings()
    return jsonify(settings.to_dict())


@machine_settings_blueprint.put("")
@require_roles(UserRole.ADMIN)
def update_machine_settings():
    """Update machine settings. Marks seeded_from_env=False (user-edited)."""
    payload = request.get_json(force=True) or {}
    try:
        # Validate by attempting to parse
        new_settings = MachineSettings.from_dict(payload)
        # Mark as user-edited so env seed won't overwrite
        new_settings.seeded_from_env = False
        machine_settings_repo.save_settings(new_settings)
        logger.info("[machine-settings] updated by user %s", g.current_user.username)
    except (TypeError, ValueError, KeyError) as exc:
        return jsonify({"error": f"Invalid settings: {exc}"}), 400

    # If PLC worker is running, update its strategy with new settings
    if plc_worker is not None:
        try:
            plc_worker.set_validator_mode("sticker", new_settings)
            logger.info("[machine-settings] PLC worker strategy refreshed")
            # Also update PLC worker guards from new settings
            plc_worker.configure_guards(
                min_reclamp_interval_ms=new_settings.sticker.min_reclamp_interval_ms,
                release_input_debounce_ms=new_settings.sticker.release_input_debounce_ms,
            )
        except Exception as exc:
            logger.warning("[machine-settings] failed to refresh PLC worker strategy: %s", exc)

    # Propagate timing settings to inspection session service (runtime update)
    if inspection_session_service is not None:
        try:
            inspection_session_service.update_timing_settings(payload)
            logger.info("[machine-settings] timing settings pushed to inspection session service")
        except Exception as exc:
            logger.warning("[machine-settings] failed to push timing settings: %s", exc)

    return jsonify(new_settings.to_dict())


# ── Seed ────────────────────────────────────────────────────────────

@machine_settings_blueprint.post("/seed")
@require_roles(UserRole.ADMIN)
def seed_machine_settings():
    """Re-seed machine settings from env vars. Force overwrite."""
    from backend.app.core.container import app_config
    force = str(request.args.get("force") or "").lower() in ("1", "true", "yes")
    seeded = machine_settings_repo.seed_from_env(app_config, force=force)
    settings = machine_settings_repo.load_settings()
    return jsonify({
        "seeded": seeded,
        "settings": settings.to_dict(),
        "note": "Seeded from env vars" if seeded else "DB already exists, skipped (use ?force=1 to overwrite)",
    })


# ── PLC Diagnostics ─────────────────────────────────────────────────

@machine_settings_blueprint.get("/plc/diagnostics")
@require_roles(UserRole.ADMIN)
def plc_diagnostics():
    """Return live PLC status including input snapshot."""
    if plc_worker is None:
        return jsonify({"enabled": False, "note": "PLC worker is disabled"})
    return jsonify({"enabled": True, **plc_worker.status()})


@machine_settings_blueprint.post("/plc/test-coil")
@require_roles(UserRole.ADMIN)
def plc_test_coil():
    """Pulse a coil for wiring verification.

    Body: { "address": 0, "duration_ms": 500, "channels": 4 }
    Safety: only works when dry_run=True or explicitly confirmed.
    """
    if plc_worker is None:
        return jsonify({"error": "PLC worker is disabled"}), 400

    payload = request.get_json(force=True) or {}
    address = int(payload.get("address", 0))
    duration_ms = min(5000, max(100, int(payload.get("duration_ms", 500))))
    confirm = str(payload.get("confirm") or "").lower() == "yes"

    status = plc_worker.status()
    if not status.get("dry_run") and not confirm:
        return jsonify({
            "error": "dry_run is False. This will fire a real coil. Pass confirm=yes to proceed.",
            "dry_run": False,
        }), 400

    import time
    try:
        plc_worker._write_coil(address, True)
        time.sleep(duration_ms / 1000.0)
        plc_worker._write_coil(address, False)
        logger.info(
            "[machine-settings] test coil addr=%d duration=%dms by user %s",
            address, duration_ms, g.current_user.username,
        )
        return jsonify({"ok": True, "address": address, "duration_ms": duration_ms})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@machine_settings_blueprint.post("/plc/all-off")
@require_roles(UserRole.ADMIN)
def plc_all_off():
    """Emergency: turn off all coils."""
    if plc_worker is None:
        return jsonify({"error": "PLC worker is disabled"}), 400
    try:
        plc_worker._all_off("admin_all_off")
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
