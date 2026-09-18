"""Machine Settings API — CRUD + PLC diagnostics.

GET  /machine-settings                  → current settings (+ restart_required)
PUT  /machine-settings                  → persist + live-apply (admin)
GET  /machine-settings/plc/diagnostics  → live PLC status + input snapshot
POST /machine-settings/plc/test-coil    → pulse a coil for wiring test (admin)
POST /machine-settings/plc/all-off      → emergency all coils off (admin)

`machine_settings.json` is the only source for these values (nothing comes from
`.env`). `io` and `timing` are applied live; `connection` and `inference` need a
backend restart — the response says so via `restart_required`.
"""
from __future__ import annotations

import logging
from dataclasses import asdict

from flask import Blueprint, g, jsonify, request

from backend.app.core.container import (
    app_config,
    boot_connection,
    inspection_session_service,
    machine_settings_repo,
    plc_worker,
)
from backend.app.core.http import require_roles
from backend.app.models.machine_settings import InferenceConfig, MachineSettings
from shared.contracts.enums import UserRole

logger = logging.getLogger(__name__)

machine_settings_blueprint = Blueprint("machine_settings", __name__, url_prefix="/machine-settings")

# Inference section as it was applied at boot (for restart_required).
_boot_inference = InferenceConfig(
    mode=app_config.sticker_inference_mode,
    device=app_config.device_mode,
    cuda_device_id=app_config.cuda_device_id,
    num_threads=app_config.inference_num_threads,
    timeout_s=app_config.inference_timeout_s,
    default_model_path=app_config.default_sticker_model_path,
    default_model_meta_path=app_config.default_sticker_model_meta_path,
)


def _restart_required(settings: MachineSettings) -> bool:
    return asdict(settings.connection) != asdict(boot_connection) or asdict(settings.inference) != asdict(_boot_inference)


def _response(settings: MachineSettings) -> dict:
    return {**settings.to_dict(), "restart_required": _restart_required(settings)}


# ── CRUD ────────────────────────────────────────────────────────────

@machine_settings_blueprint.get("")
@require_roles(UserRole.ADMIN)
def get_machine_settings():
    return jsonify(_response(machine_settings_repo.load_settings()))


@machine_settings_blueprint.put("")
@require_roles(UserRole.ADMIN)
def update_machine_settings():
    """Persist the full settings object and live-apply what can be applied."""
    payload = request.get_json(force=True) or {}
    try:
        new_settings = MachineSettings.from_dict(payload)
        machine_settings_repo.save_settings(new_settings)
        logger.info("[machine-settings] updated by user %s", g.current_user.username)
    except (TypeError, ValueError, KeyError) as exc:
        return jsonify({"error": f"Invalid settings: {exc}"}), 400

    if plc_worker is not None:
        try:
            plc_worker.apply_machine_settings(new_settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[machine-settings] failed to apply I/O settings to PLC worker: %s", exc)
    try:
        inspection_session_service.apply_machine_settings(new_settings)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[machine-settings] failed to apply timing settings: %s", exc)

    return jsonify(_response(new_settings))


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

    Body: { "address": 0, "duration_ms": 500 }
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
