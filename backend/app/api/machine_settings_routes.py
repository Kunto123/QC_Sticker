"""API Machine Settings — CRUD + diagnostik PLC.

GET  /machine-settings                  → settings saat ini (+ restart_required)
PUT  /machine-settings                  → simpan + terapkan langsung (admin)
GET  /machine-settings/plc/diagnostics  → status PLC live + snapshot input
POST /machine-settings/plc/test-coil    → pulse coil untuk test wiring (admin)
POST /machine-settings/plc/all-off      → matikan semua coil darurat (admin)

`machine_settings.json` adalah satu-satunya sumber untuk nilai-nilai ini (tidak
ada yang dari `.env`). `io` dan `timing` diterapkan langsung; `connection` dan
`inference` butuh restart backend — response memberitahu ini lewat `restart_required`.
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

# Bagian inference sebagaimana diterapkan saat boot (untuk restart_required).
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
    """Simpan seluruh object settings dan terapkan langsung apa yang bisa."""
    payload = request.get_json(force=True) or {}
    try:
        new_settings = MachineSettings.from_dict(payload)
        machine_settings_repo.save_settings(new_settings)
        logger.info("[machine-settings] updated by user %s", g.current_user.username)
    except (TypeError, ValueError, KeyError) as exc:
        return jsonify({"error": f"Settings tidak valid: {exc}"}), 400

    if plc_worker is not None:
        try:
            plc_worker.apply_machine_settings(new_settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[machine-settings] failed to apply I/O settings to PLC worker: %s", exc)
    try:
        inspection_session_service.apply_machine_settings(new_settings)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[machine-settings] failed to apply timing settings: %s", exc)

    # identity.line is live-appliable (unlike connection/inference): the
    # /auth/login operator gate reads app_config.machine_line_id directly,
    # so it must be refreshed here too — not just on InspectionSessionService
    # above — or a Line edit only takes effect after a backend restart.
    app_config.machine_line_id = str(getattr(new_settings.identity, "line", "") or "").strip()

    return jsonify(_response(new_settings))


# ── Diagnostik PLC ─────────────────────────────────────────────────

@machine_settings_blueprint.get("/plc/diagnostics")
@require_roles(UserRole.ADMIN)
def plc_diagnostics():
    """Balikin status PLC live termasuk snapshot input."""
    if plc_worker is None:
        # "PLC worker is disabled" TIDAK BOLEH diterjemahkan — dicocokkan
        # persis oleh client (operator/view.py::_friendly_error).
        return jsonify({"enabled": False, "note": "PLC worker is disabled"})
    return jsonify({"enabled": True, **plc_worker.status()})


@machine_settings_blueprint.post("/plc/test-coil")
@require_roles(UserRole.ADMIN)
def plc_test_coil():
    """Pulse satu coil untuk verifikasi wiring.

    Body: { "address": 0, "duration_ms": 500 }
    Keamanan: cuma jalan kalau dry_run=True atau sudah dikonfirmasi eksplisit.
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
            "error": "dry_run is False. Ini akan menyalakan coil sungguhan. Kirim confirm=yes untuk lanjut.",
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
    """Darurat: matikan semua coil."""
    if plc_worker is None:
        return jsonify({"error": "PLC worker is disabled"}), 400
    try:
        plc_worker._all_off("admin_all_off")
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
