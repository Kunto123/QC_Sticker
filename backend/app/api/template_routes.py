from __future__ import annotations

import base64
import os

import cv2
import numpy as np
from flask import Blueprint, g, jsonify, request

from backend.app.core.container import templates_repo
from backend.app.services.gap_detector import save_ref_patch, get_ref_path
from backend.app.core.http import require_auth, require_roles
from backend.app.services.template_config_manager import TemplateConfigManager
from shared.contracts.enums import UserRole
from shared.contracts.templates import validate_sticker_rule


template_blueprint = Blueprint("templates", __name__, url_prefix="/templates")


def _parse_canny_bound(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _encode_ref_preview(save_path: str) -> str | None:
    try:
        with open(save_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")
    except OSError:
        return None


@template_blueprint.get("")
@require_auth
def list_templates():
    return jsonify(templates_repo.list_summaries())


@template_blueprint.get("/<int:template_id>")
@require_auth
def get_template(template_id: int):
    detail = templates_repo.get_template_detail(template_id)
    if detail is None:
        return jsonify({"error": "Template not found"}), 404
    return jsonify(detail)


@template_blueprint.get("/versions/<int:version_id>")
@require_auth
def get_template_version(version_id: int):
    detail = templates_repo.get_version_detail(version_id)
    if detail is None:
        return jsonify({"error": "Template version not found"}), 404
    return jsonify(detail)


@template_blueprint.get("/versions/<int:version_id>/runtime-template")
@require_auth
def get_runtime_template(version_id: int):
    template = templates_repo.get_by_version_id(version_id)
    if template is None:
        return jsonify({"error": "Template version not found"}), 404
    return jsonify(TemplateConfigManager.to_runtime_template(template))


@template_blueprint.post("")
@require_roles(UserRole.ADMIN)
def create_template():
    payload = request.get_json(force=True) or {}
    if isinstance(payload.get("sticker"), dict):
        errors = validate_sticker_rule(payload["sticker"])
        if errors:
            return jsonify({"error": "; ".join(errors)}), 400
    try:
        record = templates_repo.create_template(payload)
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(record), 201


@template_blueprint.put("/<int:template_id>")
@require_roles(UserRole.ADMIN)
def update_template(template_id: int):
    payload = request.get_json(force=True) or {}
    if isinstance(payload.get("sticker"), dict):
        errors = validate_sticker_rule(payload["sticker"])
        if errors:
            return jsonify({"error": "; ".join(errors)}), 400
    update_current = str(request.args.get("update_current") or "").lower() in ("1", "true", "yes")
    try:
        if update_current:
            record = templates_repo.update_current_version(template_id, payload)
        else:
            record = templates_repo.update_template(template_id, payload)
    except TypeError as exc:
        return jsonify({"error": str(exc)}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(record)


@template_blueprint.delete("/<int:template_id>")
@require_roles(UserRole.ADMIN)
def delete_template(template_id: int):
    ok = templates_repo.delete_template(template_id)
    if not ok:
        return jsonify({"error": "Template not found"}), 404
    return jsonify({"deleted": True, "id": template_id})


@template_blueprint.get("/<int:template_id>/versions")
@require_auth
def list_template_versions(template_id: int):
    try:
        versions = templates_repo.list_versions(template_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(versions)


@template_blueprint.post("/<int:template_id>/transition")
@require_roles(UserRole.ADMIN)
def transition_template_lifecycle(template_id: int):
    payload = request.get_json(force=True) or {}
    new_status = str(payload.get("status") or "").strip().lower()
    if not new_status:
        return jsonify({"error": "status is required"}), 400
    change_note = str(payload.get("change_note") or "").strip() or None
    actor = getattr(g, "current_user", None)
    try:
        result = templates_repo.transition_lifecycle(
            template_id,
            new_status,
            actor_id=actor.id if actor else None,
            actor_username=actor.username if actor else None,
            change_note=change_note,
        )
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    return jsonify(result)


@template_blueprint.post("/<int:template_id>/part-ready-ref/capture")
@require_roles(UserRole.ADMIN)
def capture_part_ready_ref(template_id: int):
    """Capture reference gap patch from a calibration frame."""
    payload = request.get_json(force=True) or {}
    frame_b64 = str(payload.get("frame_b64") or "")
    roi = payload.get("roi") or {}
    canny_low = _parse_canny_bound(payload.get("canny_low"))
    canny_high = _parse_canny_bound(payload.get("canny_high"))
    if not frame_b64:
        return jsonify({"error": "frame_b64 required"}), 400
    try:
        raw = base64.b64decode(frame_b64)
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400
    except Exception as exc:
        return jsonify({"error": f"Decode failed: {exc}"}), 400

    save_path = str(get_ref_path(template_id))
    ok, err_msg = save_ref_patch(frame, roi, save_path, canny_low=canny_low, canny_high=canny_high)
    if ok:
        # Update template config with ref_path — use update_current_version with full detail
        try:
            detail = templates_repo.get_template_detail(template_id)
            if detail:
                pr = dict(detail.get("part_ready") or {})
                pr["gap_ref_path"] = save_path
                pr["method"] = "gap_template_match"
                pr["canny_low"] = canny_low
                pr["canny_high"] = canny_high
                detail["part_ready"] = pr
                templates_repo.update_current_version(template_id, detail)
        except Exception:
            pass  # non-critical
        return jsonify({
            "saved": True,
            "path": save_path,
            "preview_b64": _encode_ref_preview(save_path),
        }), 201
    reason = err_msg or "check ROI and camera"
    return jsonify({"error": f"Failed to extract gap patch — {reason}"}), 400


@template_blueprint.post("/<int:template_id>/part-ready-ref/upload")
@require_roles(UserRole.ADMIN)
def upload_part_ready_ref(template_id: int):
    """Upload reference patch image (user provides the patch directly)."""
    file_bytes = None
    canny_low = _parse_canny_bound(request.args.get("canny_low") or request.form.get("canny_low"))
    canny_high = _parse_canny_bound(request.args.get("canny_high") or request.form.get("canny_high"))

    # Try multipart file upload first
    if "file" in request.files:
        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "Empty filename"}), 400
        file_bytes = file.read()
    else:
        # Fallback: accept base64 JSON from local mode client
        payload = request.get_json(silent=True) or {}
        file_b64 = str(payload.get("file_b64") or "")
        canny_low = canny_low if canny_low is not None else _parse_canny_bound(payload.get("canny_low"))
        canny_high = canny_high if canny_high is not None else _parse_canny_bound(payload.get("canny_high"))
        if file_b64:
            try:
                file_bytes = base64.b64decode(file_b64)
            except Exception:
                return jsonify({"error": "Invalid base64"}), 400
        else:
            return jsonify({"error": "No file uploaded"}), 400

    save_path = str(get_ref_path(template_id))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image file"}), 400
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        from backend.app.services.gap_detector import _auto_canny
        edge_map = _auto_canny(gray, low=canny_low, high=canny_high)
        cv2.imwrite(save_path, edge_map)
    except Exception as exc:
        return jsonify({"error": f"Save failed: {exc}"}), 400

    # Persist gap_ref_path to template JSON
    try:
        detail = templates_repo.get_template_detail(template_id)
        if detail:
            pr = dict(detail.get("part_ready") or {})
            pr["gap_ref_path"] = save_path
            pr["method"] = "gap_template_match"
            pr["canny_low"] = canny_low
            pr["canny_high"] = canny_high
            detail["part_ready"] = pr
            templates_repo.update_current_version(template_id, detail)
    except Exception:
        pass  # non-critical

    return jsonify({
        "saved": True,
        "path": save_path,
        "preview_b64": _encode_ref_preview(save_path),
    }), 201


@template_blueprint.get("/<int:template_id>/part-ready-ref")
@require_roles(UserRole.ADMIN)
def get_part_ready_ref(template_id: int):
    """Return the currently saved reference edge-map (base64 PNG), if any."""
    ref_path = get_ref_path(template_id)
    if not ref_path.is_file():
        return jsonify({"exists": False, "preview_b64": None}), 200
    return jsonify({"exists": True, "preview_b64": _encode_ref_preview(str(ref_path))}), 200


@template_blueprint.delete("/<int:template_id>/part-ready-ref")
@require_roles(UserRole.ADMIN)
def delete_part_ready_ref(template_id: int):
    """Delete reference patch for a template."""
    ref_path = get_ref_path(template_id)
    if ref_path.exists():
        ref_path.unlink()
        return jsonify({"deleted": True}), 200
    return jsonify({"error": "No reference found"}), 404


@template_blueprint.post("/<int:template_id>/rollback")
@require_roles(UserRole.ADMIN)
def rollback_template_version(template_id: int):
    payload = request.get_json(force=True) or {}
    version_id = payload.get("version_id")
    if not version_id:
        return jsonify({"error": "version_id is required"}), 400
    try:
        result = templates_repo.rollback_version(template_id, int(version_id))
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    return jsonify(result)


