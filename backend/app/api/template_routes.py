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
from shared.contracts.templates import normalize_mode, validate_criteria
from backend.app.services.anomaly_backend import get_scorer, SimpleAnomalyScorer


template_blueprint = Blueprint("templates", __name__, url_prefix="/templates")


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
    # Normalize mode before validation
    _mode = normalize_mode(payload.get("mode"))
    if _mode and _mode not in ("sticker", "counter", "defect"):
        return jsonify({"error": f"Unknown mode {_mode!r}. Must be one of: sticker, counter, defect"}), 400
    _criteria = payload.get("criteria") or {}
    if _mode and _criteria:
        errors = validate_criteria(_mode, _criteria)
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
    # Normalize mode and validate criteria (same as create)
    _mode = normalize_mode(payload.get("mode"))
    if _mode and _mode not in ("sticker", "counter", "defect"):
        return jsonify({"error": f"Unknown mode {_mode!r}. Must be one of: sticker, counter, defect"}), 400
    _criteria = payload.get("criteria") or {}
    if _mode and _criteria:
        errors = validate_criteria(_mode, _criteria)
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
    _hsv_lower = np.array(roi.get("gap_hsv_lower", [90, 50, 50]))
    _hsv_upper = np.array(roi.get("gap_hsv_upper", [130, 255, 255]))
    _padding = int(roi.get("gap_padding_px", 20))
    _rotation = float(roi.get("rotation", 0.0) or 0.0)

    ok, err_msg = save_ref_patch(frame, roi, save_path, _hsv_lower, _hsv_upper, _padding, _rotation)
    if ok:
        # Update template config with ref_path — use update_current_version with full detail
        try:
            detail = templates_repo.get_template_detail(template_id)
            if detail:
                pr = dict(detail.get("part_ready") or {})
                pr["gap_ref_path"] = save_path
                pr["method"] = "gap_template_match"
                detail["part_ready"] = pr
                templates_repo.update_current_version(template_id, detail)
        except Exception:
            pass  # non-critical
        return jsonify({"saved": True, "path": save_path}), 201
    reason = err_msg or "check ROI and camera"
    return jsonify({"error": f"Failed to extract gap patch — {reason}"}), 400


@template_blueprint.post("/<int:template_id>/part-ready-ref/upload")
@require_roles(UserRole.ADMIN)
def upload_part_ready_ref(template_id: int):
    """Upload reference patch image (user provides the patch directly)."""
    file_bytes = None

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
        edge_map = _auto_canny(gray)
        cv2.imwrite(save_path, edge_map)
    except Exception as exc:
        return jsonify({"error": f"Save failed: {exc}"}), 400

    # Persist gap_ref_path to template JSON
    try:
        detail = templates_repo.get_template_detail(template_id)
        if detail:
            pr = dict(detail.get("part_ready") or {})
            pr["gap_ref_path"] = save_path
            pr["gap_ref_type"] = "edge_map"
            pr["method"] = "gap_template_match"
            detail["part_ready"] = pr
            templates_repo.update_current_version(template_id, detail)
    except Exception:
        pass  # non-critical

    return jsonify({"saved": True, "path": save_path}), 201


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


@template_blueprint.post("/<int:template_id>/defect-calibrate")
@require_roles(UserRole.ADMIN)
def calibrate_defect_threshold(template_id: int):
    """Calibrate defect thresholds for a defect-mode template.

    Accepts a list of known-good frame crops (base64-encoded images).
    Returns suggested thresholds per ROI.
    """
    payload = request.get_json(force=True) or {}
    frames_b64: list[str] = payload.get("frames_b64") or []
    if len(frames_b64) < 3:
        return jsonify({"error": "Need at least 3 known-good frames for calibration"}), 400
    frames = []
    for fb64 in frames_b64:
        try:
            raw = base64.b64decode(fb64)
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
        except Exception:
            continue
    if len(frames) < 3:
        return jsonify({"error": "Failed to decode enough frames"}), 400
    detail = templates_repo.get_template_detail(template_id)
    if not detail:
        return jsonify({"error": "Template not found"}), 404
    criteria = detail.get("criteria") or {}
    rois = criteria.get("rois", [])
    if not rois:
        return jsonify({"error": "Template has no defect ROIs configured"}), 400
    inference_mode = str(criteria.get("inference_mode", "whole_part")).strip().lower()
    default_model = criteria.get("default_model_path")
    fh, fw = frames[0].shape[:2]

    if inference_mode == "whole_part" and not any(r.get("model_path") for r in rois):
        # Whole-part calibration: single inference on union region, slice per ROI
        from backend.app.services.evaluators.defect import _build_union_bbox, _parse_geometry, _aggregate_score
        import logging
        logger = logging.getLogger(__name__)

        ux, uy, uw, uh = _build_union_bbox(rois, fw, fh)
        if uw <= 0 or uh <= 0:
            return jsonify({"error": "Union region is empty"}), 400

        union_crops = [fr[uy:uy+uh, ux:ux+uw] for fr in frames]
        union_crops = [c for c in union_crops if c.size > 0]
        if len(union_crops) < 3:
            return jsonify({"error": "Not enough valid union crops"}), 400

        scorer = get_scorer(default_model)
        # For each frame, score the union region once, then collect per-ROI slice scores
        roi_scores: dict[str, list[float]] = {r.get("name", f"ROI {i}"): [] for i, r in enumerate(rois)}
        for uc in union_crops:
            try:
                _, heatmap = scorer.score(uc)
            except Exception as exc:
                logger.warning("[calibrate] frame score failed: %s", exc)
                continue
            if heatmap is None or heatmap.size == 0:
                continue
            for i, roi in enumerate(rois):
                name = roi.get("name", f"ROI {i}")
                geom = roi.get("geometry", {})
                rx, ry, rw, rh = _parse_geometry(geom, fw, fh)
                # Map ROI coords to union coords
                hs_y = max(0, ry - uy)
                hs_x = max(0, rx - ux)
                hs_h = min(rh, heatmap.shape[0] - hs_y) if heatmap is not None else 0
                hs_w = min(rw, heatmap.shape[1] - hs_x) if heatmap is not None else 0
                if hs_h <= 0 or hs_w <= 0:
                    continue
                slice_ = heatmap[hs_y:hs_y+hs_h, hs_x:hs_x+hs_w]
                score = _aggregate_score(slice_, criteria.get("aggregation", "p99"))
                roi_scores[name].append(score)

        results = []
        for i, roi in enumerate(rois):
            name = roi.get("name", f"ROI {i}")
            scores = roi_scores.get(name, [])
            if len(scores) < 3:
                results.append({"name": name, "error": f"Not enough valid scores ({len(scores)})"})
                continue
            import numpy as np
            scores_arr = np.array(scores)
            mean_score = float(scores_arr.mean())
            std_score = float(scores_arr.std())
            suggested_threshold = round(mean_score + 3.0 * std_score, 4)
            results.append({
                "name": name,
                "threshold": round(suggested_threshold, 4),
                "mean_score": round(mean_score, 4),
                "std_score": round(std_score, 4),
                "num_samples": len(scores),
                "scores": [round(s, 4) for s in sorted(scores)],
            })
        logger.info(
            "[calibrate] whole_part: %d ROIs calibrated from %d frames (union %dx%d)",
            len(results), len(union_crops), uw, uh,
        )
        return jsonify({"results": results, "inference_mode": "whole_part"}), 200
    else:
        # Per-ROI crop calibration (legacy or override)
        results = []
        for roi in rois:
            name = roi.get("name", "ROI")
            geom = roi.get("geometry", {})
            x = int(float(geom.get("x", 0.0)) * fw)
            y = int(float(geom.get("y", 0.0)) * fh)
            w = max(1, int(float(geom.get("w", 1.0)) * fw))
            h = max(1, int(float(geom.get("h", 1.0)) * fh))
            x = max(0, min(fw - 1, x))
            y = max(0, min(fh - 1, y))
            w = min(fw - x, w)
            h = min(fh - y, h)
            crops = [fr[y:y+h, x:x+w] for fr in frames]
            crops = [c for c in crops if c.size > 0]
            if len(crops) < 3:
                results.append({"name": name, "error": "Not enough valid crops"})
                continue
            scorer = SimpleAnomalyScorer()
            calib_result = scorer.calibrate(crops)
            results.append({"name": name, **calib_result})
        return jsonify({"results": results, "inference_mode": "per_roi_crop"}), 200
