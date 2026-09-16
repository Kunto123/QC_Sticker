from __future__ import annotations

import base64

from flask import Blueprint, jsonify, request
import cv2
import numpy as np

from backend.app.core.container import profiles_repo
from backend.app.core.http import require_auth, require_roles
from backend.app.services.calibration import CalibrationService
from shared.contracts.enums import UserRole


calibration_blueprint = Blueprint("calibration", __name__, url_prefix="/calibration")


@calibration_blueprint.post("/color-profile")
@require_roles(UserRole.ADMIN)
def compute_color_profile():
    payload = request.get_json(force=True) or {}
    try:
        image = CalibrationService.decode_image(str(payload.get("image_b64") or ""))
        roi = payload.get("roi")
        image = CalibrationService.apply_roi(image, roi)
        profile = CalibrationService.compute_color_profile(
            image,
            str(payload.get("colorspace") or "LAB"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(profile)


@calibration_blueprint.get("/profiles")
@require_auth
def list_profiles():
    return jsonify(profiles_repo.list_profiles())


@calibration_blueprint.post("/profiles")
@require_roles(UserRole.ADMIN)
def create_profile():
    payload = request.get_json(force=True) or {}
    expiry_raw = payload.get("expiry_interval_days")
    expiry_days: int | None = None
    if expiry_raw is not None:
        try:
            expiry_days = max(1, int(expiry_raw))
        except (ValueError, TypeError):
            return jsonify({"error": "expiry_interval_days must be a positive integer"}), 400
    try:
        record = profiles_repo.create(
            str(payload.get("name") or "").strip() or "Unnamed Profile",
            dict(payload.get("profile") or {}),
            scope_line_id=str(payload.get("scope_line_id") or "").strip() or None,
            scope_station_id=str(payload.get("scope_station_id") or "").strip() or None,
            scope_part_name=str(payload.get("scope_part_name") or "").strip() or None,
            expiry_interval_days=expiry_days,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(record), 201


@calibration_blueprint.put("/profiles/<int:profile_id>")
@require_roles(UserRole.ADMIN)
def update_profile(profile_id: int):
    payload = request.get_json(force=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be an object"}), 400

    updates: dict = {}
    for field in ("name", "profile", "scope_line_id", "scope_station_id", "scope_part_name", "expiry_interval_days"):
        if field in payload:
            updates[field] = payload.get(field)

    if not updates:
        return jsonify({"error": "At least one field must be provided"}), 400

    try:
        record = profiles_repo.update(profile_id, **updates)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    return jsonify(record)


@calibration_blueprint.get("/profiles/active")
@require_auth
def get_active_profile():
    """Return the most recent non-expired profile for the given scope.

    Query params: line_id, station_id, part_name (all optional)
    """
    line_id = str(request.args.get("line_id") or "").strip() or None
    station_id = str(request.args.get("station_id") or "").strip() or None
    part_name = str(request.args.get("part_name") or "").strip() or None
    record = profiles_repo.get_active_for_scope(
        line_id=line_id,
        station_id=station_id,
        part_name=part_name,
    )
    if record is None:
        return jsonify({"error": "No active calibration profile found for the given scope"}), 404
    return jsonify(record)


@calibration_blueprint.delete("/profiles/<int:profile_id>")
@require_roles(UserRole.ADMIN)
def delete_profile(profile_id: int):
    ok = profiles_repo.delete(profile_id)
    if not ok:
        return jsonify({"error": "Profile not found"}), 404
    return jsonify({"deleted": True, "id": profile_id})


@calibration_blueprint.post("/mean-std-threshold")
@require_roles(UserRole.ADMIN)
def compute_mean_std_threshold():
    """Compute MEAN_MAX and STD_MAX from 3 calibration images.

    Expected JSON payload:
    {
      "empty": "<base64 image — ROI kosong, no part>",
      "part": "<base64 image — part hitam polos>",
      "sticker": "<base64 image — part + sticker>"
    }

    Returns computed thresholds and per-condition statistics.
    """
    payload = request.get_json(force=True) or {}
    required_keys = ("empty", "part", "sticker")
    missing = [k for k in required_keys if not payload.get(k)]
    if missing:
        return jsonify({"error": f"Missing required images: {missing}"}), 400

    def _decode_and_stats(b64_str: str) -> dict[str, float]:
        raw = base64.b64decode(b64_str)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return {"mean": round(float(gray.mean()), 4), "std": round(float(gray.std()), 4)}

    try:
        empty_stats = _decode_and_stats(str(payload["empty"]))
        part_stats = _decode_and_stats(str(payload["part"]))
        sticker_stats = _decode_and_stats(str(payload["sticker"]))
    except (ValueError, Exception) as exc:
        return jsonify({"error": f"Image processing failed: {exc}"}), 400

    mean_max = round((empty_stats["mean"] + part_stats["mean"]) / 2.0, 2)
    std_max = round((part_stats["std"] + sticker_stats["std"]) / 2.0, 2)

    return jsonify({
        "mean_max": mean_max,
        "std_max": std_max,
        "conditions": {
            "empty": empty_stats,
            "part": part_stats,
            "sticker": sticker_stats,
        },
        "gaps": {
            "mean_gap": round(empty_stats["mean"] - part_stats["mean"], 2),
            "std_gap": round(sticker_stats["std"] - part_stats["std"], 2),
        },
        "recommendation": {
            "mean_max_safe": round(mean_max * 0.9, 2),
            "mean_max_tight": round(mean_max * 1.1, 2),
            "std_max_safe": round(std_max * 0.9, 2),
            "std_max_tight": round(std_max * 1.1, 2),
        },
    })

