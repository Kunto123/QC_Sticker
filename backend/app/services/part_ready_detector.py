from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from backend.app.services.calibration import CalibrationService
from shared.contracts.enums import DecisionCode, RejectReasonCode


def _hsv_bounds(values: Any, fallback: tuple[int, int, int]) -> np.ndarray:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return np.array(fallback, dtype=np.uint8)
    parsed: list[int] = []
    for index, value in enumerate(values):
        upper = 180 if index == 0 else 255
        try:
            parsed_value = int(value)
        except (TypeError, ValueError):
            parsed_value = fallback[index]
        parsed.append(max(0, min(upper, parsed_value)))
    return np.array(parsed, dtype=np.uint8)


def evaluate_hsv_black_ratio(frame, config: Any) -> dict[str, Any]:
    if frame is None or getattr(frame, "size", 0) == 0:
        ratio = 0.0
    else:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = _hsv_bounds(getattr(config, "hsv_lower", None), (0, 0, 0))
        upper = _hsv_bounds(getattr(config, "hsv_upper", None), (180, 255, 80))
        mask = cv2.inRange(hsv, lower, upper)
        ratio = float(np.count_nonzero(mask) / max(1, mask.size))

    min_ratio = float(getattr(config, "min_match_ratio", None) or 0.75)
    ready = ratio >= min_ratio
    return {
        "enabled": True,
        "method": "hsv_black_ratio",
        "part_ready": ready,
        "part_ready_confidence": round(ratio, 6),
        "decision": DecisionCode.ACCEPT.value if ready else DecisionCode.REJECT.value,
        "reject_reason_code": None if ready else RejectReasonCode.PART_NOT_READY.value,
        "status": "ready" if ready else "not_ready",
        "match_ratio": round(ratio, 6),
        "raw_match_ratio": round(ratio, 6),
        "mean_distance": None,
        "distance_threshold": None,
        "min_match_ratio": min_ratio,
        "color_profile_id": getattr(config, "color_profile_id", None),
        "colorspace": "HSV",
        "hsv_lower": list(_hsv_bounds(getattr(config, "hsv_lower", None), (0, 0, 0)).tolist()),
        "hsv_upper": list(_hsv_bounds(getattr(config, "hsv_upper", None), (180, 255, 80)).tolist()),
    }


def evaluate_color_profile_match(
    frame,
    *,
    config: Any,
    profile: dict[str, Any],
) -> dict[str, Any]:
    evaluation = CalibrationService.evaluate_color_match(
        frame,
        profile,
        colorspace=config.colorspace,
        distance_threshold=config.distance_threshold,
        min_match_ratio=config.min_match_ratio,
    )
    ready = bool(evaluation["is_match"])
    return {
        "enabled": True,
        "method": "color_profile_match",
        "part_ready": ready,
        "part_ready_confidence": float(evaluation["match_ratio"]),
        "decision": DecisionCode.ACCEPT.value if ready else DecisionCode.REJECT.value,
        "reject_reason_code": None if ready else RejectReasonCode.PART_NOT_READY.value,
        "status": "ready" if ready else "not_ready",
        "match_ratio": float(evaluation["match_ratio"]),
        "raw_match_ratio": float(evaluation["match_ratio"]),
        "mean_distance": evaluation["mean_distance"],
        "distance_threshold": evaluation["distance_threshold"],
        "min_match_ratio": float(evaluation["min_match_ratio"]),
        "color_profile_id": config.color_profile_id,
        "colorspace": evaluation["colorspace"],
    }


def compute_mean_std_thresholds(empty_mean: float, part_mean: float,
                               part_std: float, sticker_std: float) -> dict[str, float]:
    """Auto-compute MEAN_MAX and STD_MAX from 3 calibration conditions.

    Returns dict with computed mean_max and std_max (midpoints between conditions).
    """
    mean_max = round((empty_mean + part_mean) / 2.0, 2)
    std_max = round((part_std + sticker_std) / 2.0, 2)
    return {"mean_max": mean_max, "std_max": std_max}


def compute_hsv_reference_from_roi(frame) -> dict[str, Any]:
    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError("Invalid image for HSV reference.")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean = hsv.reshape(-1, 3).mean(axis=0)
    std = hsv.reshape(-1, 3).std(axis=0)
    lower = [
        max(0, int(round(mean[0] - max(8.0, std[0] * 2.0)))),
        max(0, int(round(mean[1] - max(30.0, std[1] * 2.0)))),
        max(0, int(round(mean[2] - max(30.0, std[2] * 2.0)))),
    ]
    upper = [
        min(180, int(round(mean[0] + max(8.0, std[0] * 2.0)))),
        min(255, int(round(mean[1] + max(30.0, std[1] * 2.0)))),
        min(255, int(round(mean[2] + max(30.0, std[2] * 2.0)))),
    ]
    return {
        "method": "hsv_black_ratio",
        "hsv_lower": lower,
        "hsv_upper": upper,
        "min_match_ratio": 0.75,
        "sampling_meta": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
    }


def evaluate_mean_std_threshold(frame, config) -> dict[str, Any]:
    """Evaluate part readiness using mean and std thresholds.

    Computes grayscale mean and standard deviation from the ROI,
    then classifies the region into one of three conditions:
      - empty (mean > MEAN_MAX): no part present
      - part_normal (std <= STD_MAX && mean <= MEAN_MAX): black part only
      - sticker (std > STD_MAX && mean <= MEAN_MAX): part with sticker

    Returns a part-ready evaluation dict consistent with the other methods.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return {
            "enabled": True,
            "method": "mean_std_threshold",
            "part_ready": False,
            "part_ready_confidence": 0.0,
            "decision": DecisionCode.REJECT.value,
            "reject_reason_code": RejectReasonCode.PART_NOT_READY.value,
            "status": "error",
            "match_ratio": 0.0,
            "raw_match_ratio": 0.0,
            "mean_value": 0.0,
            "std_value": 0.0,
            "mean_max": float(getattr(config, "mean_max", 105.0) or 105.0),
            "std_max": float(getattr(config, "std_max", 35.0) or 35.0),
            "condition": "error",
        }

    # Convert to grayscale for statistical analysis
    if len(frame.shape) == 3 and frame.shape[2] >= 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif len(frame.shape) == 2:
        gray = frame
    else:
        return {
            "enabled": True,
            "method": "mean_std_threshold",
            "part_ready": False,
            "part_ready_confidence": 0.0,
            "decision": DecisionCode.REJECT.value,
            "reject_reason_code": RejectReasonCode.PART_NOT_READY.value,
            "status": "error",
            "match_ratio": 0.0,
            "raw_match_ratio": 0.0,
            "mean_value": 0.0,
            "std_value": 0.0,
            "mean_max": float(getattr(config, "mean_max", 105.0) or 105.0),
            "std_max": float(getattr(config, "std_max", 35.0) or 35.0),
            "condition": "error",
        }

    mean_val = float(gray.mean())
    std_val = float(gray.std())
    mean_max = float(getattr(config, "mean_max", 105.0) or 105.0)
    std_max = float(getattr(config, "std_max", 35.0) or 35.0)

    # Decision logic — three-way classification
    if mean_val > mean_max:
        # ROI bright = background jig visible = no part
        part_ready = False
        condition = "empty"
        match_ratio = 0.0
    elif std_val > std_max:
        # Low mean + high std = black part with white sticker (high contrast)
        part_ready = True
        condition = "sticker"
        # Confidence: 0.5 at std=std_max boundary, 1.0 at ~2x threshold.
        # This ensures the sticker operating point has confidence >= 0.5,
        # which is the default min_match_ratio threshold.
        match_ratio = min(1.0, std_val / (std_max * 2.0))
    else:
        # Low mean + low std = uniform black part
        part_ready = True
        condition = "part_normal"
        # Confidence: how far below both thresholds (0 at boundary, 1 at perfect)
        mean_conf = 1.0 - min(1.0, mean_val / mean_max)
        std_conf = 1.0 - min(1.0, std_val / std_max)
        match_ratio = (mean_conf + std_conf) / 2.0

    # min_match_ratio: minimum confidence (0-1) required to consider part ready
    # Default 0.5 means at least 50% confidence needed
    _min_confidence = float(getattr(config, "min_match_ratio", None) or 0.5)

    return {
        "enabled": True,
        "method": "mean_std_threshold",
        "part_ready": part_ready,
        "part_ready_confidence": round(match_ratio, 4),
        "decision": DecisionCode.ACCEPT.value if part_ready else DecisionCode.REJECT.value,
        "reject_reason_code": None if part_ready else RejectReasonCode.PART_NOT_READY.value,
        "status": "ready" if part_ready else "not_ready",
        "match_ratio": round(match_ratio, 4),
        "raw_match_ratio": round(match_ratio, 4),
        "mean_value": round(mean_val, 4),
        "std_value": round(std_val, 4),
        "mean_max": mean_max,
        "std_max": std_max,
        "min_match_ratio": _min_confidence,
        "condition": condition,
    }
