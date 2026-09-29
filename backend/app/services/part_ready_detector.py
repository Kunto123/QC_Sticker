from __future__ import annotations

from typing import Any

import cv2

from shared.contracts.enums import DecisionCode, RejectReasonCode


def compute_mean_std_thresholds(empty_mean: float, part_mean: float,
                               part_std: float, sticker_std: float) -> dict[str, float]:
    """Auto-hitung MEAN_MAX dan STD_MAX dari 3 kondisi kalibrasi.

    Returns dict berisi mean_max dan std_max hasil hitung (titik tengah antar kondisi).
    """
    mean_max = round((empty_mean + part_mean) / 2.0, 2)
    std_max = round((part_std + sticker_std) / 2.0, 2)
    return {"mean_max": mean_max, "std_max": std_max}


def evaluate_mean_std_threshold(frame, config) -> dict[str, Any]:
    """Evaluasi kesiapan part pakai threshold mean dan std.

    Hitung mean dan standard deviation grayscale dari ROI, lalu klasifikasikan
    region ke salah satu dari tiga kondisi:
      - empty (mean > MEAN_MAX): tidak ada part
      - part_normal (std <= STD_MAX && mean <= MEAN_MAX): part hitam polos
      - sticker (std > STD_MAX && mean <= MEAN_MAX): part dengan sticker

    Return dict evaluasi part-ready yang konsisten dengan metode lainnya.
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

    # Konversi ke grayscale untuk analisis statistik
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

    # Logika keputusan — klasifikasi tiga arah
    if mean_val > mean_max:
        # ROI terang = background jig terlihat = tidak ada part
        part_ready = False
        condition = "empty"
        match_ratio = 0.0
    elif std_val > std_max:
        # Mean rendah + std tinggi = part hitam dengan sticker putih (kontras tinggi)
        part_ready = True
        condition = "sticker"
        # Confidence: 0.5 di batas std=std_max, 1.0 di ~2x threshold.
        # Ini memastikan titik operasi sticker punya confidence >= 0.5,
        # yaitu threshold default min_match_ratio.
        match_ratio = min(1.0, std_val / (std_max * 2.0))
    else:
        # Mean rendah + std rendah = part hitam seragam
        part_ready = True
        condition = "part_normal"
        # Confidence: seberapa jauh di bawah kedua threshold (0 di batas, 1 di sempurna)
        mean_conf = 1.0 - min(1.0, mean_val / mean_max)
        std_conf = 1.0 - min(1.0, std_val / std_max)
        match_ratio = (mean_conf + std_conf) / 2.0

    # min_match_ratio: confidence minimum (0-1) yang dibutuhkan agar part dianggap ready
    # Default 0.5 berarti minimal butuh confidence 50%
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
