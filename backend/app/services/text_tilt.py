from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _to_native(value: Any) -> Any:
    """Convert NumPy scalar/array to native Python type for JSON safety."""
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _normalize_rect_angle(rect) -> float:
    width, height = rect[1]
    angle = float(rect[2])
    if width < height:
        angle = 90.0 + angle
    if angle > 90.0:
        angle -= 180.0
    if angle < -90.0:
        angle += 180.0
    return angle


def _expand_roi(roi_frame: np.ndarray, padding_ratio: float) -> tuple[np.ndarray, int, int]:
    """Expand ROI frame with padding to catch edge/text-band that sticks out.
    Returns (expanded_frame, offset_x, offset_y) relative to original frame.
    """
    h, w = roi_frame.shape[:2]
    pad_x = max(1, int(w * padding_ratio))
    pad_y = max(1, int(h * padding_ratio))
    expanded = cv2.copyMakeBorder(
        roi_frame, pad_y, pad_y, pad_x, pad_x,
        borderType=cv2.BORDER_REPLICATE,
    )
    return expanded, pad_x, pad_y


def _check_text_band_in_roi(
    rect, roi_w: int, roi_h: int, offset_x: int, offset_y: int, tolerance_px: int,
) -> dict[str, Any]:
    """Check if text-band bounding box is within original ROI (with tolerance).
    Returns dict with native Python types only.
    """
    box = cv2.boxPoints(rect)
    # Convert to native float immediately to avoid np.float32 in JSON
    xs = [float(p[0]) - float(offset_x) for p in box]
    ys = [float(p[1]) - float(offset_y) for p in box]
    inside = 0
    max_exit = 0.0
    for x, y in zip(xs, ys):
        exit_x = max(0.0, -x, x - float(roi_w))
        exit_y = max(0.0, -y, y - float(roi_h))
        max_exit = max(max_exit, exit_x, exit_y)
        if x >= -tolerance_px and x <= roi_w + tolerance_px and y >= -tolerance_px and y <= roi_h + tolerance_px:
            inside += 1
    return {
        "corners_inside": int(inside),
        "corners_total": 4,
        "inside_ratio": float(inside) / 4.0,
        "max_exit_px": round(float(max_exit), 1),
        "is_inside": bool(inside == 4 and max_exit <= tolerance_px),
    }


def estimate_sticker_rotation(
    roi_frame,
    expected_tilt_degrees: float = 0.0,
    config: Any | None = None,
) -> dict[str, Any]:
    """Estimate sticker rotation using edge detection pipeline.

    Pipeline:
    1. Expand ROI with padding to catch edge/text-band that sticks out
    2. Grayscale + Gaussian blur
    3. Otsu thresholding (binary) — captures white text/logo
    4. Morphological Closing with horizontal kernel — merges characters into text bands
    5. Aspect Ratio filter — removes logo (low ratio), keeps text bands (high ratio)
    6. minAreaRect on longest text band → angle
    7. Check if text-band is within original ROI (with tolerance)

    All return values are native Python types (no NumPy scalars).
    """
    if roi_frame is None or getattr(roi_frame, "size", 0) == 0:
        return {
            "status": "unavailable",
            "angle_degrees": None,
            "expected_tilt_degrees": round(float(expected_tilt_degrees), 2),
            "deviation_degrees": None,
            "contour_area": None,
            "threshold_mode": None,
            "text_band_roi_check": None,
        }

    orig_h, orig_w = roi_frame.shape[:2]

    # Configurable parameters
    padding_ratio = float(getattr(config, "edge_search_padding_ratio", 0.10) or 0.10)
    tolerance_px = int(getattr(config, "edge_roi_tolerance_px", 10) or 10)
    horizontal_kernel_width = int(getattr(config, "morph_kernel_width", 40) or 40)
    horizontal_kernel_height = int(getattr(config, "morph_kernel_height", 5) or 5)
    min_aspect_ratio = float(getattr(config, "min_text_aspect_ratio", 3.0) or 3.0)
    min_contour_area_ratio = float(getattr(config, "min_text_contour_area_ratio", 0.001) or 0.001)
    min_area = max(1.0, float(orig_h * orig_w) * min_contour_area_ratio)

    # Step 1: Expand ROI with padding
    expanded, pad_x, pad_y = _expand_roi(roi_frame, padding_ratio)

    # Step 2: Grayscale + blur
    gray = cv2.cvtColor(expanded, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Otsu thresholding — white text on dark background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Morphological Closing with horizontal kernel
    h_kernel = np.ones((horizontal_kernel_height, horizontal_kernel_width), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, h_kernel, iterations=2)

    # Step 5: Find contours and filter by aspect ratio
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_bands = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect >= min_aspect_ratio:
            text_bands.append((contour, area, rect))

    if not text_bands:
        all_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not all_contours:
            return {
                "status": "unavailable",
                "angle_degrees": None,
                "expected_tilt_degrees": round(float(expected_tilt_degrees), 2),
                "deviation_degrees": None,
                "contour_area": None,
                "threshold_mode": "otsu_edge_fallback",
                "text_band_roi_check": None,
            }
        contour = max(all_contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        rect = cv2.minAreaRect(contour)
        angle = _normalize_rect_angle(rect)
        deviation = abs(angle - float(expected_tilt_degrees))
        roi_check = _check_text_band_in_roi(rect, orig_w, orig_h, pad_x, pad_y, tolerance_px)
        return {
            "status": "ok",
            "angle_degrees": round(float(angle), 2),
            "expected_tilt_degrees": round(float(expected_tilt_degrees), 2),
            "deviation_degrees": round(float(deviation), 2),
            "contour_area": round(float(area), 2),
            "threshold_mode": "otsu_edge_fallback",
            "text_band_roi_check": roi_check,
        }

    # Step 6: Pick the longest text band (highest area)
    best = max(text_bands, key=lambda x: x[1])
    contour, area, rect = best
    angle = _normalize_rect_angle(rect)
    deviation = abs(angle - float(expected_tilt_degrees))

    # Step 7: Check if text-band is within original ROI
    roi_check = _check_text_band_in_roi(rect, orig_w, orig_h, pad_x, pad_y, tolerance_px)

    return {
        "status": "ok",
        "angle_degrees": round(float(angle), 2),
        "expected_tilt_degrees": round(float(expected_tilt_degrees), 2),
        "deviation_degrees": round(float(deviation), 2),
        "contour_area": round(float(area), 2),
        "threshold_mode": "otsu_edge",
        "text_band_roi_check": roi_check,
    }


# Backward compatibility: keep old name as alias
estimate_white_text_tilt = estimate_sticker_rotation
