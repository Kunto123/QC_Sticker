"""Gap detection service — part ready via template matching.

Detects the blue clamp using HSV segmentation, extracts the gap area as a
template patch from a reference image, and uses cv2.matchTemplate to confirm
the part is at the correct position within the part_ready ROI.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.app.core.config import PROJECT_ROOT as project_root

_logger = logging.getLogger(__name__)
# Default HSV range for blue clamp detection
DEFAULT_HSV_LOWER = np.array([90, 50, 50])
DEFAULT_HSV_UPPER = np.array([130, 255, 255])

# Reference storage directory
PART_READY_REF_DIR = "backend/app/assets/part_ready_refs"

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def _auto_canny(gray: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Auto-tune Canny thresholds from image median — robust terhadap perubahan cahaya."""
    gray = _clahe.apply(gray)   # normalkan brightness lokal sebelum hitung threshold
    median = float(np.median(gray))
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, low, high)


def get_ref_path(template_id: int) -> Path:
    """Get the file path for a template's reference patch."""
    ref_dir = Path(project_root) / PART_READY_REF_DIR
    ref_dir.mkdir(parents=True, exist_ok=True)
    return ref_dir / f"{template_id}.png"


def get_logo_ref_path(template_id: int) -> Path:
    """Get the file path for a template's logo reference patch."""
    ref_dir = Path(project_root) / PART_READY_REF_DIR
    ref_dir.mkdir(parents=True, exist_ok=True)
    return ref_dir / f"{template_id}_logo.png"


def load_logo_ref(ref_path: str | None = None, template_id: int | None = None) -> np.ndarray | None:
    """Load logo reference edge map from disk. Returns None if file missing."""
    if ref_path:
        p = Path(ref_path)
        if p.is_file():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            return img if img is not None else None
    if template_id is not None:
        p = get_logo_ref_path(template_id)
        if p.is_file():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            return img if img is not None else None
    return None


def save_logo_ref(frame_bgr: np.ndarray, roi: dict, save_path: str) -> tuple[bool, str]:
    """Crop logo ROI from frame, convert to edge map, save as PNG."""
    try:
        rx, ry = int(roi.get("x", 0)), int(roi.get("y", 0))
        rw, rh = int(roi.get("w", 0)), int(roi.get("h", 0))
        if rw <= 0 or rh <= 0:
            return False, f"ROI dimensions invalid: w={rw} h={rh}"
        fh, fw = frame_bgr.shape[:2]
        rx = max(0, min(rx, fw - 1))
        ry = max(0, min(ry, fh - 1))
        rw = min(rw, fw - rx)
        rh = min(rh, fh - ry)
        roi_frame = frame_bgr[ry:ry+rh, rx:rx+rw]
        if roi_frame.size == 0:
            return False, "ROI region empty"
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        edge_map = _auto_canny(gray)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, edge_map)
        return True, ""
    except Exception as exc:
        _logger.warning("save_logo_ref failed: %s", exc, exc_info=True)
        return False, str(exc)


def load_ref_patch(ref_path: str | None, template_id: int | None = None) -> np.ndarray | None:
    """Load reference patch PNG from disk. Returns None if file missing."""
    if ref_path:
        p = Path(ref_path)
        if p.is_file():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

            return img if img is not None else None
    # Fallback to standard path
    if template_id is not None:
        p = get_ref_path(template_id)
        if p.is_file():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            return img if img is not None else None
    return None


def save_ref_patch(frame_bgr: np.ndarray, roi: dict, save_path: str,
                   hsv_lower: np.ndarray = DEFAULT_HSV_LOWER,
                   hsv_upper: np.ndarray = DEFAULT_HSV_UPPER,
                   padding_px: int = 20,
                   rotation: float = 0.0) -> tuple[bool, str]:
    """Crop ROI, apply Canny edge detection, save as grayscale PNG.

    Returns (True, "") on success, (False, reason) on failure.
    """
    try:
        rx, ry = int(roi.get("x", 0)), int(roi.get("y", 0))
        rw, rh = int(roi.get("w", 0)), int(roi.get("h", 0))
        if rw <= 0 or rh <= 0:
            return False, f"ROI dimensions invalid: w={rw} h={rh} (must be > 0)"
        fh, fw = frame_bgr.shape[:2]
        rx = max(0, min(rx, fw - 1))
        ry = max(0, min(ry, fh - 1))
        rw = min(rw, fw - rx)
        rh = min(rh, fh - ry)
        roi_frame = frame_bgr[ry:ry+rh, rx:rx+rw]
        if roi_frame.size == 0:
            return False, f"ROI region empty after clipping (frame {fw}x{fh}, roi x={rx} y={ry} w={rw} h={rh})"
        if abs(rotation) > 0.1:
            center = (rw / 2, rh / 2)
            M = cv2.getRotationMatrix2D(center, -rotation, 1.0)
            roi_frame = cv2.warpAffine(
                roi_frame, M, (rw, rh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        edge_map = _auto_canny(gray)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, edge_map)
        return True, ""
    except Exception as exc:
        _logger.warning("save_ref_patch failed: %s", exc, exc_info=True)
        return False, str(exc)


def match_gap(frame_bgr: np.ndarray, roi: dict, ref_patch: np.ndarray,
              threshold: float = 0.85) -> dict[str, Any]:
    """Run cv2.matchTemplate of ref_patch on the ROI region.

    Returns:
        {"match": bool, "score": float, "location": (x, y)} — all native Python types
    """
    try:
        rx, ry = int(roi.get("x", 0)), int(roi.get("y", 0))
        rw, rh = int(roi.get("w", 0)), int(roi.get("h", 0))
        if rw <= 0 or rh <= 0 or ref_patch is None or ref_patch.size == 0:
            return {"match": False, "score": 0.0, "location": (0, 0)}

        fh, fw = frame_bgr.shape[:2]
        rx = max(0, min(rx, fw - 1))
        ry = max(0, min(ry, fh - 1))
        rw = min(rw, fw - rx)
        rh = min(rh, fh - ry)
        roi_frame = frame_bgr[ry:ry+rh, rx:rx+rw]
        if roi_frame.size == 0:
            return {"match": False, "score": 0.0, "location": (0, 0)}

        # Konversi ke edge map (auto-tune Canny) untuk matching yang robust
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        roi_frame = _auto_canny(gray)

        # Ref patch must fit inside ROI
        ph, pw = ref_patch.shape[:2]
        if ph > rh or pw > rw:
            # Resize ref patch to fit
            scale = min(rh / max(ph, 1), rw / max(pw, 1))
            new_w = max(1, int(pw * scale))
            new_h = max(1, int(ph * scale))
            ref_patch = cv2.resize(ref_patch, (new_w, new_h))
            ph, pw = ref_patch.shape[:2]

        # Template matching
        result = cv2.matchTemplate(roi_frame, ref_patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        return {
            "match": bool(max_val >= threshold),
            "score": float(round(max_val, 4)),
            "location": (int(max_loc[0]), int(max_loc[1])),
        }
    except Exception:
        _logger.warning("[gap] match_gap error", exc_info=True)
        return {"match": False, "score": 0.0, "location": (0, 0)}
