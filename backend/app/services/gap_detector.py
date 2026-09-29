"""Servis gap detection — part ready lewat template matching.

Mendeteksi clamp biru pakai segmentasi HSV, mengekstrak area gap sebagai
patch template dari gambar referensi, lalu pakai cv2.matchTemplate untuk
memastikan part ada di posisi yang benar di dalam ROI part_ready.
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
# Direktori penyimpanan referensi
PART_READY_REF_DIR = "backend/app/assets/part_ready_refs"

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def _auto_canny(
    gray: np.ndarray,
    sigma: float = 0.5,
    low: int | None = None,
    high: int | None = None,
) -> np.ndarray:
    """Peta edge Canny. Auto-tune threshold dari median gambar kecuali kedua
    `low`/`high` diberikan eksplisit (override manual per-template)."""
    gray = _clahe.apply(gray)   # normalkan brightness lokal sebelum hitung threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if low is not None and high is not None:
        return cv2.Canny(blurred, int(low), int(high))
    median = float(np.median(gray))
    auto_low = int(max(0, (1.0 - sigma) * median))
    auto_high = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(blurred, auto_low, auto_high)


def get_ref_path(template_id: int) -> Path:
    """Ambil path file untuk patch referensi milik sebuah template."""
    ref_dir = Path(project_root) / PART_READY_REF_DIR
    ref_dir.mkdir(parents=True, exist_ok=True)
    return ref_dir / f"{template_id}.png"


def load_ref_patch(ref_path: str | None, template_id: int | None = None) -> np.ndarray | None:
    """Muat PNG patch referensi dari disk. Return None kalau file tidak ada."""
    if ref_path:
        p = Path(ref_path)
        if p.is_file():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

            return img if img is not None else None
    # Fallback ke path standar
    if template_id is not None:
        p = get_ref_path(template_id)
        if p.is_file():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            return img if img is not None else None
    return None


def save_ref_patch(
    frame_bgr: np.ndarray,
    roi: dict,
    save_path: str,
    canny_low: int | None = None,
    canny_high: int | None = None,
) -> tuple[bool, str]:
    """Crop ROI, terapkan deteksi tepi Canny, simpan sebagai PNG grayscale.

    Return (True, "") kalau berhasil, (False, alasan) kalau gagal.
    """
    try:
        rx, ry = int(roi.get("x", 0)), int(roi.get("y", 0))
        rw, rh = int(roi.get("w", 0)), int(roi.get("h", 0))
        if rw <= 0 or rh <= 0:
            return False, f"Dimensi ROI tidak valid: w={rw} h={rh} (harus > 0)"
        fh, fw = frame_bgr.shape[:2]
        rx = max(0, min(rx, fw - 1))
        ry = max(0, min(ry, fh - 1))
        rw = min(rw, fw - rx)
        rh = min(rh, fh - ry)
        roi_frame = frame_bgr[ry:ry+rh, rx:rx+rw]
        if roi_frame.size == 0:
            return False, f"Region ROI kosong setelah clipping (frame {fw}x{fh}, roi x={rx} y={ry} w={rw} h={rh})"
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        edge_map = _auto_canny(gray, low=canny_low, high=canny_high)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, edge_map)
        return True, ""
    except Exception as exc:
        _logger.warning("save_ref_patch failed: %s", exc, exc_info=True)
        return False, str(exc)


def match_gap(frame_bgr: np.ndarray, roi: dict, ref_patch: np.ndarray,
              threshold: float = 0.85,
              canny_low: int | None = None,
              canny_high: int | None = None) -> dict[str, Any]:
    """Jalankan cv2.matchTemplate dari ref_patch pada region ROI.

    Returns:
        {"match": bool, "score": float, "location": (x, y)} — semua tipe Python native
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
        roi_frame = _auto_canny(gray, low=canny_low, high=canny_high)

        # Ref patch harus muat di dalam ROI
        ph, pw = ref_patch.shape[:2]
        if ph > rh or pw > rw:
            # Resize ref patch supaya muat
            scale = min(rh / max(ph, 1), rw / max(pw, 1))
            new_w = max(1, int(pw * scale))
            new_h = max(1, int(ph * scale))
            ref_patch = cv2.resize(ref_patch, (new_w, new_h))
            ph, pw = ref_patch.shape[:2]

        # Guard: TM_CCOEFF_NORMED membagi dengan variance milik masing-masing
        # patch. Referensi (blank) atau edge map live yang konstan membuat
        # variance itu persis nol -> 0/0 di OpenCV menghasilkan 1.0 palsu
        # "perfect match" TERLEPAS dari isi patch lainnya (terverifikasi:
        # referensi blank skor 1.0 melawan gambar live apa pun, blank atau
        # tidak). Referensi dengan nol edge terdeteksi berarti Canny tidak
        # pernah aktif saat kalibrasi dan tidak bisa memberi perbandingan yang
        # bermakna; capture live yang blank (mis. lensa tertutup) juga tidak
        # bisa mengonfirmasi apa pun — fail closed daripada percaya korelasi
        # yang degenerate.
        if ref_patch.min() == ref_patch.max() or roi_frame.min() == roi_frame.max():
            return {"match": False, "score": 0.0, "location": (0, 0)}

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
