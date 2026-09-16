from __future__ import annotations

import io
import base64
import logging
from typing import Any

import cv2
import numpy as np

from shared.contracts.decision import Decision
from backend.app.services.evaluators.base import ModeEvaluator, EvalContext
from backend.app.services.anomaly_backend import get_scorer

logger = logging.getLogger(__name__)


def _aggregate_score(heatmap_slice: np.ndarray, method: str = "p99") -> float | None:
    """Aggregate anomaly scores from a heatmap slice into a single score.

    Args:
        heatmap_slice: 2D numpy array of anomaly scores.
        method: One of "p99", "max", "topk_mean" (mean of top 10 pixels).

    Returns:
        Aggregated score as float, or None if the slice is empty (JSON-safe).
    """
    if heatmap_slice.size == 0:
        return None
    flat = heatmap_slice.ravel()
    if method == "max":
        return float(flat.max())
    elif method == "topk_mean":
        k = min(10, flat.size)
        topk = np.partition(flat, -k)[-k:]
        return float(topk.mean())
    else:  # p99 (default)
        return float(np.percentile(flat, 99))


def _parse_geometry(geom: dict, fw: int, fh: int) -> tuple[int, int, int, int]:
    """Convert fractional geometry to pixel coordinates. Returns (x, y, w, h)."""
    x = int(float(geom.get("x", 0.0)) * fw)
    y = int(float(geom.get("y", 0.0)) * fh)
    w = max(1, int(float(geom.get("w", 1.0)) * fw))
    h = max(1, int(float(geom.get("h", 1.0)) * fh))
    x = max(0, min(fw - 1, x))
    y = max(0, min(fh - 1, y))
    w = min(fw - x, w)
    h = min(fh - y, h)
    return x, y, w, h


def _build_union_bbox(rois: list[dict], fw: int, fh: int, padding: float = 0.05) -> tuple[int, int, int, int]:
    """Compute union bounding box of all ROIs with padding, clipped to frame."""
    xs, ys, xe, ye = fw, fh, 0, 0
    for roi in rois:
        geom = roi.get("geometry", {})
        x = int(float(geom.get("x", 0.0)) * fw)
        y = int(float(geom.get("y", 0.0)) * fh)
        w = max(1, int(float(geom.get("w", 1.0)) * fw))
        h = max(1, int(float(geom.get("h", 1.0)) * fh))
        xs = min(xs, x)
        ys = min(ys, y)
        xe = max(xe, x + w)
        ye = max(ye, y + h)
    # Add padding
    pad_x = int((xe - xs) * padding)
    pad_y = int((ye - ys) * padding)
    xs = max(0, xs - pad_x)
    ys = max(0, ys - pad_y)
    xe = min(fw, xe + pad_x)
    ye = min(fh, ye + pad_y)
    return xs, ys, xe - xs, ye - ys


def _encode_heatmap_jpg(heatmap: np.ndarray | None, quality: int = 70) -> str | None:
    """Encode heatmap as base64 JPEG. Returns None on failure."""
    if heatmap is None or heatmap.size == 0:
        return None
    try:
        hm_norm = heatmap.copy()
        hm_min, hm_max = float(hm_norm.min()), float(hm_norm.max())
        if hm_max > hm_min:
            hm_norm = (hm_norm - hm_min) / (hm_max - hm_min) * 255
        hm_uint8 = hm_norm.astype(np.uint8)
        hm_colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        ok_flag, hm_buf = cv2.imencode(".jpg", hm_colored, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok_flag:
            return base64.b64encode(hm_buf.tobytes()).decode("ascii")
    except Exception:
        pass
    return None


class DefectEvaluator(ModeEvaluator):
    """Evaluator for anomaly-based defect detection mode.

    Supports two inference strategies:
    - ``"whole_part"`` (default): Single inference on the union of all ROIs + padding.
      Heatmap is sliced per-ROI, scores aggregated via ``p99`` (configurable).
      Faster, gives spatial context to the anomaly model.
    - ``"per_roi_crop"``: Legacy mode — each ROI is cropped and scored independently.
      Required when a specific ROI has its own override model_path.

    ROI field ``model_path``: if set, that ROI ALWAYS uses per_roi_crop with its own model,
    even when ``inference_mode`` is ``"whole_part"``.

    Criteria keys:
    - ``rois``: list of dicts with ``{name, geometry, threshold, model_path}``
    - ``default_model_path``: fallback model for ROIs without override
    - ``inference_mode``: ``"whole_part"`` | ``"per_roi_crop"`` (default ``"whole_part"``)
    - ``aggregation``: ``"p99"`` | ``"max"`` | ``"topk_mean"`` (default ``"p99"``)
    """

    mode_name = "defect"

    def evaluate(self, ctx: EvalContext) -> Decision:
        rois = ctx.criteria.get("rois", [])
        if not rois:
            return Decision.rejected("NO_DEFECT_ROIS", {
                "mode": "defect",
                "error": "No defect ROIs defined",
            })

        frame = ctx.frame
        if frame is None or frame.size == 0:
            return Decision.rejected("NO_FRAME", {
                "mode": "defect",
                "error": "No frame provided",
            })

        fh, fw = frame.shape[:2]
        inference_mode = str(ctx.criteria.get("inference_mode", "whole_part")).strip().lower()
        aggregation = str(ctx.criteria.get("aggregation", "p99")).strip().lower()
        default_model = ctx.criteria.get("default_model_path")

        # Identify which ROIs use override models (must use per_roi_crop)
        override_roi_indices = {
            i for i, r in enumerate(rois)
            if r.get("model_path")
        }

        # Phase 1: whole_part inference (if applicable)
        whole_heatmap = None
        whole_offset = (0, 0)  # (ox, oy) offset of union region in frame coords
        if inference_mode == "whole_part" and not override_roi_indices:
            # Single inference on union of all ROIs + padding
            ux, uy, uw, uh = _build_union_bbox(rois, fw, fh)
            if uw > 0 and uh > 0:
                region = frame[uy:uy+uh, ux:ux+uw]
                scorer = get_scorer(default_model)
                try:
                    _, whole_heatmap = scorer.score(region)
                    whole_offset = (ux, uy)
                    logger.info(
                        "[defect] whole_part inference on %dx%d region (offset %d,%d), "
                        "inference_mode=%s aggregation=%s",
                        uw, uh, ux, uy, inference_mode, aggregation,
                    )
                except Exception as exc:
                    logger.error("[defect] whole_part inference failed: %s", exc)
                    # Fall through to per-ROI mode

        # Phase 2: evaluate each ROI
        all_ok = True
        roi_results = []
        reject_reason = None

        for i, roi in enumerate(rois):
            name = roi.get("name", f"ROI {i+1}")
            geom = roi.get("geometry", {})
            x, y, w, h = _parse_geometry(geom, fw, fh)

            if w <= 0 or h <= 0:
                roi_results.append({
                    "name": name, "ok": False,
                    "score": None, "threshold": roi.get("threshold", 0.5),
                    "error": "empty crop", "roi": geom,
                })
                all_ok = False
                if not reject_reason:
                    reject_reason = "ANOMALY_DETECTED"
                continue

            # Determine if this ROI uses per_roi_crop (override_model or whole_part failed)
            use_per_roi = (
                inference_mode == "per_roi_crop"
                or i in override_roi_indices
                or whole_heatmap is None
            )

            if use_per_roi:
                # Per-ROI crop scoring
                crop = frame[y:y+h, x:x+w]
                model_path = roi.get("model_path") or default_model
                scorer = get_scorer(model_path)
                try:
                    score, heatmap = scorer.score(crop)
                except Exception as exc:
                    logger.error("[defect] ROI %s per-roi score failed: %s", name, exc)
                    roi_results.append({
                        "name": name, "ok": False,
                        "score": 0.0, "threshold": roi.get("threshold", 0.5),
                        "error": str(exc), "roi": geom,
                    })
                    all_ok = False
                    if not reject_reason:
                        reject_reason = "ANOMALY_DETECTED"
                    continue
                heatmap_b64 = _encode_heatmap_jpg(heatmap)
            else:
                # Slice from whole heatmap
                # Map ROI coords (relative to frame) to union region coords
                ox, oy = whole_offset
                hs_y = max(0, y - oy)
                hs_x = max(0, x - ox)
                hs_h = min(h, whole_heatmap.shape[0] - hs_y) if whole_heatmap is not None else 0
                hs_w = min(w, whole_heatmap.shape[1] - hs_x) if whole_heatmap is not None else 0

                if whole_heatmap is None or hs_h <= 0 or hs_w <= 0:
                    score_val = None
                    heatmap_slice = None
                else:
                    heatmap_slice = whole_heatmap[hs_y:hs_y+hs_h, hs_x:hs_x+hs_w]
                    score_val = _aggregate_score(heatmap_slice, aggregation)

                heatmap_b64 = _encode_heatmap_jpg(heatmap_slice)
                score = score_val

            threshold = float(roi.get("threshold", 0.5))
            ok = (score is not None and score <= threshold) if score is not None else False

            roi_results.append({
                "name": name,
                "ok": ok,
                "score": round(float(score), 4) if score is not None else None,
                "threshold": threshold,
                "heatmap_b64": heatmap_b64,
                "roi": geom,
            })

            if not ok:
                all_ok = False
                if not reject_reason:
                    reject_reason = "ANOMALY_DETECTED"

        details = {
            "mode": "defect",
            "rois": roi_results,
        }

        if all_ok:
            return Decision.accepted(details)
        return Decision.rejected(reject_reason or "ANOMALY_DETECTED", details)
