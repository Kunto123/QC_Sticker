"""label_geometry.py — transform bounding-box annotations to match augmented images.

This module is the label geometry engine for Phase 4 of the geometric augmentation
rollout.  It takes a list of annotation labels (dicts with ``bbox`` or ``points``)
and a transform trace produced by the augment worker, and returns new labels whose
coordinates have been transformed to match the augmented image.

Supported transforms (GA)
--------------------------
flip_h  — horizontal mirror: x' = (W - x - w),  y' = y,  w' = w,  h' = h
flip_v  — vertical mirror:   x' = x,  y' = (H - y - h),  w' = w,  h' = h
rotate  — rotation by angle_deg degrees around (cx, cy) using warpAffine conventions

Coordinate conventions
-----------------------
All internal computations use **pixel (absolute) coordinates** stored in the label
dict as ``{"x": left, "y": top, "w": width, "h": height}`` (top-left origin).
If a label carries ``"normalized": True`` the stored values are fractions [0, 1]
and are scaled to pixels before transformation, then scaled back afterwards.

The engine handles both ``bbox`` dict sub-key format and flat x/y/w/h keys at the
top level of a label dict.  Polygon (``points``) labels are also supported via the
same per-vertex transformation pipeline.
"""
from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Low-level geometry helpers
# ---------------------------------------------------------------------------

def _rotate_point(px: float, py: float, cx: float, cy: float, cos_a: float, sin_a: float) -> tuple[float, float]:
    """Rotate point (px, py) around centre (cx, cy) using pre-computed cos/sin.

    OpenCV's ``warpAffine`` with ``getRotationMatrix2D((cx,cy), angle, 1)`` applies:
        x' = cos(angle)*(x-cx) - sin(angle)*(y-cy) + cx
        y' = sin(angle)*(x-cx) + cos(angle)*(y-cy) + cy
    where angle is in degrees with CLOCKWISE positive (OpenCV convention).
    Note: math.cos/sin work in radians with CCW positive, so we negate the angle.
    """
    dx = px - cx
    dy = py - cy
    return (cos_a * dx - sin_a * dy + cx,
            sin_a * dx + cos_a * dy + cy)


def _transform_bbox(
    x: float, y: float, w: float, h: float,
    transform_name: str,
    params: dict,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """Apply a single transform to a top-left-origin bbox and return the new bbox.

    Returns ``(x', y', w', h')`` clipped to the image boundary.
    """
    if transform_name == "flip_h":
        x2 = img_w - x - w
        return (max(0.0, x2), y, w, h)

    if transform_name == "flip_v":
        y2 = img_h - y - h
        return (x, max(0.0, y2), w, h)

    if transform_name == "rotate":
        angle_deg = float(params.get("angle_deg", 0.0))
        cx = float(params.get("cx", img_w / 2))
        cy = float(params.get("cy", img_h / 2))
        # OpenCV rotates CW for positive angle; negate for standard math rotation.
        rad = math.radians(-angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        # Transform all 4 corners of the bbox.
        corners = [
            (x,     y),
            (x + w, y),
            (x + w, y + h),
            (x,     y + h),
        ]
        transformed = [_rotate_point(px, py, cx, cy, cos_a, sin_a) for px, py in corners]
        xs = [p[0] for p in transformed]
        ys = [p[1] for p in transformed]
        nx = min(xs)
        ny = min(ys)
        nw = max(xs) - nx
        nh = max(ys) - ny
        # Clip to image bounds.
        nx = max(0.0, min(float(img_w), nx))
        ny = max(0.0, min(float(img_h), ny))
        nw = max(0.0, min(float(img_w) - nx, nw))
        nh = max(0.0, min(float(img_h) - ny, nh))
        return (nx, ny, nw, nh)

    # Photometric or unknown — coordinates unchanged.
    return (x, y, w, h)


def _transform_point(
    px: float, py: float,
    transform_name: str,
    params: dict,
    img_w: int,
    img_h: int,
) -> tuple[float, float]:
    """Apply a single transform to a single 2-D point."""
    if transform_name == "flip_h":
        return (img_w - px, py)
    if transform_name == "flip_v":
        return (px, img_h - py)
    if transform_name == "rotate":
        angle_deg = float(params.get("angle_deg", 0.0))
        cx = float(params.get("cx", img_w / 2))
        cy = float(params.get("cy", img_h / 2))
        rad = math.radians(-angle_deg)
        return _rotate_point(px, py, cx, cy, math.cos(rad), math.sin(rad))
    return (px, py)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transform_labels(
    labels: list[dict],
    trace_transforms: list[dict],
    img_w: int,
    img_h: int,
) -> list[dict]:
    """Return a new label list with coordinates transformed according to *trace_transforms*.

    Parameters
    ----------
    labels:
        List of annotation label dicts (``bbox`` or ``points`` style).
    trace_transforms:
        The ``"transforms"`` list from a ``.trace.json`` sidecar produced by
        ``augment_worker._apply_transforms_traced``.
    img_w, img_h:
        Pixel dimensions of the **source** image (before augmentation).

    Returns a deep copy of *labels* with updated coordinates.
    """
    import copy
    result = copy.deepcopy(labels)
    for label in result:
        _transform_label_inplace(label, trace_transforms, img_w, img_h)
    return result


def _transform_label_inplace(
    label: dict,
    trace_transforms: list[dict],
    img_w: int,
    img_h: int,
) -> None:
    """Mutate *label* in-place, applying each transform in *trace_transforms* sequentially."""
    if not isinstance(label, dict):
        return

    normalized = bool(label.get("normalized"))

    # --- bbox style ---
    bbox_source: dict | None = None
    if isinstance(label.get("bbox"), dict):
        bbox_source = label["bbox"]
    elif all(k in label for k in ("x", "y", "w", "h")):
        bbox_source = label  # flat format

    if bbox_source is not None:
        try:
            x = float(bbox_source.get("x"))
            y = float(bbox_source.get("y"))
            w = float(bbox_source.get("w"))
            h = float(bbox_source.get("h"))
        except (TypeError, ValueError):
            return

        # Scale to pixels if stored normalized.
        if normalized:
            x *= img_w; y *= img_h; w *= img_w; h *= img_h

        for step in trace_transforms:
            name = str(step.get("name") or "")
            params = step.get("params") or {}
            x, y, w, h = _transform_bbox(x, y, w, h, name, params, img_w, img_h)

        if normalized:
            x /= img_w; y /= img_h; w /= img_w; h /= img_h

        bbox_source["x"] = x
        bbox_source["y"] = y
        bbox_source["w"] = w
        bbox_source["h"] = h
        return

    # --- polygon / points style ---
    raw_points = label.get("points")
    if not isinstance(raw_points, list):
        return

    new_points = []
    for pt in raw_points:
        if not isinstance(pt, dict):
            new_points.append(pt)
            continue
        try:
            px = float(pt.get("x"))
            py = float(pt.get("y"))
        except (TypeError, ValueError):
            new_points.append(pt)
            continue

        if normalized:
            px *= img_w; py *= img_h

        for step in trace_transforms:
            name = str(step.get("name") or "")
            params = step.get("params") or {}
            px, py = _transform_point(px, py, name, params, img_w, img_h)

        if normalized:
            px /= img_w; py /= img_h

        new_points.append({**pt, "x": px, "y": py})

    label["points"] = new_points
