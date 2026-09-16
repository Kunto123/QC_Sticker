from __future__ import annotations

from dataclasses import asdict, dataclass, field
# FIX_VERSION_MARKER: 2026-06-24-v3 — this line confirms the fix is loaded. If you see this in error, new code is running.
from typing import Any


@dataclass(slots=True)
class CameraDefaults:
    camera_index: int = 0
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    rotation_degrees: float = 0.0


@dataclass(slots=True)
class RoiGeometry:
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0
    rotation: float = 0.0
    width: int | None = None


@dataclass(slots=True)
class VisionConfig:
    model_path: str = "models/dummy.pt"
    model_meta_path: str | None = None
    runtime: str = "ultralytics"
    conf_threshold: float = 0.25
    stream_fps: float = 10.0
    inference_fps: float = 4.0
    imgsz: int = 640
    classes: list[str] = field(default_factory=list)
    enable_ergonomic_check: bool = False
    ergonomic_pose_model_path: str | None = None
    ergonomic_min_keypoint_conf: float = 0.35
    text_anchor_class: str = "text_anchor"
    center_dot_class: str = "center_dot"
    anchor_crop_padding_ratio: float = 0.08
    anchor_crop_scale: float = 2.0


@dataclass(slots=True)
class PartReadyConfig:
    enabled: bool = True
    method: str = "gap_template_match"
    gap_match_threshold: float = 0.85
    gap_ref_path: str | None = None
    gap_ref_type: str = "raw"
    gap_hsv_lower: list[int] = field(default_factory=lambda: [90, 50, 50])
    gap_hsv_upper: list[int] = field(default_factory=lambda: [130, 255, 255])
    gap_padding_px: int = 20
    color_profile_id: int | None = None
    colorspace: str = "LAB"
    distance_threshold: float | None = None
    min_match_ratio: float = 0.5
    hsv_lower: list[int] = field(default_factory=lambda: [0, 0, 0])
    hsv_upper: list[int] = field(default_factory=lambda: [180, 255, 80])
    stable_ms: int = 500
    release_ms: int = 300
    ema_alpha: float = 0.3
    hsv_adaptive: bool = False
    hsv_adaptive_alpha: float = 0.1
    hsv_adaptive_min_ratio: float = 0.85
    mean_max: float = 105.0
    std_max: float = 35.0
    # Calibration raw data — captured from 3 conditions, used to auto-compute mean_max/std_max
    calibration_empty_mean: float = 0.0   # mean when ROI is empty (no part)
    calibration_part_mean: float = 0.0     # mean when black part is present
    calibration_part_std: float = 0.0      # std when black part is present
    calibration_sticker_std: float = 0.0   # std when sticker is on part
    logo_ref_path: str | None = None  # deprecated, ignored — kept for backward compat with old DB data


@dataclass(slots=True)
class ComponentClassTarget:
    class_name: str
    count: int  # kept for backward compat — used to derive min/max if not set
    min_count: int | None = None
    max_count: int | None = None

    def __post_init__(self) -> None:
        """Backward compat: derive min/max from count ONLY for legacy data."""
        if self.min_count is None and self.max_count is None:
            # Legacy data: only has `count` → exact match
            self.min_count = self.count
            self.max_count = self.count
        elif self.min_count is None:
            self.min_count = 0
        # max_count stays None = unlimited — DO NOT derive from count


@dataclass(slots=True)
class ComponentRoiRule:
    name: str
    roi: RoiGeometry
    classes: list[ComponentClassTarget]
    strict_foreign_class: bool = False
    model_override: str | None = None


@dataclass(slots=True)
class StickerRule:
    part_name: str
    expected_class: str
    enabled: bool = True
    validator_mode: str = "ml_detection"
    min_roi_confidence: float = 0.0
    min_class_confidence: float | None = None
    max_offset_x: float | None = None
    max_offset_y: float | None = None
    expected_center_x: float | None = None
    expected_center_y: float | None = None
    expected_tilt_degrees: float = 0.0
    max_tilt_degrees: float | None = None
    tilt_gate_enabled: bool = False
    edge_roi_tolerance_px: int = 10
    edge_search_padding_ratio: float = 0.10
    morph_kernel_width: int = 40
    morph_kernel_height: int = 5
    min_text_aspect_ratio: float = 3.0
    commit_stable_frames: int = 1
    part_ready_settle_ms: int | None = None
    part_ready_settle_frames: int = 3
    white_hsv_lower: list[int] = field(default_factory=lambda: [0, 0, 160])
    white_hsv_upper: list[int] = field(default_factory=lambda: [180, 70, 255])
    min_text_contour_area_ratio: float = 0.002


@dataclass(slots=True)
class PersistenceConfig:
    write_to_db: bool = True


@dataclass(slots=True)
class InspectionTemplate:
    id: int | None
    version_id: int | None
    version_number: int
    name: str
    description: str
    is_active: bool
    camera: CameraDefaults
    part_ready_roi: RoiGeometry
    sticker_roi: RoiGeometry
    vision: VisionConfig
    part_ready: PartReadyConfig
    sticker: StickerRule
    persistence: PersistenceConfig
    component_rois: list[ComponentRoiRule] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # ── New multi-mode fields (FASE 2) ──
    mode: str = "sticker"  # "sticker" | "counter" | "defect"
    criteria: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Sync mode + criteria from legacy fields for backward compat."""
        # Derive mode from sticker.validator_mode if criteria is empty
        if not self.criteria:
            _vm = str(getattr(self.sticker, "validator_mode", "ml_detection")).strip().lower()
            if _vm == "component_count":
                self.mode = "counter"
            elif _vm in ("defect",):
                self.mode = "defect"
            else:
                self.mode = "sticker"
            # Populate criteria from legacy fields
            if self.mode == "sticker":
                self.criteria = {
                    "expected_class": self.sticker.expected_class,
                    "enabled": self.sticker.enabled,
                    "min_roi_confidence": self.sticker.min_roi_confidence,
                    "min_class_confidence": self.sticker.min_class_confidence,
                    "max_offset_x": self.sticker.max_offset_x,
                    "max_offset_y": self.sticker.max_offset_y,
                    "tilt_gate_enabled": self.sticker.tilt_gate_enabled,
                    "max_tilt_degrees": self.sticker.max_tilt_degrees,
                    "expected_tilt_degrees": self.sticker.expected_tilt_degrees,
                }
            elif self.mode == "counter":
                self.criteria = {
                    "component_rois": [
                        {
                            "name": cr.name,
                            "roi": asdict(cr.roi),
                            "classes": [
                                {
                                    "class_name": ct.class_name,
                                    "count": ct.count,
                                    "min_count": ct.min_count,
                                    "max_count": ct.max_count,
                                }
                                for ct in cr.classes
                            ],
                            "strict_foreign_class": cr.strict_foreign_class,
                        }
                        for cr in self.component_rois
                    ],
                }

    @property
    def roi(self) -> RoiGeometry:
        return self.sticker_roi

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict — writes both legacy format AND new format for forward compat."""
        result = {
            "id": self.id,
            "version_id": self.version_id,
            "version_number": self.version_number,
            "name": self.name,
            "description": self.description,
            "is_active": self.is_active,
            "camera": asdict(self.camera),
            "part_ready_roi": asdict(self.part_ready_roi),
            "sticker_roi": asdict(self.sticker_roi),
            "vision": asdict(self.vision),
            "part_ready": asdict(self.part_ready),
            "sticker": asdict(self.sticker),
            "persistence": asdict(self.persistence),
            "component_rois": [
                {
                    "name": cr.name,
                    "roi": asdict(cr.roi),
                    "classes": [asdict(c) for c in cr.classes],
                    "strict_foreign_class": cr.strict_foreign_class,
                    "model_override": cr.model_override,
                }
                for cr in self.component_rois
            ],
            "metadata": dict(self.metadata),
        }
        # Add new multi-mode fields
        result["mode"] = self.mode
        result["criteria"] = dict(self.criteria) if self.criteria else {}
        return result


_ROI_ALLOWED = {"x", "y", "w", "h", "rotation", "width"}

_MODE_ALIASES = {
    "component_count": "counter", "count": "counter", "counter": "counter",
    "ml_detection": "sticker", "sticker": "sticker",
    "defect": "defect", "": "sticker",
}


def normalize_mode(raw: str | None) -> str:
    """Normalize legacy validator_mode strings to canonical mode names.

    Returns one of ``"sticker" | "counter" | "defect"``.
    Unknown values are returned as-is so caller's validator can reject with a clear message.
    """
    key = str(raw or "").strip().lower()
    return _MODE_ALIASES.get(key, key)


def _pick_roi_payload(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


_VALID_PART_READY_FIELDS = {
    "enabled", "method", "gap_match_threshold", "gap_ref_path", "logo_ref_path",
    "gap_ref_type", "gap_hsv_lower", "gap_hsv_upper", "gap_padding_px",
    "color_profile_id", "colorspace", "distance_threshold", "min_match_ratio",
    "hsv_lower", "hsv_upper", "stable_ms", "release_ms",
    "ema_alpha", "hsv_adaptive", "hsv_adaptive_alpha", "hsv_adaptive_min_ratio",
    "mean_max", "std_max",
    "calibration_empty_mean", "calibration_part_mean",
    "calibration_part_std", "calibration_sticker_std",
}

_VALID_STICKER_FIELDS = {
    "part_name", "expected_class", "enabled", "validator_mode", "min_roi_confidence",
    "min_class_confidence", "max_offset_x", "max_offset_y", "expected_center_x",
    "expected_center_y", "expected_tilt_degrees", "max_tilt_degrees",
    "tilt_gate_enabled", "edge_roi_tolerance_px",
    "edge_search_padding_ratio", "morph_kernel_width", "morph_kernel_height",
    "min_text_aspect_ratio", "commit_stable_frames", "part_ready_settle_ms",
    "part_ready_settle_frames", "white_hsv_lower", "white_hsv_upper",
    "min_text_contour_area_ratio",
}


def _parse_component_rois(payload: dict[str, Any]) -> list[ComponentRoiRule]:
    """Parse component_rois from template payload dict."""
    raw_rois = payload.get("component_rois") or []
    if not isinstance(raw_rois, list):
        return []
    result = []
    for raw_roi in raw_rois:
        if not isinstance(raw_roi, dict):
            continue
        roi_raw = raw_roi.get("roi") or {}
        roi_geom = RoiGeometry(**{k: v for k, v in roi_raw.items() if k in _ROI_ALLOWED})
        raw_classes = raw_roi.get("classes") or []
        classes = []
        for rc in raw_classes:
            if not isinstance(rc, dict):
                continue
            cn = str(rc.get("class_name") or "").strip()
            if not cn:
                continue
            # Read min/max if explicitly provided
            _min = rc.get("min_count")
            _max = rc.get("max_count")
            min_count = int(_min) if _min is not None else None
            max_count = int(_max) if _max is not None else None
            # count may be 0 when min/max are explicitly set — don't skip
            try:
                cnt = int(rc.get("count") or 0)
            except (TypeError, ValueError):
                cnt = 0
            if cnt <= 0 and min_count is None and max_count is None:
                continue
            if cnt <= 0:
                cnt = max(min_count or 1, 0)  # derive from min_count for compat
            classes.append(ComponentClassTarget(
                class_name=cn, count=cnt,
                min_count=min_count, max_count=max_count,
            ))
        if not classes:
            continue
        result.append(ComponentRoiRule(
            name=str(raw_roi.get("name") or "ROI").strip() or "ROI",
            roi=roi_geom,
            classes=classes,
            strict_foreign_class=bool(raw_roi.get("strict_foreign_class", False)),
            model_override=raw_roi.get("model_override"),
        ))
    return result


def template_from_dict(payload: dict[str, Any]) -> InspectionTemplate:
    """Parse template dict. Accepts both legacy flat format and new criteria format.

    Version: 2026-06-24-fix-v4-THIS-IS-THE-NEW-CODE
    """
    part_ready_roi_payload = _pick_roi_payload(payload, "part_ready_roi", "roi", "sticker_roi")
    sticker_roi_payload = _pick_roi_payload(payload, "sticker_roi", "roi", "part_ready_roi")
    # Strip unknown keys that RoiGeometry doesn't accept (e.g. 'height' from old DB data)
    part_ready_roi_payload = {k: v for k, v in part_ready_roi_payload.items() if k in _ROI_ALLOWED}
    sticker_roi_payload = {k: v for k, v in sticker_roi_payload.items() if k in _ROI_ALLOWED}
    _sticker_raw = dict(payload.get("sticker") or {})
    _vision_raw = payload.get("vision") or {}
    _vision_filtered = {k: v for k, v in _vision_raw.items() if k in VisionConfig.__slots__}
    _sticker_filtered = {k: v for k, v in _sticker_raw.items() if k in _VALID_STICKER_FIELDS}
    _part_ready_raw = payload.get("part_ready") or {}
    _part_ready_filtered = {k: v for k, v in _part_ready_raw.items() if k in _VALID_PART_READY_FIELDS}
    # Normalize empty method to default
    _method_raw = _part_ready_filtered.get("method")
    if _method_raw is None or str(_method_raw).strip() == "":
        _part_ready_filtered["method"] = "gap_template_match"

    # Read mode + criteria from new format if available
    _mode_raw = str(payload.get("mode") or "").strip().lower()
    _mode = normalize_mode(_mode_raw)
    _criteria = payload.get("criteria") or {}

    # Backward compat: if mode not explicitly set, derive from sticker.validator_mode
    if not _mode_raw or _mode == "sticker":
        _vm = str(_sticker_raw.get("validator_mode") or "ml_detection").strip().lower()
        _mode = normalize_mode(_vm)

    # Parse component_rois — try top-level first, fall back to criteria
    _component_rois = _parse_component_rois(payload)
    if not _component_rois and _mode == "counter":
        # New format: criteria is the source of truth
        _criteria_rois = _criteria.get("component_rois", [])
        if _criteria_rois:
            _component_rois = _parse_component_rois({"component_rois": _criteria_rois})
            # Also ensure criteria has it (normalize structure)
            _criteria = dict(_criteria)
            _criteria["component_rois"] = _criteria_rois

    return InspectionTemplate(
        id=payload.get("id"),
        version_id=payload.get("version_id"),
        version_number=int(payload.get("version_number") or 1),
        name=str(payload.get("name") or "").strip(),
        description=str(payload.get("description") or "").strip(),
        is_active=bool(payload.get("is_active", True)),
        camera=CameraDefaults(**(payload.get("camera") or {})),
        part_ready_roi=RoiGeometry(**part_ready_roi_payload),
        sticker_roi=RoiGeometry(**sticker_roi_payload),
        vision=VisionConfig(**_vision_filtered),
        part_ready=PartReadyConfig(**_part_ready_filtered),
        sticker=StickerRule(**_sticker_filtered),
        persistence=PersistenceConfig(**(payload.get("persistence") or {})),
        component_rois=_component_rois,
        metadata=dict(payload.get("metadata") or {}),
        mode=_mode,
        criteria=dict(_criteria) if _criteria else {},
    )


def validate_criteria(mode: str, criteria: dict[str, Any]) -> list[str]:
    """Validate mode-specific criteria and return list of error strings.

    Empty list means valid.
    """
    mode = normalize_mode(mode)
    errors: list[str] = []
    if mode == "sticker":
        if not criteria.get("expected_class"):
            errors.append("sticker: expected_class is required")
        if criteria.get("min_roi_confidence") is not None:
            try:
                v = float(criteria["min_roi_confidence"])
                if v < 0 or v > 1:
                    errors.append(f"sticker: min_roi_confidence {v} out of range [0,1]")
            except (TypeError, ValueError):
                errors.append("sticker: min_roi_confidence must be a float")
    elif mode == "counter":
        rois = criteria.get("component_rois", [])
        if not rois:
            errors.append("counter: at least one component_roi required")
        for i, roi in enumerate(rois):
            name = roi.get("name", f"ROI {i}")
            classes = roi.get("classes", [])
            if not classes:
                errors.append(f"counter: {name} has no classes")
            for j, ct in enumerate(classes):
                cn = ct.get("class_name", "").strip()
                if not cn:
                    errors.append(f"counter: {name} class[{j}] has no class_name")
                min_c = ct.get("min_count", ct.get("count", 1))
                max_c = ct.get("max_count")
                if max_c is not None and min_c > max_c:
                    errors.append(f"counter: {name} class '{cn}' min > max ({min_c} > {max_c})")
    elif mode == "defect":
        rois = criteria.get("rois", [])
        if not rois:
            errors.append("defect: at least one ROI required")
        _default_model = criteria.get("default_model_path", "")
        if _default_model and not isinstance(_default_model, str):
            errors.append("defect: default_model_path must be a string")
        for i, roi in enumerate(rois):
            name = roi.get("name", f"ROI {i}")
            if not roi.get("geometry"):
                errors.append(f"defect: {name} has no geometry")
            thresh = roi.get("threshold")
            if thresh is not None:
                try:
                    t = float(thresh)
                    if t < 0 or t > 1:
                        errors.append(f"defect: {name} threshold {t} out of range [0,1]")
                except (TypeError, ValueError):
                    errors.append(f"defect: {name} threshold must be a float")
            _roi_model = roi.get("model_path", "")
            if _roi_model and not isinstance(_roi_model, str):
                errors.append(f"defect: {name} model_path must be a string")
    else:
        errors.append(f"unknown mode: {mode!r}")
    return errors
