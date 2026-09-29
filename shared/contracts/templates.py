from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CameraDefaults:
    camera_index: int = 0
    width: int | None = None
    height: int | None = None
    fps: float | None = None


@dataclass(slots=True)
class RoiGeometry:
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0


@dataclass(slots=True)
class VisionConfig:
    model_path: str = "models/dummy.pt"
    model_meta_path: str | None = None
    runtime: str = "ultralytics"
    conf_threshold: float = 0.25
    inference_fps: float = 4.0
    imgsz: int = 640
    classes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PartReadyConfig:
    enabled: bool = True
    # "gap_template_match" (pencocokan template edge-map terhadap referensi
    # yang sudah diambil) atau "mean_std_threshold" (gate mean/std grayscale).
    # Selain itu dianggap gagal (fail closed).
    method: str = "gap_template_match"
    gap_match_threshold: float = 0.85
    gap_ref_path: str | None = None
    # Threshold edge Canny untuk gap_template_match. None (salah satu) = auto-tuned
    # dari median brightness ROI itu sendiri (perilaku lama, default tidak berubah).
    canny_low: int | None = None
    canny_high: int | None = None
    # Khusus gap_template_match: seberapa jauh (fraksi dari w/h ROI itu
    # sendiri, tiap sisi) area pencarian saat runtime diperbesar melebihi
    # part_ready_roi, supaya cv2.matchTemplate punya ruang untuk menemukan
    # part yang bergeser, bukan cuma dibandingkan pada satu offset tetap.
    # 0.0 = perilaku lama (area pencarian == ROI == ukuran patch referensi,
    # tanpa toleransi pergeseran).
    gap_search_margin: float = 0.0
    min_match_ratio: float = 0.5
    stable_ms: int = 500
    release_ms: int = 300
    ema_alpha: float = 0.3
    mean_max: float = 105.0
    std_max: float = 35.0


@dataclass(slots=True)
class StickerRule:
    part_name: str
    expected_class: str
    enabled: bool = True
    min_roi_confidence: float = 0.0
    min_class_confidence: float | None = None
    max_offset_x: float | None = None
    max_offset_y: float | None = None
    expected_center_x: float | None = None
    expected_center_y: float | None = None
    part_ready_settle_ms: int | None = None


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
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def roi(self) -> RoiGeometry:
        return self.sticker_roi

    def to_dict(self) -> dict[str, Any]:
        return {
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
            "metadata": dict(self.metadata),
        }


_ROI_ALLOWED = {"x", "y", "w", "h"}


def _pick_roi_payload(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


# Key yang tidak dikenal (data DB lama: field OCR/tilt/colour-profile/counter,
# `mode`, `criteria`, `component_rois`, ...) langsung dibuang diam-diam saat parsing.
_VALID_PART_READY_FIELDS = set(PartReadyConfig.__slots__)
_VALID_STICKER_FIELDS = set(StickerRule.__slots__)
_VALID_VISION_FIELDS = set(VisionConfig.__slots__)
_VALID_CAMERA_FIELDS = set(CameraDefaults.__slots__)


def template_from_dict(payload: dict[str, Any]) -> InspectionTemplate:
    """Parse dict template, mentolerir key dari layout template versi lama."""
    part_ready_roi_payload = _pick_roi_payload(payload, "part_ready_roi", "roi", "sticker_roi")
    sticker_roi_payload = _pick_roi_payload(payload, "sticker_roi", "roi", "part_ready_roi")
    part_ready_roi_payload = {k: v for k, v in part_ready_roi_payload.items() if k in _ROI_ALLOWED}
    sticker_roi_payload = {k: v for k, v in sticker_roi_payload.items() if k in _ROI_ALLOWED}
    _sticker_filtered = {k: v for k, v in dict(payload.get("sticker") or {}).items() if k in _VALID_STICKER_FIELDS}
    _vision_filtered = {k: v for k, v in dict(payload.get("vision") or {}).items() if k in _VALID_VISION_FIELDS}
    _camera_filtered = {k: v for k, v in dict(payload.get("camera") or {}).items() if k in _VALID_CAMERA_FIELDS}
    _part_ready_filtered = {k: v for k, v in dict(payload.get("part_ready") or {}).items() if k in _VALID_PART_READY_FIELDS}
    _method_raw = _part_ready_filtered.get("method")
    if _method_raw is None or str(_method_raw).strip() == "":
        _part_ready_filtered["method"] = "gap_template_match"

    return InspectionTemplate(
        id=payload.get("id"),
        version_id=payload.get("version_id"),
        version_number=int(payload.get("version_number") or 1),
        name=str(payload.get("name") or "").strip(),
        description=str(payload.get("description") or "").strip(),
        is_active=bool(payload.get("is_active", True)),
        camera=CameraDefaults(**_camera_filtered),
        part_ready_roi=RoiGeometry(**part_ready_roi_payload),
        sticker_roi=RoiGeometry(**sticker_roi_payload),
        vision=VisionConfig(**_vision_filtered),
        part_ready=PartReadyConfig(**_part_ready_filtered),
        sticker=StickerRule(**_sticker_filtered),
        persistence=PersistenceConfig(**(payload.get("persistence") or {})),
        metadata=dict(payload.get("metadata") or {}),
    )


def validate_sticker_rule(sticker: dict[str, Any]) -> list[str]:
    """Validasi bagian `sticker` dari payload template. List kosong berarti valid."""
    errors: list[str] = []
    if not str(sticker.get("expected_class") or "").strip():
        errors.append("sticker: expected_class wajib diisi")
    if sticker.get("min_roi_confidence") is not None:
        try:
            v = float(sticker["min_roi_confidence"])
            if v < 0 or v > 1:
                errors.append(f"sticker: min_roi_confidence {v} di luar rentang [0,1]")
        except (TypeError, ValueError):
            errors.append("sticker: min_roi_confidence harus berupa angka desimal")
    return errors
