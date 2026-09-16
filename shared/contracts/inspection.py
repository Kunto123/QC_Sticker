from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from shared.contracts.enums import DecisionCode, SessionStatus


@dataclass(slots=True)
class TemplateDeployment:
    id: int
    template_id: int
    template_version_id: int
    is_active: bool
    deployed_by: int | None
    effective_from: str | None
    effective_until: str | None
    created_at: str | None
    template_name: str
    version_number: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InspectionSession:
    session_id: str
    client_id: str
    camera_index: int
    template_version_id: int
    status: SessionStatus
    line_id: str | None = None
    station_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(slots=True)
class InspectionResult:
    id: int | None
    template_version_id: int | None
    line_id: str | None
    station_id: str | None
    part_name: str | None
    mp_check: str | None
    data1: float | None
    data2: float | None
    decision: DecisionCode
    decision_code: str
    reject_reason_code: str | None
    push_status: str
    retry_count: int
    operator_user_id: int | None
    inspected_at: str | None
    part_ready_status: str | None = None
    part_ready_match_ratio: float | None = None
    part_ready_distance: float | None = None
    detected_class: str | None = None
    expected_class: str | None = None
    sticker_confidence: float | None = None
    sticker_bbox: dict[str, Any] | None = None
    sticker_backend: str | None = None
    sticker_tilt_angle: float | None = None
    sticker_tilt_expected: float | None = None
    sticker_tilt_deviation: float | None = None
    sticker_tilt_threshold: float | None = None
    ocr_text: str | None = None
    ocr_confidence: float | None = None
    ocr_engine: str | None = None
    ocr_status: str | None = None
    text_bbox: dict[str, Any] | None = None
    dot_bbox: dict[str, Any] | None = None
    dot_position: dict[str, Any] | None = None
    anchor_offset: dict[str, Any] | None = None
    pose_angle: float | None = None
    validation_details: dict[str, Any] | None = None
    part_ready_roi_meta: dict[str, Any] | None = None
    sticker_roi_meta: dict[str, Any] | None = None
    targets: list[dict[str, Any]] = field(default_factory=list)
    actuation_status: str | None = None
    actuation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision"] = self.decision.value
        return payload
