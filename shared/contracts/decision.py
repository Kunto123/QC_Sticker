from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Decision:
    """Uniform evaluation result — the ONLY output type from any ModeEvaluator.

    All downstream systems (PLC, logging DB, ACC/NG display) read this struct.
    They NEVER access raw model output (detections, heatmaps, etc.).

    ``details`` payload per mode:

    **sticker**:
        mode: "sticker"
        status: str (e.g. "pass", "disabled", "inferring")
        candidate_source: str | None
        selected_candidate: dict | None
        candidate_count: int
        matching_candidate_count: int
        expected_class: str
        detected_class: str | None
        confidence: float | None
        bbox: dict | None
        tilt: dict  (angle, expected, deviation, threshold)
        thresholds: dict
        backend: str | None
        model_path: str | None
        meta_path: str | None

    **counter**:
        mode: "counter"
        rois: list[dict]  # each has: name, ok, classes{name, detected, min, max, ok}, total_detected, foreign_classes
        consecutive_ok: int
        consecutive_needed: int

    **defect**:
        mode: "defect"
        rois: list[dict]  # each has: name, ok, anomaly_score, threshold, heatmap_ref (optional)
    """
    accept: bool
    reason_code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def accepted(details: dict[str, Any] | None = None) -> "Decision":
        return Decision(accept=True, reason_code=None, details=details or {})

    @staticmethod
    def rejected(reason_code: str, details: dict[str, Any] | None = None) -> "Decision":
        return Decision(accept=False, reason_code=reason_code, details=details or {})
