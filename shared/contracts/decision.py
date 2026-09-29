from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Decision:
    """Hasil evaluasi yang seragam — SATU-SATUNYA tipe output dari ModeEvaluator manapun.

    Semua sistem hilir (PLC, DB logging, tampilan ACC/NG) membaca struct ini.
    Mereka TIDAK PERNAH mengakses output model mentah (detections, heatmap, dll).

    Payload ``details`` per mode:

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
        thresholds: dict
        backend: str | None
        model_path: str | None
        meta_path: str | None

    **counter**:
        mode: "counter"
        rois: list[dict]  # tiap elemen punya: name, ok, classes{name, detected, min, max, ok}, total_detected, foreign_classes
        consecutive_ok: int
        consecutive_needed: int

    **defect**:
        mode: "defect"
        rois: list[dict]  # tiap elemen punya: name, ok, anomaly_score, threshold, heatmap_ref (opsional)
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
