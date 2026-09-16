from __future__ import annotations

from typing import Any

from shared.contracts.decision import Decision
from shared.contracts.enums import DecisionCode, RejectReasonCode
from backend.app.services.evaluators.base import ModeEvaluator, EvalContext


class StickerEvaluator(ModeEvaluator):
    """Evaluator for sticker QC mode (ml_detection / ml_roi_classification).

    Core decision logic extracted from InspectionSessionService._validate_sticker.
    Pure function of (frame, detections, criteria, state) → Decision.
    """

    mode_name = "sticker"

    def evaluate(self, ctx: EvalContext) -> Decision:
        criteria = ctx.criteria
        detections = ctx.detections
        roi_frame = ctx.roi_frame
        sticker = ctx.additional.get("sticker_rule")
        tilt_info = ctx.additional.get("tilt_info", {})
        candidates = ctx.additional.get("candidates", [])
        selected_candidate = ctx.additional.get("selected_candidate")
        candidate_source = ctx.additional.get("candidate_source")
        matching_candidate_count = ctx.additional.get("matching_candidate_count", 0)
        expected_center = ctx.additional.get("expected_center")
        thresholds = ctx.additional.get("thresholds", {})

        if not sticker or not sticker.get("enabled", True):
            return Decision.rejected("DISABLED", {
                "mode": "sticker",
                "status": "disabled",
            })

        if selected_candidate is None:
            return Decision.rejected(RejectReasonCode.NOT_FOUND.value, {
                "mode": "sticker",
                "status": "inferring",
                "candidate_source": "none",
                "selected_candidate": None,
                "candidate_count": len(candidates),
                "matching_candidate_count": matching_candidate_count,
                "expected_center": expected_center,
                "tilt": tilt_info,
                "thresholds": thresholds,
                "candidates": candidates,
            })

        offset_x = float((selected_candidate.get("offset") or {}).get("x", 0.0))
        offset_y = float((selected_candidate.get("offset") or {}).get("y", 0.0))
        reject_reason = None

        if float(selected_candidate.get("confidence") or 0.0) < thresholds.get("min_roi_confidence", 0.0):
            reject_reason = RejectReasonCode.LOW_ROI_CONF.value
        elif not bool(selected_candidate.get("match_expected")):
            reject_reason = RejectReasonCode.WRONG_TYPE.value
        elif thresholds.get("min_class_confidence") is not None and float(selected_candidate.get("class_confidence") or 0.0) < float(thresholds["min_class_confidence"]):
            reject_reason = RejectReasonCode.LOW_CLASS_CONF.value
        elif criteria.get("tilt_gate_enabled", False) and criteria.get("max_tilt_degrees") is not None and tilt_info.get("angle_degrees") is not None and float(tilt_info.get("deviation_degrees") or 0.0) > criteria["max_tilt_degrees"]:
            reject_reason = RejectReasonCode.OUT_OF_ANGLE.value
        elif criteria.get("tilt_gate_enabled", False) and criteria.get("max_tilt_degrees") is not None:
            roi_check = (tilt_info or {}).get("text_band_roi_check") or {}
            if roi_check and not roi_check.get("is_inside", True):
                reject_reason = RejectReasonCode.OUT_OF_ANGLE.value
        elif criteria.get("position_gate_enabled", False) and thresholds.get("max_offset_x") is not None and abs(offset_x) > float(thresholds["max_offset_x"]):
            reject_reason = RejectReasonCode.OUT_OF_POSITION.value
        elif criteria.get("position_gate_enabled", False) and thresholds.get("max_offset_y") is not None and abs(offset_y) > float(thresholds["max_offset_y"]):
            reject_reason = RejectReasonCode.OUT_OF_POSITION.value

        accept = reject_reason is None
        status = "pass" if accept else reject_reason.lower()

        details = {
            "mode": "sticker",
            "status": status,
            "candidate_source": candidate_source,
            "selected_candidate": selected_candidate,
            "candidate_count": len(candidates),
            "matching_candidate_count": matching_candidate_count,
            "expected_center": expected_center,
            "tilt": tilt_info,
            "thresholds": thresholds,
            "candidates": candidates,
            "detected_class": selected_candidate.get("label"),
            "confidence": selected_candidate.get("confidence"),
            "class_confidence": selected_candidate.get("class_confidence"),
            "bbox": dict(selected_candidate.get("bbox") or {}) if selected_candidate.get("bbox") else None,
            "offset": {"x": round(offset_x, 2), "y": round(offset_y, 2)},
        }

        if accept:
            return Decision.accepted(details)
        return Decision.rejected(reject_reason, details)
