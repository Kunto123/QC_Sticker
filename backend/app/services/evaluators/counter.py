from __future__ import annotations

from collections import Counter
from typing import Any

from shared.contracts.decision import Decision
from backend.app.services.evaluators.base import ModeEvaluator, EvalContext


class CounterEvaluator(ModeEvaluator):
    """Evaluator for component_count mode.

    Core decision logic extracted from InspectionSessionService._validate_component_count.
    Rules per ROI:
    1. For each class: min_count <= detected <= max_count (max_count=None = unlimited)
    2. (if strict_foreign_class) no foreign class detections in ROI
    No total_ok gate.
    """

    mode_name = "counter"

    def evaluate(self, ctx: EvalContext) -> Decision:
        component_rois = ctx.criteria.get("component_rois", [])
        if not component_rois:
            return Decision.rejected("NO_COMPONENT_ROIS", {
                "mode": "counter",
                "error": "No component ROIs defined",
            })

        detections = ctx.detections
        # Count detections per (ROI_index, class)
        roi_counts: list[Counter] = [Counter() for _ in component_rois]
        for det in detections:
            label = str(det.get("label") or det.get("class_name") or "").strip().lower()
            tile_idx = det.get("tile_index", -1)
            if 0 <= tile_idx < len(component_rois):
                roi_counts[tile_idx][label] += 1

        all_ok = True
        roi_results = []
        reject_reason = None

        for roi_idx, roi_rule in enumerate(component_rois):
            roi_ok = True
            class_results = {}
            registered_classes = set()

            for ct in roi_rule.get("classes", []):
                cn = ct.get("class_name", "").strip().lower()
                if not cn:
                    continue
                min_count = int(ct.get("min_count", ct.get("count", 1)))
                max_count = ct.get("max_count")  # None = unlimited
                registered_classes.add(cn)
                detected = roi_counts[roi_idx].get(cn, 0)
                ok = detected >= min_count and (max_count is None or detected <= max_count)
                class_results[cn] = {
                    "detected": detected,
                    "min": min_count,
                    "max": max_count if max_count is not None else None,  # None = unlimited, UI displays as "∞"
                    "ok": ok,
                }
                if not ok:
                    roi_ok = False

            total_detected = sum(roi_counts[roi_idx].values())

            foreign_classes = []
            if roi_rule.get("strict_foreign_class", False):
                for cls_name in roi_counts[roi_idx]:
                    if cls_name not in registered_classes:
                        foreign_classes.append(cls_name)
                        roi_ok = False

            if not roi_ok:
                all_ok = False
                if not reject_reason:
                    reject_reason = "UNEXPECTED_COMPONENT" if foreign_classes else "COMPONENT_COUNT_MISMATCH"

            roi_results.append({
                "name": roi_rule.get("name", f"ROI {roi_idx}"),
                "ok": roi_ok,
                "classes": class_results,
                "total_detected": total_detected,
                "foreign_classes": foreign_classes,
                "roi": roi_rule.get("roi", {}),
            })

        consecutive_ok = getattr(ctx.state, "consecutive_component_ok", 0)
        _needed = max(1, int(ctx.additional.get("accept_stable_frames", 2)))

        if all_ok:
            consecutive_ok += 1
            ctx.state.consecutive_component_ok = consecutive_ok
        else:
            consecutive_ok = 0
            ctx.state.consecutive_component_ok = 0

        all_ok_final = all_ok and consecutive_ok >= _needed

        details = {
            "mode": "counter",
            "rois": roi_results,
            "consecutive_ok": consecutive_ok,
            "consecutive_needed": _needed,
            "all_rois_ok": all_ok,
        }

        if all_ok_final:
            return Decision.accepted(details)
        elif all_ok:
            # stabilizing, not yet committed — return as pending
            details["stabilizing"] = True
            return Decision.rejected("STABILIZING", details)
        else:
            return Decision.rejected(reject_reason or "COMPONENT_COUNT_MISMATCH", details)
