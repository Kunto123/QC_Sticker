"""Unit tests for backend/app/services/evaluators/*.

These import the evaluator package DIRECTLY (the core coverage gap: zero tests
touched it before FASE 0). Each evaluator is a pure-ish function of
(frame, detections, criteria, state) -> Decision.

Covers:
- CounterEvaluator: in-range, out-of-range, foreign class, multi-ROI.
- DefectEvaluator (scorer STUB): all-OK, one ROI NG, empty crop, model load fail.
- StickerEvaluator: pass, wrong-type, low-confidence, disabled, not-found.
- registry: unknown mode rejected; known modes resolve.
- JSON safety: json.dumps(details, allow_nan=False) on every branch — no
  Infinity / NaN leaks into the Decision.details payload.
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

import numpy as np

from shared.contracts.decision import Decision
from shared.contracts.templates import template_from_dict
from backend.app.models.session_state import SessionState
from backend.app.services.evaluators.base import EvalContext
from backend.app.services.evaluators.counter import CounterEvaluator
from backend.app.services.evaluators.defect import DefectEvaluator
from backend.app.services.evaluators.sticker import StickerEvaluator
from backend.app.services.evaluators import registry


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_state() -> SessionState:
    payload = {
        "id": 1, "version_id": 1, "version_number": 1, "name": "t",
        "description": "", "is_active": True, "camera": {"camera_index": 0},
        "part_ready_roi": {}, "sticker_roi": {}, "vision": {}, "part_ready": {},
        "sticker": {"part_name": "P", "expected_class": "C"}, "persistence": {},
    }
    tpl = template_from_dict(payload)
    return SessionState(session_id="s", client_id="c", camera_index=0, template=tpl)


def _ctx(*, detections=None, criteria=None, frame=None, additional=None) -> EvalContext:
    return EvalContext(
        detections=detections or [],
        frame=frame if frame is not None else np.zeros((100, 100, 3), dtype=np.uint8),
        criteria=criteria or {},
        state=_make_state(),
        additional=additional or {},
    )


def _assert_json_safe(decision: Decision) -> None:
    """A Decision must serialize with allow_nan=False — no Infinity/NaN leaks."""
    json.dumps(decision.details, allow_nan=False)


class _StubScorer:
    """Deterministic anomaly scorer stub. score() returns (scalar, heatmap)."""

    def __init__(self, value: float, size: tuple[int, int] = (8, 8)) -> None:
        self._value = value
        self._size = size

    def score(self, crop):  # noqa: ANN001 - matches AnomalyScorer.score signature
        h, w = self._size
        return self._value, np.full((h, w), self._value, dtype=np.float32)


class _RaisingScorer:
    def score(self, crop):  # noqa: ANN001
        raise RuntimeError("model failed to load")


# ── Counter ───────────────────────────────────────────────────────────────────

class CounterEvaluatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ev = CounterEvaluator()

    def _roi(self, class_name="bolt", min_c=2, max_c=2, strict=False, name="ROI-0"):
        return {
            "name": name,
            "roi": {"x": 0, "y": 0, "w": 1, "h": 1},
            "classes": [{"class_name": class_name, "min_count": min_c, "max_count": max_c}],
            "strict_foreign_class": strict,
        }

    def test_in_range_accepts(self) -> None:
        crit = {"component_rois": [self._roi(min_c=2, max_c=2)]}
        dets = [{"label": "bolt", "tile_index": 0}, {"label": "bolt", "tile_index": 0}]
        # counter requires accept_stable_frames consecutive OKs; default is 2.
        d = self.ev.evaluate(_ctx(detections=dets, criteria=crit, additional={"accept_stable_frames": 1}))
        self.assertTrue(d.accept, d.details)
        self.assertEqual(d.details["rois"][0]["classes"]["bolt"]["detected"], 2)
        _assert_json_safe(d)

    def test_out_of_range_rejects(self) -> None:
        crit = {"component_rois": [self._roi(min_c=3, max_c=3)]}
        dets = [{"label": "bolt", "tile_index": 0}]  # only 1, need 3
        d = self.ev.evaluate(_ctx(detections=dets, criteria=crit))
        self.assertFalse(d.accept)
        self.assertEqual(d.reason_code, "COMPONENT_COUNT_MISMATCH")
        _assert_json_safe(d)

    def test_foreign_class_rejects_when_strict(self) -> None:
        crit = {"component_rois": [self._roi(class_name="bolt", min_c=1, max_c=1, strict=True)]}
        dets = [
            {"label": "bolt", "tile_index": 0},
            {"label": "intruder", "tile_index": 0},
        ]
        d = self.ev.evaluate(_ctx(detections=dets, criteria=crit))
        self.assertFalse(d.accept)
        self.assertEqual(d.reason_code, "UNEXPECTED_COMPONENT")
        self.assertIn("intruder", d.details["rois"][0]["foreign_classes"])
        _assert_json_safe(d)

    def test_foreign_class_ignored_when_not_strict(self) -> None:
        crit = {"component_rois": [self._roi(class_name="bolt", min_c=1, max_c=1, strict=False)]}
        dets = [{"label": "bolt", "tile_index": 0}, {"label": "intruder", "tile_index": 0}]
        d = self.ev.evaluate(_ctx(detections=dets, criteria=crit, additional={"accept_stable_frames": 1}))
        self.assertTrue(d.accept, d.details)
        _assert_json_safe(d)

    def test_multi_roi_one_bad_rejects(self) -> None:
        crit = {
            "component_rois": [
                self._roi(class_name="bolt", min_c=1, max_c=1, name="ROI-0"),
                self._roi(class_name="nut", min_c=2, max_c=2, name="ROI-1"),
            ]
        }
        dets = [
            {"label": "bolt", "tile_index": 0},
            {"label": "nut", "tile_index": 1},  # only 1 nut, need 2
        ]
        d = self.ev.evaluate(_ctx(detections=dets, criteria=crit))
        self.assertFalse(d.accept)
        self.assertEqual(len(d.details["rois"]), 2)
        self.assertTrue(d.details["rois"][0]["ok"])
        self.assertFalse(d.details["rois"][1]["ok"])
        _assert_json_safe(d)

    def test_no_component_rois_rejects(self) -> None:
        d = self.ev.evaluate(_ctx(criteria={"component_rois": []}))
        self.assertFalse(d.accept)
        self.assertEqual(d.reason_code, "NO_COMPONENT_ROIS")
        _assert_json_safe(d)

    def test_unbounded_max_accepts_any_count(self) -> None:
        crit = {"component_rois": [self._roi(min_c=1, max_c=None)]}
        dets = [{"label": "bolt", "tile_index": 0}] * 9
        d = self.ev.evaluate(_ctx(detections=dets, criteria=crit, additional={"accept_stable_frames": 1}))
        self.assertTrue(d.accept, d.details)
        self.assertIsNone(d.details["rois"][0]["classes"]["bolt"]["max"])
        _assert_json_safe(d)

    def test_stabilizing_before_threshold(self) -> None:
        crit = {"component_rois": [self._roi(min_c=1, max_c=1)]}
        dets = [{"label": "bolt", "tile_index": 0}]
        # need 2 consecutive OKs; first frame is OK but not yet committed
        d = self.ev.evaluate(_ctx(detections=dets, criteria=crit, additional={"accept_stable_frames": 2}))
        self.assertFalse(d.accept)
        self.assertEqual(d.reason_code, "STABILIZING")
        _assert_json_safe(d)


# ── Defect (scorer stubbed) ───────────────────────────────────────────────────

class DefectEvaluatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ev = DefectEvaluator()

    def _roi(self, name="R", x=0.0, y=0.0, w=0.5, h=0.5, threshold=0.5):
        return {"name": name, "geometry": {"x": x, "y": y, "w": w, "h": h}, "threshold": threshold}

    def test_all_ok_accepts(self) -> None:
        crit = {"rois": [self._roi(threshold=0.9)], "inference_mode": "per_roi_crop"}
        with mock.patch(
            "backend.app.services.evaluators.defect.get_scorer",
            return_value=_StubScorer(0.1),
        ):
            d = self.ev.evaluate(_ctx(criteria=crit))
        self.assertTrue(d.accept, d.details)
        self.assertTrue(d.details["rois"][0]["ok"])
        _assert_json_safe(d)

    def test_one_roi_ng_rejects(self) -> None:
        crit = {
            "rois": [
                self._roi(name="good", x=0.0, w=0.4, threshold=0.9),
                self._roi(name="bad", x=0.5, w=0.4, threshold=0.2),
            ],
            "inference_mode": "per_roi_crop",
        }
        with mock.patch(
            "backend.app.services.evaluators.defect.get_scorer",
            return_value=_StubScorer(0.5),  # 0.5 > 0.2 threshold for 'bad'
        ):
            d = self.ev.evaluate(_ctx(criteria=crit))
        self.assertFalse(d.accept)
        self.assertEqual(d.reason_code, "ANOMALY_DETECTED")
        _assert_json_safe(d)

    def test_degenerate_geometry_stays_json_safe(self) -> None:
        # A degenerate ROI (zero fractional size, off-frame origin) must not
        # crash and must produce a JSON-safe Decision. NOTE: _parse_geometry
        # clamps w/h to >=1px (max(1,...)), so the literal w<=0 "empty crop"
        # branch in DefectEvaluator is effectively unreachable via geometry —
        # recorded as a code-smell in HANDOFF.md. This test locks in JSON safety
        # of whatever branch is actually taken.
        crit = {
            "rois": [{"name": "R", "geometry": {"x": 1.0, "y": 1.0, "w": 0.0, "h": 0.0}, "threshold": 0.5}],
            "inference_mode": "per_roi_crop",
        }
        with mock.patch(
            "backend.app.services.evaluators.defect.get_scorer",
            return_value=_StubScorer(0.1),
        ):
            d = self.ev.evaluate(_ctx(criteria=crit))
        _assert_json_safe(d)

    def test_aggregate_score_empty_slice_returns_none_json_safe(self) -> None:
        # Empty heatmap slice should return None (JSON-safe), not float('inf').
        # This prevents json.dumps(..., allow_nan=False) from raising.
        from backend.app.services.evaluators.defect import _aggregate_score

        empty = np.empty((0, 0), dtype=np.float32)
        self.assertIsNone(_aggregate_score(empty))
        # Verify JSON safety
        import json
        json.dumps({"score": _aggregate_score(empty)}, allow_nan=False)

    def test_model_load_failure_rejects(self) -> None:
        crit = {"rois": [self._roi(threshold=0.5)], "inference_mode": "per_roi_crop"}
        with mock.patch(
            "backend.app.services.evaluators.defect.get_scorer",
            return_value=_RaisingScorer(),
        ):
            d = self.ev.evaluate(_ctx(criteria=crit))
        self.assertFalse(d.accept)
        self.assertEqual(d.reason_code, "ANOMALY_DETECTED")
        # error string is recorded on the ROI result
        self.assertIn("failed", d.details["rois"][0]["error"])
        _assert_json_safe(d)

    def test_no_rois_rejects(self) -> None:
        d = self.ev.evaluate(_ctx(criteria={"rois": []}))
        self.assertFalse(d.accept)
        self.assertEqual(d.reason_code, "NO_DEFECT_ROIS")
        _assert_json_safe(d)

    def test_no_frame_rejects(self) -> None:
        ctx = _ctx(criteria={"rois": [self._roi()]}, frame=np.zeros((0, 0, 3), dtype=np.uint8))
        d = self.ev.evaluate(ctx)
        self.assertFalse(d.accept)
        self.assertEqual(d.reason_code, "NO_FRAME")
        _assert_json_safe(d)

    def test_whole_part_mode_accepts(self) -> None:
        crit = {"rois": [self._roi(threshold=0.9)], "inference_mode": "whole_part"}
        with mock.patch(
            "backend.app.services.evaluators.defect.get_scorer",
            return_value=_StubScorer(0.1, size=(100, 100)),
        ):
            d = self.ev.evaluate(_ctx(criteria=crit))
        self.assertTrue(d.accept, d.details)
        _assert_json_safe(d)


# ── Sticker (pure-context, no model) ──────────────────────────────────────────

class StickerEvaluatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ev = StickerEvaluator()

    def _candidate(self, **over):
        base = {
            "label": "C",
            "confidence": 0.9,
            "class_confidence": 0.9,
            "match_expected": True,
            "offset": {"x": 0.0, "y": 0.0},
            "bbox": {"x": 1, "y": 1, "w": 2, "h": 2},
        }
        base.update(over)
        return base

    def _additional(self, candidate, **over):
        add = {
            "sticker_rule": {"enabled": True},
            "selected_candidate": candidate,
            "candidates": [candidate] if candidate else [],
            "thresholds": {"min_roi_confidence": 0.5},
        }
        add.update(over)
        return add

    def test_pass_accepts(self) -> None:
        d = self.ev.evaluate(_ctx(additional=self._additional(self._candidate())))
        self.assertTrue(d.accept, d.details)
        self.assertEqual(d.details["status"], "pass")
        _assert_json_safe(d)

    def test_disabled_rule_rejects(self) -> None:
        add = self._additional(self._candidate())
        add["sticker_rule"] = {"enabled": False}
        d = self.ev.evaluate(_ctx(additional=add))
        self.assertFalse(d.accept)
        self.assertEqual(d.reason_code, "DISABLED")
        _assert_json_safe(d)

    def test_no_candidate_rejects_not_found(self) -> None:
        add = self._additional(None)
        d = self.ev.evaluate(_ctx(additional=add))
        self.assertFalse(d.accept)
        self.assertEqual(d.reason_code, "NOT_FOUND")
        _assert_json_safe(d)

    def test_wrong_type_rejects(self) -> None:
        cand = self._candidate(match_expected=False)
        d = self.ev.evaluate(_ctx(additional=self._additional(cand)))
        self.assertFalse(d.accept)
        self.assertEqual(d.reason_code, "WRONG_TYPE")
        _assert_json_safe(d)

    def test_low_roi_confidence_rejects(self) -> None:
        cand = self._candidate(confidence=0.1)
        d = self.ev.evaluate(_ctx(additional=self._additional(cand)))
        self.assertFalse(d.accept)
        self.assertEqual(d.reason_code, "LOW_ROI_CONF")
        _assert_json_safe(d)


# ── Registry ──────────────────────────────────────────────────────────────────

class RegistryTest(unittest.TestCase):
    def test_known_modes_resolve(self) -> None:
        self.assertEqual(registry.get_evaluator("counter").mode_name, "counter")
        self.assertEqual(registry.get_evaluator("defect").mode_name, "defect")

    def test_unknown_mode_raises_keyerror(self) -> None:
        with self.assertRaises(KeyError):
            registry.get_evaluator("banana")

    def test_has_evaluator(self) -> None:
        self.assertTrue(registry.has_evaluator("counter"))
        self.assertFalse(registry.has_evaluator("banana"))

    def test_registered_modes_lists_wired_evaluators(self) -> None:
        modes = registry.registered_modes()
        self.assertIn("counter", modes)
        self.assertIn("defect", modes)


if __name__ == "__main__":
    unittest.main()
