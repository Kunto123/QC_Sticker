"""Golden template fixtures + light multi-mode dispatch integration.

The three JSON files under fixtures/ are FROZEN contracts derived from real
template shapes in the codebase (sticker: templates_repository._sample_template;
counter/defect: the ComponentRoiRule / DefectEvaluator criteria contracts).

What this locks in for later phases:
1. All three golden templates parse (template_from_dict) without raising.
2. Their criteria validate (validate_criteria returns no errors).
3. Each dispatches through the evaluator registry to a Decision whose
   validation_details survive intact and are JSON-safe (allow_nan=False).

The sticker mode has NO wired evaluator yet (registry.py TODO B5 — sticker uses
an inline path in InspectionSessionService), so sticker dispatch is asserted at
the has_evaluator/normalize_mode level rather than through registry.get_evaluator.
That gap is documented in HANDOFF.md.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from shared.contracts.templates import (
    normalize_mode,
    template_from_dict,
    validate_criteria,
)
from backend.app.models.session_state import SessionState
from backend.app.services.evaluators import registry
from backend.app.services.evaluators.base import EvalContext

FIXTURES = Path(__file__).resolve().parent / "fixtures"

GOLDEN = {
    "sticker": FIXTURES / "golden_template_sticker.json",
    "counter": FIXTURES / "golden_template_counter.json",
    "defect": FIXTURES / "golden_template_defect.json",
}


def _load(mode: str) -> dict:
    return json.loads(GOLDEN[mode].read_text(encoding="utf-8"))


class _StubScorer:
    def score(self, crop):  # noqa: ANN001
        return 0.05, np.full((100, 100), 0.05, dtype=np.float32)


class GoldenTemplateParseTest(unittest.TestCase):
    def test_all_three_fixtures_exist(self) -> None:
        for mode, path in GOLDEN.items():
            with self.subTest(mode=mode):
                self.assertTrue(path.exists(), f"missing golden fixture: {path}")

    def test_all_parse_without_raising(self) -> None:
        for mode in GOLDEN:
            with self.subTest(mode=mode):
                tpl = template_from_dict(_load(mode))
                self.assertEqual(tpl.mode, mode)

    def test_all_roundtrip_idempotent(self) -> None:
        for mode in GOLDEN:
            with self.subTest(mode=mode):
                tpl = template_from_dict(_load(mode))
                d1 = tpl.to_dict()
                d2 = template_from_dict(d1).to_dict()
                self.assertEqual(d1, d2)

    def test_all_criteria_validate_clean(self) -> None:
        for mode in GOLDEN:
            with self.subTest(mode=mode):
                tpl = template_from_dict(_load(mode))
                errors = validate_criteria(tpl.mode, tpl.criteria)
                self.assertEqual(errors, [], f"{mode} criteria invalid: {errors}")


class GoldenTemplateDispatchTest(unittest.TestCase):
    """Simulate InspectionSessionService's per-mode evaluate dispatch with stubs."""

    def _state(self, tpl) -> SessionState:
        return SessionState(session_id="s", client_id="c", camera_index=0, template=tpl)

    def _ctx(self, tpl, *, detections=None, additional=None) -> EvalContext:
        return EvalContext(
            detections=detections or [],
            frame=np.zeros((200, 200, 3), dtype=np.uint8),
            criteria=tpl.criteria,
            state=self._state(tpl),
            additional=additional or {},
        )

    def test_counter_dispatch_carries_validation_details(self) -> None:
        tpl = template_from_dict(_load("counter"))
        # criteria.component_rois is what the evaluator reads; ensure it's present
        crit = dict(tpl.criteria)
        crit.setdefault("component_rois", [
            {
                "name": cr.name,
                "roi": {"x": cr.roi.x, "y": cr.roi.y, "w": cr.roi.w, "h": cr.roi.h},
                "classes": [
                    {"class_name": ct.class_name, "min_count": ct.min_count, "max_count": ct.max_count}
                    for ct in cr.classes
                ],
                "strict_foreign_class": cr.strict_foreign_class,
            }
            for cr in tpl.component_rois
        ])
        evaluator = registry.get_evaluator(tpl.mode)
        # supply detections satisfying ROI-0 (2 bolts + 1 washer) and ROI-1 (4 nuts)
        dets = (
            [{"label": "bolt", "tile_index": 0}] * 2
            + [{"label": "washer", "tile_index": 0}]
            + [{"label": "nut", "tile_index": 1}] * 4
        )
        ctx = EvalContext(
            detections=dets, frame=np.zeros((200, 200, 3), dtype=np.uint8),
            criteria=crit, state=self._state(tpl),
            additional={"accept_stable_frames": 1},
        )
        decision = evaluator.evaluate(ctx)
        self.assertTrue(decision.accept, decision.details)
        self.assertEqual(decision.details["mode"], "counter")
        self.assertEqual(len(decision.details["rois"]), 2)
        json.dumps(decision.details, allow_nan=False)  # JSON safety

    def test_defect_dispatch_carries_validation_details(self) -> None:
        tpl = template_from_dict(_load("defect"))
        evaluator = registry.get_evaluator(tpl.mode)
        with mock.patch(
            "backend.app.services.evaluators.defect.get_scorer",
            return_value=_StubScorer(),
        ):
            decision = evaluator.evaluate(self._ctx(tpl))
        self.assertTrue(decision.accept, decision.details)
        self.assertEqual(decision.details["mode"], "defect")
        self.assertEqual(len(decision.details["rois"]), 2)
        json.dumps(decision.details, allow_nan=False)  # JSON safety

    def test_sticker_mode_normalizes_but_has_no_wired_evaluator(self) -> None:
        # Documents the current state: sticker parses to mode 'sticker' but the
        # registry does not (yet) wire a StickerEvaluator (see HANDOFF.md B5).
        tpl = template_from_dict(_load("sticker"))
        self.assertEqual(normalize_mode(tpl.mode), "sticker")
        self.assertFalse(
            registry.has_evaluator("sticker"),
            "If sticker is now wired into the registry, update HANDOFF.md and this test.",
        )


if __name__ == "__main__":
    unittest.main()
