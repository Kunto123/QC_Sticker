"""Contract tests for shared/contracts/templates.py.

These are FROZEN behavioral contracts for the multi-mode template layer:
parsing, round-tripping, mode normalization and criteria validation. Later
deployment-hardening phases rely on these staying green.

Covers:
- Round-trip idempotency across all modes (dict -> parse -> to_dict -> parse).
- Legacy-only payloads (no mode/criteria) still parse and back-fill mode.
- Criteria-only / new-format payloads back-fill component_rois.
- normalize_mode covers every alias.
- validate_criteria rejects broken input with a clear message.
- min/max count semantics: max=None (unbounded), min=0/max=0, count-only legacy.
"""
from __future__ import annotations

import unittest

from shared.contracts.templates import (
    ComponentClassTarget,
    InspectionTemplate,
    normalize_mode,
    template_from_dict,
    validate_criteria,
)


def _minimal_sticker_payload() -> dict:
    """A minimal legacy-flat sticker template payload (no mode/criteria)."""
    return {
        "id": 1,
        "version_id": 1,
        "version_number": 1,
        "name": "Sticker Template",
        "description": "",
        "is_active": True,
        "camera": {"camera_index": 0},
        "part_ready_roi": {"x": 0.2, "y": 0.2, "w": 0.5, "h": 0.5},
        "sticker_roi": {"x": 0.1, "y": 0.1, "w": 0.6, "h": 0.6},
        "vision": {"model_path": "models/sticker.pt", "conf_threshold": 0.3},
        "part_ready": {"enabled": True, "method": "gap_template_match"},
        "sticker": {
            "part_name": "P1",
            "expected_class": "K0W-HB0",
            "enabled": True,
            "validator_mode": "ml_detection",
            "min_roi_confidence": 0.25,
        },
        "persistence": {"write_to_db": True},
    }


def _counter_payload_legacy() -> dict:
    """Counter template in the legacy top-level component_rois format."""
    p = _minimal_sticker_payload()
    p["name"] = "Counter Template"
    p["sticker"]["validator_mode"] = "component_count"
    p["component_rois"] = [
        {
            "name": "ROI-A",
            "roi": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0},
            "classes": [
                {"class_name": "bolt", "count": 4},
                {"class_name": "washer", "min_count": 0, "max_count": None},
            ],
            "strict_foreign_class": True,
        }
    ]
    return p


def _counter_payload_criteria_only() -> dict:
    """Counter template where component_rois live ONLY in criteria (new format)."""
    p = _minimal_sticker_payload()
    p["name"] = "Counter New Format"
    p["mode"] = "counter"
    p["sticker"]["validator_mode"] = "component_count"
    p["criteria"] = {
        "component_rois": [
            {
                "name": "ROI-1",
                "roi": {"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.4},
                "classes": [{"class_name": "screw", "count": 2, "min_count": 2, "max_count": 2}],
                "strict_foreign_class": False,
            }
        ]
    }
    return p


class NormalizeModeTest(unittest.TestCase):
    def test_all_aliases_resolve(self) -> None:
        cases = {
            "component_count": "counter",
            "count": "counter",
            "counter": "counter",
            "ml_detection": "sticker",
            "sticker": "sticker",
            "defect": "defect",
            "": "sticker",
            None: "sticker",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_mode(raw), expected)

    def test_case_and_whitespace_insensitive(self) -> None:
        self.assertEqual(normalize_mode("  COMPONENT_COUNT  "), "counter")
        self.assertEqual(normalize_mode("Defect"), "defect")

    def test_unknown_mode_returned_as_is(self) -> None:
        # normalize_mode deliberately passes unknown values through so validators
        # can reject them with a clear message.
        self.assertEqual(normalize_mode("banana"), "banana")


class TemplateParseTest(unittest.TestCase):
    def test_legacy_only_payload_parses_and_backfills_mode(self) -> None:
        tpl = template_from_dict(_minimal_sticker_payload())
        self.assertIsInstance(tpl, InspectionTemplate)
        self.assertEqual(tpl.mode, "sticker")
        # criteria back-filled from legacy fields
        self.assertEqual(tpl.criteria.get("expected_class"), "K0W-HB0")

    def test_counter_legacy_backfills_mode_and_rois(self) -> None:
        tpl = template_from_dict(_counter_payload_legacy())
        self.assertEqual(tpl.mode, "counter")
        self.assertEqual(len(tpl.component_rois), 1)
        self.assertEqual(tpl.component_rois[0].name, "ROI-A")

    def test_counter_criteria_only_backfills_component_rois(self) -> None:
        tpl = template_from_dict(_counter_payload_criteria_only())
        self.assertEqual(tpl.mode, "counter")
        # component_rois came from criteria, not top-level
        self.assertEqual(len(tpl.component_rois), 1)
        self.assertEqual(tpl.component_rois[0].name, "ROI-1")
        self.assertEqual(tpl.component_rois[0].classes[0].class_name, "screw")

    def test_unknown_roi_keys_are_stripped(self) -> None:
        # old DB data sometimes carries a 'height' key RoiGeometry rejects
        p = _minimal_sticker_payload()
        p["sticker_roi"] = {"x": 0.1, "y": 0.1, "w": 0.6, "h": 0.6, "height": 999}
        tpl = template_from_dict(p)  # must not raise
        self.assertEqual(tpl.sticker_roi.w, 0.6)


class RoundTripTest(unittest.TestCase):
    """dict -> parse -> to_dict -> parse must be idempotent for every mode."""

    def _assert_roundtrip_idempotent(self, payload: dict) -> None:
        tpl1 = template_from_dict(payload)
        d1 = tpl1.to_dict()
        tpl2 = template_from_dict(d1)
        d2 = tpl2.to_dict()
        self.assertEqual(d1, d2, "second round-trip diverged from first")
        self.assertEqual(tpl1.mode, tpl2.mode)

    def test_sticker_roundtrip(self) -> None:
        self._assert_roundtrip_idempotent(_minimal_sticker_payload())

    def test_counter_legacy_roundtrip(self) -> None:
        self._assert_roundtrip_idempotent(_counter_payload_legacy())

    def test_counter_criteria_only_roundtrip(self) -> None:
        self._assert_roundtrip_idempotent(_counter_payload_criteria_only())

    def test_roundtrip_preserves_mode_and_criteria_keys(self) -> None:
        tpl = template_from_dict(_counter_payload_legacy())
        d = tpl.to_dict()
        self.assertIn("mode", d)
        self.assertIn("criteria", d)
        self.assertEqual(d["mode"], "counter")


class ComponentClassTargetSemanticsTest(unittest.TestCase):
    """min/max count semantics survive parsing and round-trip."""

    def test_count_only_legacy_becomes_exact_match(self) -> None:
        ct = ComponentClassTarget(class_name="bolt", count=4)
        self.assertEqual(ct.min_count, 4)
        self.assertEqual(ct.max_count, 4)

    def test_max_none_stays_unbounded(self) -> None:
        ct = ComponentClassTarget(class_name="washer", count=1, min_count=1, max_count=None)
        self.assertEqual(ct.min_count, 1)
        self.assertIsNone(ct.max_count, "max_count=None must survive as unbounded")

    def test_min_zero_is_preserved(self) -> None:
        ct = ComponentClassTarget(class_name="opt", count=0, min_count=0, max_count=3)
        self.assertEqual(ct.min_count, 0)
        self.assertEqual(ct.max_count, 3)

    def test_only_min_set_leaves_max_unbounded(self) -> None:
        ct = ComponentClassTarget(class_name="x", count=5, min_count=2)
        self.assertEqual(ct.min_count, 2)
        self.assertIsNone(ct.max_count)

    def test_unbounded_max_survives_full_roundtrip(self) -> None:
        p = _counter_payload_legacy()  # 'washer' has max_count=None
        tpl = template_from_dict(p)
        washer = next(
            c for c in tpl.component_rois[0].classes if c.class_name == "washer"
        )
        self.assertIsNone(washer.max_count)
        # round-trip through dict
        tpl2 = template_from_dict(tpl.to_dict())
        washer2 = next(
            c for c in tpl2.component_rois[0].classes if c.class_name == "washer"
        )
        self.assertIsNone(washer2.max_count)


class ValidateCriteriaTest(unittest.TestCase):
    def test_sticker_requires_expected_class(self) -> None:
        errors = validate_criteria("sticker", {})
        self.assertTrue(errors)
        self.assertIn("expected_class", errors[0])

    def test_sticker_valid_returns_no_errors(self) -> None:
        self.assertEqual(validate_criteria("sticker", {"expected_class": "K0W"}), [])

    def test_sticker_confidence_out_of_range_rejected(self) -> None:
        errors = validate_criteria("sticker", {"expected_class": "K0W", "min_roi_confidence": 5})
        self.assertTrue(any("out of range" in e for e in errors))

    def test_sticker_confidence_non_float_rejected(self) -> None:
        errors = validate_criteria(
            "sticker", {"expected_class": "K0W", "min_roi_confidence": "abc"}
        )
        self.assertTrue(any("must be a float" in e for e in errors))

    def test_counter_requires_at_least_one_roi(self) -> None:
        errors = validate_criteria("counter", {"component_rois": []})
        self.assertTrue(any("at least one component_roi" in e for e in errors))

    def test_counter_min_gt_max_rejected_with_clear_message(self) -> None:
        criteria = {
            "component_rois": [
                {
                    "name": "R",
                    "classes": [{"class_name": "bolt", "min_count": 5, "max_count": 2}],
                }
            ]
        }
        errors = validate_criteria("counter", criteria)
        self.assertTrue(any("min > max" in e for e in errors), errors)

    def test_counter_missing_class_name_rejected(self) -> None:
        criteria = {"component_rois": [{"name": "R", "classes": [{"class_name": ""}]}]}
        errors = validate_criteria("counter", criteria)
        self.assertTrue(any("no class_name" in e for e in errors), errors)

    def test_defect_requires_roi(self) -> None:
        errors = validate_criteria("defect", {"rois": []})
        self.assertTrue(any("at least one ROI" in e for e in errors))

    def test_defect_threshold_out_of_range_rejected(self) -> None:
        criteria = {"rois": [{"name": "R", "geometry": {"x": 0}, "threshold": 9.0}]}
        errors = validate_criteria("defect", criteria)
        self.assertTrue(any("out of range" in e for e in errors), errors)

    def test_defect_missing_geometry_rejected(self) -> None:
        criteria = {"rois": [{"name": "R", "threshold": 0.5}]}
        errors = validate_criteria("defect", criteria)
        self.assertTrue(any("no geometry" in e for e in errors), errors)

    def test_unknown_mode_rejected_with_clear_message(self) -> None:
        errors = validate_criteria("banana", {})
        self.assertTrue(errors)
        self.assertIn("unknown mode", errors[0])

    def test_validate_normalizes_mode_alias(self) -> None:
        # 'component_count' alias must route to the counter validator, not 'unknown'
        errors = validate_criteria("component_count", {"component_rois": []})
        self.assertTrue(any("component_roi" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
