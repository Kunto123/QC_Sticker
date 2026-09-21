"""Contract tests for shared/contracts/templates.py (sticker-only contract).

Covers:
- Legacy payloads (validator_mode / tilt / mode / criteria / component_rois keys)
  still parse, and the unknown keys are dropped.
- Round-trip idempotency (dict -> parse -> to_dict -> parse).
- validate_sticker_rule rejects broken input with a clear message.
"""
from __future__ import annotations

import unittest

from shared.contracts.templates import (
    InspectionTemplate,
    template_from_dict,
    validate_sticker_rule,
)


def _minimal_sticker_payload() -> dict:
    """A minimal template payload as older releases wrote it."""
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


class TemplateParseTest(unittest.TestCase):
    def test_legacy_payload_parses(self) -> None:
        tpl = template_from_dict(_minimal_sticker_payload())
        self.assertIsInstance(tpl, InspectionTemplate)
        self.assertEqual(tpl.sticker.expected_class, "K0W-HB0")
        self.assertEqual(tpl.sticker.min_roi_confidence, 0.25)

    def test_removed_keys_are_dropped_everywhere(self) -> None:
        p = _minimal_sticker_payload()
        p["mode"] = "counter"
        p["criteria"] = {"component_rois": [{"name": "ROI-A"}]}
        p["component_rois"] = [{"name": "ROI-A", "roi": {"x": 0, "y": 0, "w": 1, "h": 1}}]
        p["camera"]["rotation_degrees"] = 90
        p["sticker_roi"]["rotation"] = 45.0
        p["sticker"].update({"tilt_gate_enabled": True, "max_tilt_degrees": 5.0, "use_ocr": True})
        p["vision"].update({"stream_fps": 10, "enable_ergonomic_check": True})
        p["part_ready"].update({"color_profile_id": 3, "hsv_lower": [0, 0, 0]})
        d = template_from_dict(p).to_dict()  # must not raise
        for key in ("mode", "criteria", "component_rois"):
            self.assertNotIn(key, d)
        self.assertNotIn("rotation_degrees", d["camera"])
        self.assertNotIn("rotation", d["sticker_roi"])
        for key in ("validator_mode", "tilt_gate_enabled", "max_tilt_degrees", "use_ocr"):
            self.assertNotIn(key, d["sticker"])
        for key in ("stream_fps", "enable_ergonomic_check"):
            self.assertNotIn(key, d["vision"])
        for key in ("color_profile_id", "hsv_lower"):
            self.assertNotIn(key, d["part_ready"])

    def test_unknown_roi_keys_are_stripped(self) -> None:
        # old DB data sometimes carries a 'height' key RoiGeometry rejects
        p = _minimal_sticker_payload()
        p["sticker_roi"] = {"x": 0.1, "y": 0.1, "w": 0.6, "h": 0.6, "height": 999}
        tpl = template_from_dict(p)  # must not raise
        self.assertEqual(tpl.sticker_roi.w, 0.6)

    def test_empty_part_ready_method_defaults_to_gap(self) -> None:
        p = _minimal_sticker_payload()
        p["part_ready"]["method"] = ""
        self.assertEqual(template_from_dict(p).part_ready.method, "gap_template_match")

    def test_canny_bounds_default_to_none(self) -> None:
        tpl = template_from_dict(_minimal_sticker_payload())
        self.assertIsNone(tpl.part_ready.canny_low)
        self.assertIsNone(tpl.part_ready.canny_high)

    def test_canny_bounds_round_trip(self) -> None:
        p = _minimal_sticker_payload()
        p["part_ready"]["canny_low"] = 40
        p["part_ready"]["canny_high"] = 120
        tpl = template_from_dict(p)
        self.assertEqual(tpl.part_ready.canny_low, 40)
        self.assertEqual(tpl.part_ready.canny_high, 120)
        self.assertEqual(tpl.to_dict()["part_ready"]["canny_low"], 40)
        self.assertEqual(tpl.to_dict()["part_ready"]["canny_high"], 120)


class RoundTripTest(unittest.TestCase):
    def test_sticker_roundtrip_is_idempotent(self) -> None:
        d1 = template_from_dict(_minimal_sticker_payload()).to_dict()
        d2 = template_from_dict(d1).to_dict()
        self.assertEqual(d1, d2, "second round-trip diverged from first")


class ValidateStickerRuleTest(unittest.TestCase):
    def test_requires_expected_class(self) -> None:
        errors = validate_sticker_rule({})
        self.assertTrue(errors)
        self.assertIn("expected_class", errors[0])

    def test_valid_returns_no_errors(self) -> None:
        self.assertEqual(validate_sticker_rule({"expected_class": "K0W"}), [])

    def test_confidence_out_of_range_rejected(self) -> None:
        errors = validate_sticker_rule({"expected_class": "K0W", "min_roi_confidence": 5})
        self.assertTrue(any("out of range" in e for e in errors))

    def test_confidence_non_float_rejected(self) -> None:
        errors = validate_sticker_rule({"expected_class": "K0W", "min_roi_confidence": "abc"})
        self.assertTrue(any("must be a float" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
