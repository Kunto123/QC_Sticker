"""Golden sticker template fixture.

`fixtures/golden_template_sticker.json` is a FROZEN template payload as it was
written by an older release (it still carries `validator_mode` and the tilt
fields). What this locks in:

1. The payload parses (template_from_dict) without raising.
2. Keys the current contract no longer knows are dropped, not carried along.
3. The `sticker` section validates (validate_sticker_rule returns no errors).
4. to_dict() round-trips to the current contract shape and is JSON-safe.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from shared.contracts.templates import template_from_dict, validate_sticker_rule

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_template_sticker.json"


def _load() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class GoldenStickerTemplateTest(unittest.TestCase):
    def test_fixture_exists(self) -> None:
        self.assertTrue(FIXTURE.exists(), f"missing golden fixture: {FIXTURE}")

    def test_parses_without_raising(self) -> None:
        template = template_from_dict(_load())
        self.assertEqual(template.sticker.expected_class, "K0W-HB0")
        self.assertEqual(template.name, _load()["name"])

    def test_legacy_keys_are_dropped(self) -> None:
        payload = _load()
        # The frozen payload still carries pre-cleanup fields.
        self.assertIn("validator_mode", payload["sticker"])
        self.assertIn("tilt_gate_enabled", payload["sticker"])
        template = template_from_dict(payload)
        serialized = template.to_dict()
        for key in ("validator_mode", "tilt_gate_enabled", "max_tilt_degrees", "expected_tilt_degrees"):
            self.assertNotIn(key, serialized["sticker"])
        for key in ("mode", "criteria", "component_rois"):
            self.assertNotIn(key, serialized)

    def test_sticker_rule_validates_clean(self) -> None:
        self.assertEqual(validate_sticker_rule(_load()["sticker"]), [])

    def test_roundtrip_is_json_safe_and_stable(self) -> None:
        first = template_from_dict(_load()).to_dict()
        second = template_from_dict(first).to_dict()
        self.assertEqual(first, second)
        json.dumps(first, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
