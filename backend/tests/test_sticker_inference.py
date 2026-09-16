from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from backend.app.core.config import AppConfig
from backend.app.services.sticker_inference import StickerInferenceService
from shared.contracts.templates import StickerRule, VisionConfig


class _Scalar:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeBox:
    def __init__(self, *, xyxy: list[float], class_id: int, confidence: float) -> None:
        self.xyxy = np.array([xyxy], dtype=float)
        self.cls = [_Scalar(float(class_id))]
        self.conf = [_Scalar(float(confidence))]


class _FakeResult:
    def __init__(self, boxes: list[_FakeBox]) -> None:
        self.boxes = boxes


class StickerInferenceFilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = StickerInferenceService(AppConfig(), mock.Mock())

    def test_normalize_label_key_removes_non_alnum(self) -> None:
        self.assertEqual(self.service._normalize_label_key("K0W-HB0"), "k0whb0")
        self.assertEqual(self.service._normalize_label_key("K0W_HB0"), "k0whb0")

    def test_normalize_detections_accepts_canonical_label_match(self) -> None:
        result = _FakeResult(
            [
                _FakeBox(xyxy=[1.0, 2.0, 3.0, 4.0], class_id=0, confidence=0.91),
                _FakeBox(xyxy=[5.0, 6.0, 7.0, 8.0], class_id=1, confidence=0.88),
            ]
        )
        names = {0: "K0W_HB0", 1: "OTHER"}
        detections = self.service._normalize_detections(
            result=result,
            names=names,
            allowed_labels={"k0w-hb0"},
            allowed_label_keys={"k0whb0"},
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["label"], "K0W_HB0")
        self.assertEqual(detections[0]["class_id"], 0)

    def test_normalize_ocr_text_applies_regex_and_canonical_map(self) -> None:
        result = self.service.normalize_ocr_text(
            " code: k0w hb0 ",
            expected_text="K0W-HB0",
            regex=r"K0W\s*HB0",
            canonical_map={"K0WHB0": "K0W-HB0"},
        )
        self.assertEqual(result, "K0W-HB0")

    def test_parse_unique_code_uses_last_dash_segment(self) -> None:
        self.assertEqual(self.service.parse_unique_code("VEHICLE EMISSION MODEL NAME - ADV160A"), "ADV160A")
        self.assertEqual(self.service.parse_unique_code("CHASIS NO - CH12345678"), "CH12345678")
        self.assertEqual(self.service.parse_unique_code("SOME TEXT WITH - DASH - CODE123"), "CODE123")
        self.assertEqual(self.service.parse_unique_code(""), "")
        self.assertEqual(self.service.parse_unique_code("NO DASH HERE"), "NO DASH HERE")

    def test_ocr_with_flip_fallback_prefers_flipped_expected_code(self) -> None:
        vision = VisionConfig()

        def fake_ocr(image, vision, *, expected_text, regex, canonical_map):
            if int(image[0, 0]) == 7:
                return {
                    "status": "ok",
                    "engine": "tesseract",
                    "text": "MODEL NAME - ADV160A",
                    "raw_text": "MODEL NAME - ADV160A",
                    "canonical_text": "MODEL NAME - ADV160A",
                    "confidence": 0.91,
                    "expected_text": expected_text,
                    "match_expected": False,
                    "error": None,
                }
            return {
                "status": "ok",
                "engine": "tesseract",
                "text": "noise",
                "raw_text": "noise",
                "canonical_text": "noise",
                "confidence": 0.20,
                "expected_text": expected_text,
                "match_expected": False,
                "error": None,
            }

        image = np.array([[0, 0], [0, 7]], dtype=np.uint8)
        with mock.patch.object(self.service, "_ocr_with_tesseract", side_effect=fake_ocr):
            result = self.service._ocr_with_flip_fallback(
                image,
                vision,
                expected_text="ADV160A",
                regex=None,
                canonical_map={},
            )

        self.assertTrue(result["was_flipped"])
        self.assertTrue(result["match_expected"])
        self.assertEqual(result["canonical_text"], "MODEL NAME - ADV160A")

    # NOTE (FASE 0): OCR-based sticker validation was removed by design. The two
    # tests that exercised _augment_with_anchor_ocr / _augment_with_ocr_only (using
    # the removed ocr_engine / use_ocr / ocr_expected_code / expected_dot_x/y fields)
    # were retired. The OCR text-normalization utility tests above
    # (_normalize_ocr_text / _parse_unique_code / flip-fallback) are kept because
    # they test helpers that still exist. See TESTING.md + HANDOFF.md R5.


if __name__ == "__main__":
    unittest.main()
