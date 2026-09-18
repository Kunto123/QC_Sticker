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

    # NOTE: OCR-based sticker validation was removed by design (FASE 0); the OCR
    # helper methods and their tests were deleted 2026-09-18. See TESTING.md.


class UnloadModelTest(unittest.TestCase):
    """unload_model must evict every cache entry for a model file (plain key and the
    `::tN` tflite variant) plus the meta cache of its folder, so purge can delete files."""

    def test_unload_evicts_model_and_meta_entries(self) -> None:
        service = StickerInferenceService(AppConfig(), mock.Mock())
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "m"
            folder.mkdir()
            model = folder / "best.xml"
            model.write_bytes(b"x")
            other = Path(tmp) / "other.onnx"
            other.write_bytes(b"y")
            resolved = str(model.resolve())
            service._loaded_models[resolved] = object()
            service._loaded_models[f"{resolved}::t4"] = object()
            service._loaded_models[str(other.resolve())] = object()
            service._meta_cache[str(folder / "best.meta.json")] = ({}, 0.0)
            service._meta_cache[str(Path(tmp) / "other.meta.json")] = ({}, 0.0)

            self.assertEqual(service.unload_model(str(model)), 2)

            self.assertEqual(list(service._loaded_models), [str(other.resolve())])
            self.assertEqual(list(service._meta_cache), [str(Path(tmp) / "other.meta.json")])
            # unknown path is a no-op
            self.assertEqual(service.unload_model(str(Path(tmp) / "nope.pt")), 0)


if __name__ == "__main__":
    unittest.main()
