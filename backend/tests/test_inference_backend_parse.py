"""inference_backend.py — raw YOLO output parsing shared by the OpenVINO / ONNX / TFLite backends.

Ultralytics exports differ in box units: OpenVINO / ONNX emit cx,cy,w,h in input
pixels (0..imgsz), TFLite emits 0..1. Before 2026-09-18 the parser assumed 0..1 and
multiplied pixel boxes by imgsz again, so an imported OpenVINO model detected fine
but every bbox landed off-screen ("raw detection 8400, nothing on screen").
"""
from __future__ import annotations

import threading
import unittest
from unittest import mock

import numpy as np

from backend.app.core.config import AppConfig
from backend.app.services.inference_backend import ONNXBackend, _apply_nms, _parse_yolo_output


def _rows(boxes: list[tuple[float, float, float, float]], scores: list[list[float]]) -> np.ndarray:
    """Build an [N, 4+nc] tensor (YOLOv8/11 layout), padded with zero-score anchor rows so
    N > C like a real head (the layout auto-detect relies on rows outnumbering channels)."""
    rows = [[*b, *s] for b, s in zip(boxes, scores)]
    filler = [0.0] * len(rows[0])
    rows += [filler] * max(0, 64 - len(rows))
    return np.array(rows, dtype=np.float32)


class ParseYoloOutputTest(unittest.TestCase):
    def test_pixel_boxes_openvino_style_are_normalized_to_content(self) -> None:
        # 640x640 input, no padding; box centred at (320, 320), 160x80 px, class 1 @ 0.9
        out = _rows([(320.0, 320.0, 160.0, 80.0)], [[0.1, 0.9]])
        cands = _parse_yolo_output(out, 0.25, 0, 0, 640, 640)
        self.assertEqual(len(cands), 1)
        conf, cls, x1, y1, x2, y2 = cands[0]
        self.assertAlmostEqual(conf, 0.9, places=5)
        self.assertEqual(cls, 1)
        self.assertAlmostEqual(x1, 0.375, places=4)  # (320-80)/640
        self.assertAlmostEqual(y1, 0.4375, places=4)  # (320-40)/640
        self.assertAlmostEqual(x2, 0.625, places=4)
        self.assertAlmostEqual(y2, 0.5625, places=4)

    def test_normalized_boxes_tflite_style_give_the_same_result(self) -> None:
        pixel = _rows([(320.0, 320.0, 160.0, 80.0)], [[0.1, 0.9]])
        normalized = pixel.copy()
        normalized[:, :4] /= 640.0
        a = _parse_yolo_output(pixel, 0.25, 0, 0, 640, 640)
        b = _parse_yolo_output(normalized, 0.25, 0, 0, 640, 640)
        self.assertEqual(len(a), 1)
        for va, vb in zip(a[0], b[0]):
            self.assertAlmostEqual(va, vb, places=4)

    def test_channel_first_layout_is_transposed(self) -> None:
        out = _rows([(320.0, 320.0, 160.0, 80.0), (100.0, 100.0, 20.0, 20.0)], [[0.9, 0.0], [0.0, 0.3]])
        cands = _parse_yolo_output(out.T.copy(), 0.25, 0, 0, 640, 640)  # [C, N]
        self.assertEqual([c[1] for c in cands], [0, 1])

    def test_letterbox_padding_is_removed(self) -> None:
        # portrait image letterboxed into 640x640 → 100 px pad left/right; content 440 wide
        out = _rows([(320.0, 320.0, 220.0, 320.0)], [[0.8]])
        (conf, cls, x1, y1, x2, y2), = _parse_yolo_output(out, 0.25, 100, 0, 640, 640)
        self.assertAlmostEqual(x1, (210 - 100) / 440, places=4)
        self.assertAlmostEqual(x2, (430 - 100) / 440, places=4)
        self.assertAlmostEqual(y1, 160 / 640, places=4)
        self.assertAlmostEqual(y2, 480 / 640, places=4)

    def test_threshold_and_degenerate_boxes(self) -> None:
        out = _rows(
            [(320.0, 320.0, 160.0, 80.0), (320.0, 320.0, 160.0, 80.0), (-500.0, 320.0, 10.0, 10.0)],
            [[0.2, 0.1], [0.5, 0.1], [0.9, 0.0]],
        )
        cands = _parse_yolo_output(out, 0.25, 0, 0, 640, 640)
        # row 0 below threshold, row 2 entirely outside the image → dropped
        self.assertEqual(len(cands), 1)
        self.assertAlmostEqual(cands[0][0], 0.5, places=5)

    def test_objectness_layout_multiplies_scores(self) -> None:
        out = np.array([[320.0, 320.0, 160.0, 80.0, 0.5, 0.2, 0.8]] + [[0.0] * 7] * 63, dtype=np.float32)
        cands = _parse_yolo_output(out, 0.25, 0, 0, 640, 640, has_objectness=True)
        self.assertEqual(len(cands), 1)
        self.assertAlmostEqual(cands[0][0], 0.4, places=5)
        self.assertEqual(cands[0][1], 1)

    def test_empty_or_malformed_input(self) -> None:
        self.assertEqual(_parse_yolo_output(None, 0.25, 0, 0, 640, 640), [])
        self.assertEqual(_parse_yolo_output(np.zeros((0, 6), np.float32), 0.25, 0, 0, 640, 640), [])
        self.assertEqual(_parse_yolo_output(np.zeros((6,), np.float32), 0.25, 0, 0, 640, 640), [])
        self.assertEqual(_parse_yolo_output(np.zeros((8400, 4), np.float32), 0.25, 0, 0, 640, 640), [])

    def test_nms_collapses_overlapping_anchor_rows(self) -> None:
        """8400-row heads fire on many neighbouring anchors for one object; NMS must keep one."""
        boxes = [(320.0 + dx, 320.0, 160.0, 80.0) for dx in (-2.0, -1.0, 0.0, 1.0, 2.0)]
        out = _rows(boxes, [[0.7, 0.0], [0.8, 0.0], [0.9, 0.0], [0.85, 0.0], [0.75, 0.0]])
        cands = _parse_yolo_output(out, 0.25, 0, 0, 640, 640)
        self.assertEqual(len(cands), 5)
        dets = _apply_nms(cands, 0.25)
        self.assertEqual(len(dets), 1)
        self.assertAlmostEqual(dets[0]["confidence"], 0.9, places=4)


class OnnxInputLayoutTest(unittest.TestCase):
    """The ONNX backend must feed NCHW to an Ultralytics export and NHWC to a TFLite-origin one."""

    def _run(self, shape: list, tmp_model: str) -> tuple[np.ndarray, dict]:
        backend = ONNXBackend(config=AppConfig(), loaded_models={}, meta_cache={}, runtime_lock=threading.RLock())
        seen: dict = {}

        class _Input:
            name = "images"

            def __init__(self, shp):
                self.shape = shp

        class _Sess:
            def get_inputs(self):
                return [_Input(shape)]

            def run(self, _outputs, feeds):
                seen["input"] = feeds["images"]
                # one pixel-unit box, class 0 @ 0.9 plus zero anchors — [1, 5, 64] channel-first like the real head
                head = np.zeros((1, 5, 64), dtype=np.float32)
                head[0, :, 0] = [300.0, 300.0, 100.0, 100.0, 0.9]
                return [head]

        vision = mock.Mock()
        vision.model_path = tmp_model
        vision.model_meta_path = None
        vision.conf_threshold = 0.25
        vision.classes = []
        with mock.patch.object(backend, "_load_onnx_session", return_value=_Sess()), \
                mock.patch.object(backend, "_load_meta", return_value={"class_names": ["sticker"]}):
            result = backend.predict(np.zeros((600, 600, 3), np.uint8), vision, expected_class="sticker")
        return seen["input"], result

    def test_nchw_export(self) -> None:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as fh:
            path = fh.name
        fed, result = self._run([1, 3, 640, 640], path)
        self.assertEqual(fed.shape, (1, 3, 640, 640))
        self.assertEqual(len(result["detections"]), 1)
        self.assertEqual(result["detections"][0]["label"], "sticker")
        # 600x600 image letterboxed to 640: box (300±50 px) → (250..350)/640 * 600
        x1, y1, x2, y2 = result["detections"][0]["bbox"]
        self.assertAlmostEqual(x1, 250 / 640 * 600, places=1)
        self.assertAlmostEqual(x2, 350 / 640 * 600, places=1)

    def test_nhwc_export_and_dynamic_dims(self) -> None:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as fh:
            path = fh.name
        fed, _ = self._run(["batch", 640, 640, 3], path)
        self.assertEqual(fed.shape, (1, 640, 640, 3))
        fed, _ = self._run(["batch", 3, "height", "width"], path)
        self.assertEqual(fed.shape, (1, 3, 640, 640))


if __name__ == "__main__":
    unittest.main()
