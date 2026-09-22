"""Manual Canny threshold override for gap_template_match part-ready detection.

Covers: _auto_canny falls back to median-based auto-tuning when low/high are
not both given (legacy behaviour, unchanged), and uses the explicit bounds
verbatim when they are. Also covers gap_search_margin — the fix for
"part shifted a few px against a fixed-position reference always fails"
(search area used to equal the reference patch size exactly, giving
cv2.matchTemplate zero room to search for a shifted alignment).
"""
from __future__ import annotations

import os
import tempfile
import unittest

import cv2
import numpy as np

from backend.app.services.gap_detector import _auto_canny, match_gap, save_ref_patch


def _checkerboard(size: int = 64) -> np.ndarray:
    gray = np.zeros((size, size), dtype=np.uint8)
    gray[::8, :] = 255
    gray[:, ::8] = 255
    return gray


class AutoCannyTest(unittest.TestCase):
    def test_explicit_bounds_are_used_verbatim(self) -> None:
        gray = _checkerboard()
        low_result = _auto_canny(gray, low=10, high=20)
        high_result = _auto_canny(gray, low=200, high=250)
        # A near-white threshold band should detect far fewer/equal edge pixels
        # than a near-black one on the same image.
        self.assertLessEqual(int(np.count_nonzero(high_result)), int(np.count_nonzero(low_result)))

    def test_only_one_bound_given_falls_back_to_auto(self) -> None:
        gray = _checkerboard()
        auto_result = _auto_canny(gray)
        partial_result = _auto_canny(gray, low=10, high=None)
        self.assertTrue(np.array_equal(auto_result, partial_result))

    def test_omitting_bounds_preserves_legacy_auto_tuning(self) -> None:
        gray = _checkerboard()
        self.assertTrue(np.array_equal(_auto_canny(gray), _auto_canny(gray, low=None, high=None)))


class GapMatchCannyThreadingTest(unittest.TestCase):
    def test_save_and_match_with_fixed_bounds_round_trips(self) -> None:
        frame = cv2.cvtColor(_checkerboard(128), cv2.COLOR_GRAY2BGR)
        roi = {"x": 0, "y": 0, "w": 128, "h": 128}
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            ok, err = save_ref_patch(frame, roi, path, canny_low=30, canny_high=90)
            self.assertTrue(ok, err)
            ref_patch = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(ref_patch)
            # Matching the same frame with the SAME fixed bounds must score high.
            result = match_gap(frame, roi, ref_patch, threshold=0.5, canny_low=30, canny_high=90)
            self.assertTrue(result["match"], result)
        finally:
            os.unlink(path)


class ExpandRoiForGapSearchTest(unittest.TestCase):
    """_expand_roi_for_gap_search — pure geometry, no session/mocks needed."""

    def setUp(self) -> None:
        from unittest.mock import MagicMock
        from backend.app.services.inspection_session import InspectionSessionService
        from shared.contracts.templates import RoiGeometry

        self.service = InspectionSessionService(
            template_runtime=MagicMock(), results_repo=MagicMock(), sticker_inference=MagicMock(),
        )
        self.roi = RoiGeometry(x=0.35, y=0.35, w=0.3, h=0.3)  # 70,70,60,60 px in a 200x200 frame
        self.frame = np.zeros((200, 200, 3), dtype=np.uint8)

    def test_zero_margin_matches_plain_crop(self) -> None:
        expanded = self.service._expand_roi_for_gap_search(self.frame, self.roi, {}, 0.0)
        cropped, _meta = self.service._crop_stage_roi(self.frame, self.roi, {})
        self.assertEqual(expanded.shape, cropped.shape)

    def test_positive_margin_grows_each_side(self) -> None:
        expanded = self.service._expand_roi_for_gap_search(self.frame, self.roi, {}, 0.5)
        # w/h grown by 0.5 * roi_w/h on each side -> 60 * (1 + 2*0.5) = 120 px
        self.assertEqual(expanded.shape[:2], (120, 120))

    def test_margin_clamps_at_frame_bounds(self) -> None:
        from shared.contracts.templates import RoiGeometry as _RoiGeometry
        near_edge_roi = _RoiGeometry(x=0.0, y=0.0, w=0.1, h=0.1)  # 0,0,20,20 px
        expanded = self.service._expand_roi_for_gap_search(self.frame, near_edge_roi, {}, 5.0)
        h, w = expanded.shape[:2]
        self.assertLessEqual(h, 200)
        self.assertLessEqual(w, 200)


class GapSearchMarginPositionToleranceTest(unittest.TestCase):
    """End-to-end: a part shifted a few px against a fixed part_ready_roi must
    fail with gap_search_margin=0 (legacy, zero search room) and pass once a
    margin large enough to cover the shift is configured — without touching
    the calibrated reference patch at all."""

    def _make_state(self, gap_ref_path: str, gap_search_margin: float, threshold: float = 0.5):
        from backend.app.models.session_state import SessionState
        from shared.contracts.enums import SessionStatus
        from shared.contracts.templates import (
            CameraDefaults, InspectionTemplate, PartReadyConfig,
            PersistenceConfig, RoiGeometry, StickerRule, VisionConfig,
        )

        part_ready_roi = RoiGeometry(x=0.35, y=0.35, w=0.3, h=0.3)  # 70,70,60,60 px in 200x200
        template = InspectionTemplate(
            id=1, version_id=1, version_number=1, name="T", description="", is_active=True,
            camera=CameraDefaults(),
            part_ready_roi=part_ready_roi,
            sticker_roi=RoiGeometry(),
            vision=VisionConfig(),
            part_ready=PartReadyConfig(
                gap_ref_path=gap_ref_path,
                gap_match_threshold=threshold,
                canny_low=30, canny_high=90,
                gap_search_margin=gap_search_margin,
            ),
            sticker=StickerRule(part_name="P1", expected_class="sticker", enabled=True),
            persistence=PersistenceConfig(),
        )
        return SessionState(
            session_id="s1", client_id="c1", camera_index=0, template=template,
            status=SessionStatus.RUNNING,
        )

    def setUp(self) -> None:
        from unittest.mock import MagicMock
        from backend.app.services.inspection_session import InspectionSessionService

        self.service = InspectionSessionService(
            template_runtime=MagicMock(), results_repo=MagicMock(), sticker_inference=MagicMock(),
        )
        # Calibration frame: 30x30 white square centered at (100,100) on a flat
        # gray background — captured via the same ROI used at runtime (70,70,60,60).
        calib_frame = np.full((200, 200, 3), 60, dtype=np.uint8)
        cv2.rectangle(calib_frame, (85, 85), (115, 115), (255, 255, 255), -1)
        roi_px = {"x": 70, "y": 70, "w": 60, "h": 60}
        fd, self.ref_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        ok, err = save_ref_patch(calib_frame, roi_px, self.ref_path, canny_low=30, canny_high=90)
        self.assertTrue(ok, err)

        # Live frame: same square shifted 25px right — part_ready_roi itself
        # (the fixed camera FOV crop) does NOT move, only the physical part does.
        self.shifted_frame = np.full((200, 200, 3), 60, dtype=np.uint8)
        cv2.rectangle(self.shifted_frame, (110, 85), (140, 115), (255, 255, 255), -1)

    def tearDown(self) -> None:
        os.unlink(self.ref_path)

    def _run_gap_eval(self, gap_search_margin: float) -> dict:
        state = self._make_state(self.ref_path, gap_search_margin)
        part_ready_frame, _meta = self.service._crop_stage_roi(
            self.shifted_frame, state.template.part_ready_roi, {},
        )
        return self.service._evaluate_part_ready(part_ready_frame, state, raw_frame=self.shifted_frame)

    def test_zero_margin_rejects_shifted_part(self) -> None:
        result = self._run_gap_eval(0.0)
        self.assertFalse(result["part_ready"], result)

    def test_sufficient_margin_accepts_shifted_part(self) -> None:
        result = self._run_gap_eval(0.6)
        self.assertTrue(result["part_ready"], result)


if __name__ == "__main__":
    unittest.main()
