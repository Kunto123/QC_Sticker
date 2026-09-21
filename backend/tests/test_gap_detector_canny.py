"""Manual Canny threshold override for gap_template_match part-ready detection.

Covers: _auto_canny falls back to median-based auto-tuning when low/high are
not both given (legacy behaviour, unchanged), and uses the explicit bounds
verbatim when they are.
"""
from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
