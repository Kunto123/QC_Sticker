"""Accept-streak counting (`InspectionSessionService._update_accept_generation_count`).

`accept_stable_frames` = N consecutive *fresh* ACCEPT inference results. Before
2026-09-18 the count was also reset whenever more than `accept_stable_ms × 3` had
passed since the first counted result — with `accept_stable_ms=100` and inference
every ~250 ms (template `inference_fps=4`) three results could never fit in the
300 ms window, so the count cycled 1, 2, 1, 2 … and nothing ever committed
("bbox stabil tapi tidak bertambah apa-apa"). HANDOFF §9e.
"""
from __future__ import annotations

import types
import unittest

from backend.app.services.inspection_session import InspectionSessionService


def _service(*, frames: int = 3, stable_ms: int = 100) -> InspectionSessionService:
    service = InspectionSessionService.__new__(InspectionSessionService)
    service._accept_stable_frames = frames
    service._accept_stable_ms = stable_ms
    return service


def _state() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        inference_result_generation=0,
        inference_accept_count=0,
        inference_accept_first_ts=0.0,
        inference_last_counted_generation=-1,
    )


class AcceptGenerationCountTest(unittest.TestCase):
    def _accept(self, service, state, generation: int, now_s: float) -> bool:
        state.inference_result_generation = generation
        return service._update_accept_generation_count(
            state, effective_is_accept=True, is_non_hard_reject=False, now_s=now_s,
        )

    def test_three_consecutive_results_count_regardless_of_cadence(self) -> None:
        """The user's setting: frames=3, stable_ms=100, one inference every 250 ms."""
        service, state = _service(frames=3, stable_ms=100), _state()
        self.assertFalse(self._accept(service, state, 1, 10.00))
        self.assertFalse(self._accept(service, state, 2, 10.25))
        self.assertFalse(self._accept(service, state, 3, 10.50))
        self.assertEqual(state.inference_accept_count, 3)
        self.assertGreaterEqual(state.inference_accept_count, service._accept_stable_frames)

    def test_slow_inference_still_reaches_threshold(self) -> None:
        service, state = _service(frames=3, stable_ms=200), _state()
        for gen, t in ((1, 100.0), (2, 102.0), (3, 104.0)):
            self._accept(service, state, gen, t)
        self.assertEqual(state.inference_accept_count, 3)

    def test_same_generation_is_not_double_counted(self) -> None:
        service, state = _service(), _state()
        self._accept(service, state, 1, 100.0)
        self._accept(service, state, 1, 100.1)
        self._accept(service, state, 1, 100.2)
        self.assertEqual(state.inference_accept_count, 1)

    def test_skipped_generation_restarts_streak(self) -> None:
        """A generation read by a non-accept frame (sticker lost past the holdover) breaks the streak."""
        service, state = _service(), _state()
        self._accept(service, state, 1, 100.0)
        self._accept(service, state, 2, 100.25)
        # generation 3 arrives while the frame is NOT_FOUND outside holdover → not counted
        state.inference_result_generation = 3
        service._update_accept_generation_count(
            state, effective_is_accept=False, is_non_hard_reject=True, now_s=100.5,
        )
        self.assertEqual(state.inference_accept_count, 2)  # non-hard reject leaves counters alone
        restarted = self._accept(service, state, 4, 100.75)
        self.assertTrue(restarted)
        self.assertEqual(state.inference_accept_count, 1)
        self.assertEqual(state.inference_accept_first_ts, 100.75)
        self.assertFalse(self._accept(service, state, 5, 101.0))
        self.assertEqual(state.inference_accept_count, 2)

    def test_hard_reject_resets_to_zero(self) -> None:
        service, state = _service(), _state()
        self._accept(service, state, 1, 100.0)
        self._accept(service, state, 2, 100.25)
        state.inference_result_generation = 3
        self.assertFalse(service._update_accept_generation_count(
            state, effective_is_accept=False, is_non_hard_reject=False, now_s=100.5,
        ))
        self.assertEqual(state.inference_accept_count, 0)
        self.assertEqual(state.inference_last_counted_generation, -1)
        # first accept after the reset starts a fresh streak without being "broken"
        self.assertFalse(self._accept(service, state, 4, 100.75))
        self.assertEqual(state.inference_accept_count, 1)

    def test_holdover_frames_keep_counting(self) -> None:
        """Inside the holdover the caller passes effective_is_accept=True for a NOT_FOUND
        frame; that generation counts and does not break the streak."""
        service, state = _service(), _state()
        self._accept(service, state, 1, 100.0)
        self._accept(service, state, 2, 100.25)  # NOT_FOUND but within holdover → effective accept
        self._accept(service, state, 3, 100.50)
        self.assertEqual(state.inference_accept_count, 3)


if __name__ == "__main__":
    unittest.main()
