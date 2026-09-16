from __future__ import annotations

import unittest

from backend.app.workers.plc_worker import PlcWorker


class _FakeAdapter:
    def __init__(self, inputs: list[bool] | None = None) -> None:
        self.inputs = inputs or [False] * 8
        self.writes: list[tuple[int, bool]] = []
        self.all_off_count = 0

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def read_inputs(self, address: int = 0, count: int = 8):
        return list(self.inputs[:count])

    def write_coil(self, address: int, value: bool) -> None:
        self.writes.append((address, bool(value)))

    def all_off(self, num_channels: int) -> None:
        self.all_off_count += 1

    def is_connected(self) -> bool:
        return True

    def status(self) -> dict:
        return {}


class PlcWorkerFeedbackTest(unittest.TestCase):
    def test_input3_clamp_feedback_marks_worker_clamped(self) -> None:
        adapter = _FakeAdapter([False, False, True, False, False, False, False, False])
        worker = PlcWorker(
            adapter,
            clamp_feedback_enabled=True,
            input_clamp_engaged_address=2,
        )

        worker._cmd_part_ready({"event_id": "evt-1"})  # noqa: SLF001
        worker._poll_inputs()  # noqa: SLF001

        status = worker.status()
        self.assertEqual(status["state"], "CLAMPED")
        self.assertTrue(status["clamp_engaged"])
        # Default relay map: clamp=3, ok_light_buzzer=2, enji_buzzer=1 (lihat config.py CH1/CH2/CH3)
        self.assertEqual(adapter.writes[:3], [(3, True), (2, False), (1, False)])

    def test_decision_is_allowed_after_clamped_feedback(self) -> None:
        adapter = _FakeAdapter([False, False, True, False, False, False, False, False])
        worker = PlcWorker(
            adapter,
            clamp_feedback_enabled=True,
            input_clamp_engaged_address=2,
        )
        worker._cmd_part_ready({"event_id": "evt-1"})  # noqa: SLF001
        worker._poll_inputs()  # noqa: SLF001

        worker._cmd_decision({"decision": "ACCEPT", "event_id": "evt-1"})  # noqa: SLF001

        self.assertEqual(worker.status()["state"], "ACCEPT_PULSE")


if __name__ == "__main__":
    unittest.main()
