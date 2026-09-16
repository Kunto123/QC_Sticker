from __future__ import annotations

import types
import unittest

from backend.app.services.inspection_session import InspectionSessionService


class OperatorPhaseDelayTest(unittest.TestCase):
    def _service(self) -> InspectionSessionService:
        service = InspectionSessionService.__new__(InspectionSessionService)
        service._phase_sticker_install_delay_ms = 1000
        service._phase_next_part_delay_ms = 2000
        service._plc_worker = object()
        service._plc_clamp_feedback_enabled = False
        service._plc_clamp_feedback_timeout_ms = 1500
        service._plc_clamp_feedback_fallback_delay_ms = 300
        return service

    def test_sticker_install_delay_gates_inference_until_elapsed(self) -> None:
        service = self._service()
        state = types.SimpleNamespace(
            operator_sticker_delay_started_at=0.0,
            operator_sticker_ready_at=0.0,
        )

        ready, payload = service._operator_phase_status(  # noqa: SLF001
            state,
            raw_part_ready=True,
            part_ready_settled=True,
            clamp_ready=True,
            now_s=100.0,
        )
        self.assertFalse(ready)
        self.assertEqual(payload["status"], "sticker_install_delay")

        ready, payload = service._operator_phase_status(  # noqa: SLF001
            state,
            raw_part_ready=True,
            part_ready_settled=True,
            clamp_ready=True,
            now_s=101.1,
        )
        self.assertTrue(ready)
        self.assertEqual(payload["status"], "ready")

    def test_next_part_delay_blocks_clamp_gate(self) -> None:
        service = self._service()
        state = types.SimpleNamespace(
            manual_release_cooldown_until=102.0,
            plc_clamp_requested_at=100.0,
            plc_clamp_ready_at=0.0,
            plc_clamp_timeout=False,
        )

        ready, payload = service._clamp_gate_status(  # noqa: SLF001
            state,
            part_ready_settled=True,
            now_s=100.5,
        )
        self.assertFalse(ready)
        self.assertEqual(payload["status"], "next_part_delay")
        self.assertGreater(payload["remaining_ms"], 0)


if __name__ == "__main__":
    unittest.main()
