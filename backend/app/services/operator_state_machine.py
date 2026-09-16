from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class OperatorRuntimeState(StrEnum):
    IDLE = "IDLE"
    INSPECTION = "INSPECTION"
    RESULT = "RESULT"


@dataclass(frozen=True, slots=True)
class OperatorStateDecision:
    state: OperatorRuntimeState
    run_inspection: bool
    use_cached_result: bool
    reset_event: bool


class OperatorInspectionStateMachine:
    """Per-session state policy for production inspection.

    Simple mode: no cache, no hysteresis.
    - Part ready + settled → always run inference
    - Part leave → reset to IDLE
    """

    SETTLE_MIN_FRAMES = 0

    def update(
        self,
        session_state,
        *,
        part_ready: bool,
        present: bool,
        settled: bool,
    ) -> OperatorStateDecision:
        # ── Part leave or not present → reset to IDLE ──
        if not part_ready or not present:
            session_state.operator_state = OperatorRuntimeState.IDLE.value
            session_state.inspection_result_cache = None
            session_state.inspection_has_run_for_current_part = False
            session_state.part_removed_seen_at = datetime.now(UTC)
            session_state.settle_frame_count = 0
            session_state.last_inference_ms = 0
            session_state.consecutive_reject_count = 0
            return OperatorStateDecision(
                state=OperatorRuntimeState.IDLE,
                run_inspection=False,
                use_cached_result=False,
                reset_event=True,
            )

        # ── Settled → always run inference (no cache) ──
        if settled:
            session_state.settle_frame_count = int(getattr(session_state, "settle_frame_count", 0)) + 1
            session_state.last_inference_ms = int(time.time() * 1000)
            session_state.operator_state = OperatorRuntimeState.INSPECTION.value
            session_state.inspection_has_run_for_current_part = True
            return OperatorStateDecision(
                state=OperatorRuntimeState.INSPECTION,
                run_inspection=True,
                use_cached_result=False,
                reset_event=False,
            )

        # ── Not settled → stay IDLE ──
        session_state.settle_frame_count = 0
        session_state.operator_state = OperatorRuntimeState.IDLE.value
        return OperatorStateDecision(
            state=OperatorRuntimeState.IDLE,
            run_inspection=False,
            use_cached_result=False,
            reset_event=False,
        )

    def mark_result(self, session_state, payload: dict) -> None:
        session_state.operator_state = OperatorRuntimeState.RESULT.value
        session_state.inspection_result_cache = dict(payload)
