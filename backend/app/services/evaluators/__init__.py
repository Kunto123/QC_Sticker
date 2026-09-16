from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shared.contracts.decision import Decision
from backend.app.models.session_state import SessionState


@dataclass(slots=True)
class EvalContext:
    """Input context passed to every ModeEvaluator.evaluate() call.

    Contains everything an evaluator needs to make a per-frame decision,
    abstracted from the HTTP/WebSocket transport layer.
    """
    detections: list[dict[str, Any]]
    frame: np.ndarray
    criteria: dict[str, Any]  # mode-specific config from template
    state: SessionState
    roi_frame: np.ndarray | None = None
    part_ready_payload: dict[str, Any] | None = None
    detection_payload: dict[str, Any] | None = None
    additional: dict[str, Any] = field(default_factory=dict)  # catch-all for extras


class ModeEvaluator(ABC):
    """Abstract evaluator — one subclass per validator mode.

    Each evaluator is stateless (all state lives in SessionState).
    Thread-safe: called from background inference thread.
    """

    mode_name: str  # set by subclass, e.g. "sticker", "counter", "defect"

    @abstractmethod
    def evaluate(self, ctx: EvalContext) -> Decision:
        """Evaluate one frame and return a Decision.

        *Never* returns raw model output.  All downstream code reads Decision.
        """
        ...
