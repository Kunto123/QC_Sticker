"""PLC Flow Strategy pattern.

Each strategy encapsulates:
  - How to interpret inputs (release, template cycle, clamp feedback)
  - How to map inspection events → coil actions (accept/reject/part_ready)
  - What timing to use (pulse duration, debounce, etc.)

The PlcWorker delegates all mode-specific behavior to the active strategy.
The strategy reads I/O addresses/timing from MachineSettings, NOT from env vars.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.services.plc_adapter import PlcAdapter

logger = logging.getLogger(__name__)


class PlcFlowStrategy(ABC):
    """Abstract base for PLC flow strategies.

    Each strategy owns the full lifecycle:
      part_ready → clamp → inspection → accept/reject → release
    """

    def __init__(
        self,
        adapter: PlcAdapter,
        settings,  # StickerModeConfig or CounterModeConfig
        num_channels: int = 4,
    ):
        self._adapter = adapter
        self._settings = settings
        self._num_channels = num_channels

    # ── Subclass must implement ──────────────────────────────────────

    @abstractmethod
    def on_part_ready(self, worker) -> None:
        """Engage clamp when part is ready. State → CLAMPING."""
        ...

    @abstractmethod
    def on_accept(self, worker) -> None:
        """Accept: release clamp + pulse OK signal. State → ACCEPT_PULSE."""
        ...

    @abstractmethod
    def on_reject(self, worker) -> None:
        """Reject: engage reject buzzer + clamp stays. State → REJECT_BUZZER."""
        ...

    @abstractmethod
    def handle_input_release(self, worker, inputs: list[bool]) -> bool:
        """Return True if release was triggered (caller should all_off)."""
        ...

    @abstractmethod
    def handle_input_template_cycle(self, worker, inputs: list[bool]) -> bool:
        """Return True if template cycle was triggered."""
        ...

    @abstractmethod
    def handle_clamp_feedback(self, worker, inputs: list[bool]) -> None:
        """Transition CLAMPING → CLAMPED when feedback confirms."""
        ...

    @abstractmethod
    def get_input_release_address(self) -> int:
        ...

    @abstractmethod
    def get_input_template_address(self) -> int:
        ...

    @abstractmethod
    def get_input_clamp_engaged_address(self) -> int:
        ...

    # ── Shared helpers ───────────────────────────────────────────────

    def all_off(self, worker, reason: str) -> None:
        logger.info("[%s] ALL OFF — %s", self.flow_name, reason)
        for i in range(self._num_channels):
            try:
                self._adapter.write_coil(i, False)
            except Exception as exc:
                logger.error("[%s] coil[%d] OFF failed: %s", self.flow_name, i, exc)

    def write_coil(self, worker, addr: int, value: bool) -> None:
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                self._adapter.write_coil(addr, value)
                return
            except Exception as exc:
                logger.error(
                    "[%s] write_coil addr=%d attempt %d failed: %s",
                    self.flow_name, addr, attempt, exc,
                )
                if attempt < max_retries:
                    time.sleep(0.1 * attempt)
        raise RuntimeError(
            f"write_coil addr={addr} failed after {max_retries} attempts"
        )

    @property
    @abstractmethod
    def flow_name(self) -> str:
        ...
