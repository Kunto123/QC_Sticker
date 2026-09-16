"""StickerFlow — PLC flow strategy for QC Sticker inspection mode.

This is the extracted behavior from the original PlcWorker, now reading
all addresses/timing from MachineSettings.sticker instead of constructor args.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from backend.app.services.plc_flow_strategy import PlcFlowStrategy

if TYPE_CHECKING:
    from backend.app.services.plc_adapter import PlcAdapter
    from backend.app.models.machine_settings import StickerModeConfig

logger = logging.getLogger(__name__)


class StickerFlow(PlcFlowStrategy):
    """QC Sticker mode: part_ready → clamp → inspect → accept/reject → release."""

    def __init__(
        self,
        adapter: PlcAdapter,
        settings: StickerModeConfig,
        num_channels: int = 4,
    ):
        super().__init__(adapter, settings, num_channels)
        self._sticker_settings = settings
        self._accept_pulse_end: float | None = None

    @property
    def flow_name(self) -> str:
        return "sticker-flow"

    # ── Address accessors ────────────────────────────────────────────

    def get_input_release_address(self) -> int:
        return self._sticker_settings.input_release_address

    def get_input_template_address(self) -> int:
        return self._sticker_settings.input_template_address

    def get_input_clamp_engaged_address(self) -> int:
        return self._sticker_settings.input_clamp_engaged_address

    # ── Event handlers ──────────────────────────────────────────────

    def on_part_ready(self, worker) -> None:
        """Engage clamp (CH3=ON), turn off OK light + enji buzzer."""
        s = self._sticker_settings
        self.write_coil(worker, s.relay_clamp_address, True)
        self.write_coil(worker, s.relay_ok_light_buzzer_address, False)
        self.write_coil(worker, s.relay_enji_buzzer_address, False)
        logger.info(
            "[sticker-flow] CLAMPING — clamp_addr=%d",
            s.relay_clamp_address,
        )

    def on_accept(self, worker) -> None:
        """ACCEPT: release clamp, pulse OK light+buzzer, → IDLE."""
        s = self._sticker_settings
        self.write_coil(worker, s.relay_clamp_address, False)
        self.write_coil(worker, s.relay_ok_light_buzzer_address, True)
        self._accept_pulse_end = time.time() + (s.accept_pulse_ms / 1000.0)
        logger.info(
            "[sticker-flow] ACCEPT — pulse %dms (addr=%d)",
            s.accept_pulse_ms, s.relay_ok_light_buzzer_address,
        )

    def on_reject(self, worker) -> None:
        """REJECT: enji buzzer ON, clamp stays, wait for manual release."""
        s = self._sticker_settings
        self.write_coil(worker, s.relay_enji_buzzer_address, True)
        self.write_coil(worker, s.relay_clamp_address, True)
        logger.info(
            "[sticker-flow] REJECT — enji buzzer ON (addr=%d)",
            s.relay_enji_buzzer_address,
        )

    def finish_accept_pulse(self, worker) -> None:
        """Called by worker when accept pulse timer expires."""
        s = self._sticker_settings
        self.write_coil(worker, s.relay_ok_light_buzzer_address, False)
        self._accept_pulse_end = None
        logger.info("[sticker-flow] ACCEPT done → IDLE")

    def is_accept_pulse_complete(self) -> bool:
        return (
            self._accept_pulse_end is not None
            and time.time() >= self._accept_pulse_end
        )

    # ── Input handlers ──────────────────────────────────────────────

    def handle_input_release(self, worker, inputs: list[bool]) -> bool:
        """Manual release on IN1. Returns True if triggered."""
        addr = self._sticker_settings.input_release_address
        if addr < len(inputs) and inputs[addr]:
            logger.info("[sticker-flow] INPUT release (addr=%d) — Manual Release", addr)
            return True
        return False

    def handle_input_template_cycle(self, worker, inputs: list[bool]) -> bool:
        """Template cycle on IN2. Returns True if triggered."""
        addr = self._sticker_settings.input_template_address
        if addr < len(inputs) and inputs[addr]:
            logger.info("[sticker-flow] INPUT template cycle (addr=%d)", addr)
            return True
        return False

    def handle_clamp_feedback(self, worker, inputs: list[bool]) -> None:
        """Transition CLAMPING → CLAMPED when feedback confirms."""
        if not self._sticker_settings.clamp_feedback_enabled:
            return
        addr = self._sticker_settings.input_clamp_engaged_address
        if addr < len(inputs) and inputs[addr]:
            logger.info("[sticker-flow] clamp feedback ON → CLAMPED")
