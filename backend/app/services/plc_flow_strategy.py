"""Pola PLC Flow Strategy.

Tiap strategy membungkus:
  - Cara menafsirkan input (release, template cycle, clamp feedback)
  - Cara memetakan event inspeksi → aksi coil (accept/reject/part_ready)
  - Timing apa yang dipakai (durasi pulse, debounce, dll.)

PlcWorker mendelegasikan semua behavior spesifik-mode ke strategy aktif.
Strategy membaca alamat I/O/timing dari MachineSettings, BUKAN dari env var.
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
    """Base abstrak untuk PLC flow strategy.

    Tiap strategy memegang seluruh lifecycle:
      part_ready → clamp → inspection → accept/reject → release
    """

    def __init__(
        self,
        adapter: PlcAdapter,
        settings,  # PlcIoConfig
        num_channels: int = 4,
    ):
        self._adapter = adapter
        self._settings = settings
        self._num_channels = num_channels

    # ── Subclass wajib implement ──────────────────────────────────────

    @abstractmethod
    def on_part_ready(self, worker) -> None:
        """Kunci clamp saat part siap. State → CLAMPING."""
        ...

    @abstractmethod
    def on_accept(self, worker) -> None:
        """Accept: lepas clamp + pulse sinyal OK. State → ACCEPT_PULSE."""
        ...

    @abstractmethod
    def on_reject(self, worker) -> None:
        """Reject: nyalakan buzzer reject + clamp tetap. State → REJECT_BUZZER."""
        ...

    @abstractmethod
    def handle_input_release(self, worker, inputs: list[bool]) -> bool:
        """Return True kalau release ter-trigger (pemanggil harus all_off)."""
        ...

    @abstractmethod
    def handle_input_template_cycle(self, worker, inputs: list[bool]) -> bool:
        """Return True kalau template cycle ter-trigger."""
        ...

    @abstractmethod
    def handle_clamp_feedback(self, worker, inputs: list[bool]) -> None:
        """Transisi CLAMPING → CLAMPED saat feedback mengonfirmasi."""
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

    # ── Helper bersama ───────────────────────────────────────────────

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
            f"write_coil addr={addr} gagal setelah {max_retries} percobaan"
        )

    @property
    @abstractmethod
    def flow_name(self) -> str:
        ...
