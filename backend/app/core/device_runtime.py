from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.core.config import AppConfig


_VALID_DEVICE_MODES = {"auto", "gpu", "cpu"}


@dataclass(slots=True)
class DeviceResolution:
    requested_mode: str
    effective_device: str
    backend: str
    gpu_available: bool
    cuda_device_id: int | None
    fallback_reason: str | None = None


class DeviceRuntimeResolver:
    def __init__(self, app_config: AppConfig) -> None:
        self._config = app_config
        self._torch: object | None = None
        self._cuda_state: tuple[bool, int, str | None] | None = None

    def _normalize_mode(self, value: str | None) -> str:
        mode = str(value or "auto").strip().lower() or "auto"
        return mode if mode in _VALID_DEVICE_MODES else "auto"

    def _load_torch(self):
        if self._torch is not None:
            return self._torch
        try:
            import torch  # type: ignore
        except Exception:  # noqa: BLE001
            self._torch = False
        else:
            self._torch = torch
        return self._torch

    def _cuda_status(self) -> tuple[bool, int, str | None]:
        """Return (available, device_count, unavail_reason).

        ``unavail_reason`` is one of:
        * ``None``                    — CUDA is available (no failure)
        * ``"torch_not_installed"``   — torch package cannot be imported
        * ``"cuda_unavailable"``      — torch imported but ``cuda.is_available()`` is False
        * ``"cuda_device_count_zero"``— CUDA is available but no devices are visible
        """
        if self._cuda_state is not None:
            return self._cuda_state
        torch = self._load_torch()
        if torch is False or torch is None:
            self._cuda_state = (False, 0, "torch_not_installed")
            return self._cuda_state
        try:
            available = bool(torch.cuda.is_available())
            count = int(torch.cuda.device_count()) if available else 0
            if not available:
                reason: str | None = "cuda_unavailable"
            elif count == 0:
                reason = "cuda_device_count_zero"
            else:
                reason = None
        except Exception:  # noqa: BLE001
            available, count, reason = False, 0, "cuda_unavailable"
        self._cuda_state = (available, count, reason)
        return self._cuda_state

    def resolve(self, requested_mode: str | None = None) -> DeviceResolution:
        mode = self._normalize_mode(requested_mode or self._config.device_mode)
        cuda_device_id = max(0, int(self._config.cuda_device_id))

        if mode == "cpu":
            return DeviceResolution(
                requested_mode=mode,
                effective_device="cpu",
                backend="cpu",
                gpu_available=False,
                cuda_device_id=None,
            )

        gpu_available, device_count, unavail_reason = self._cuda_status()
        if not gpu_available or device_count <= 0:
            return DeviceResolution(
                requested_mode=mode,
                effective_device="cpu",
                backend="cpu",
                gpu_available=False,
                cuda_device_id=None,
                fallback_reason=unavail_reason or "cuda_unavailable",
            )

        effective_index = min(cuda_device_id, device_count - 1)
        fallback_reason = None
        if effective_index != cuda_device_id:
            fallback_reason = f"cuda_device_id_out_of_range:{cuda_device_id}->{effective_index}"

        return DeviceResolution(
            requested_mode=mode,
            effective_device=f"cuda:{effective_index}",
            backend="cuda",
            gpu_available=True,
            cuda_device_id=effective_index,
            fallback_reason=fallback_reason,
        )