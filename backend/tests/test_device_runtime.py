from __future__ import annotations

import unittest

from backend.app.core.device_runtime import DeviceRuntimeResolver


class _Config:
    def __init__(self, *, device_mode: str = "auto", cuda_device_id: int = 0) -> None:
        self.device_mode = device_mode
        self.cuda_device_id = cuda_device_id


class DeviceRuntimeResolverTest(unittest.TestCase):
    def test_auto_uses_cuda_when_available(self) -> None:
        resolver = DeviceRuntimeResolver(_Config(device_mode="auto", cuda_device_id=1))
        resolver._cuda_state = (True, 2, None)  # type: ignore[assignment]

        resolution = resolver.resolve()

        self.assertEqual(resolution.requested_mode, "auto")
        self.assertEqual(resolution.effective_device, "cuda:1")
        self.assertEqual(resolution.backend, "cuda")
        self.assertIsNone(resolution.fallback_reason)

    def test_gpu_falls_back_to_cpu_when_cuda_missing(self) -> None:
        resolver = DeviceRuntimeResolver(_Config(device_mode="gpu", cuda_device_id=0))
        resolver._cuda_state = (False, 0, "cuda_unavailable")  # type: ignore[assignment]

        resolution = resolver.resolve()

        self.assertEqual(resolution.requested_mode, "gpu")
        self.assertEqual(resolution.effective_device, "cpu")
        self.assertEqual(resolution.backend, "cpu")
        self.assertEqual(resolution.fallback_reason, "cuda_unavailable")

    def test_invalid_cuda_index_clamps_to_first_available_device(self) -> None:
        resolver = DeviceRuntimeResolver(_Config(device_mode="auto", cuda_device_id=5))
        resolver._cuda_state = (True, 1, None)  # type: ignore[assignment]

        resolution = resolver.resolve()

        self.assertEqual(resolution.effective_device, "cuda:0")
        self.assertEqual(resolution.backend, "cuda")
        self.assertIn("cuda_device_id_out_of_range", resolution.fallback_reason or "")

    def test_torch_not_installed_returns_distinct_reason(self) -> None:
        resolver = DeviceRuntimeResolver(_Config(device_mode="gpu", cuda_device_id=0))
        # Simulate torch import failure via cached torch=False + no cached cuda state.
        resolver._torch = False
        resolver._cuda_state = None

        resolution = resolver.resolve()

        self.assertEqual(resolution.effective_device, "cpu")
        self.assertEqual(resolution.fallback_reason, "torch_not_installed")

    def test_cuda_device_count_zero_returns_distinct_reason(self) -> None:
        resolver = DeviceRuntimeResolver(_Config(device_mode="gpu", cuda_device_id=0))
        # Simulate CUDA "available" but zero devices visible.
        resolver._cuda_state = (True, 0, "cuda_device_count_zero")

        resolution = resolver.resolve()

        self.assertEqual(resolution.effective_device, "cpu")
        self.assertEqual(resolution.fallback_reason, "cuda_device_count_zero")

    def test_auto_mode_cuda_unavailable_falls_back_with_reason(self) -> None:
        resolver = DeviceRuntimeResolver(_Config(device_mode="auto", cuda_device_id=0))
        resolver._cuda_state = (False, 0, "cuda_unavailable")

        resolution = resolver.resolve()

        self.assertEqual(resolution.requested_mode, "auto")
        self.assertEqual(resolution.effective_device, "cpu")
        self.assertEqual(resolution.fallback_reason, "cuda_unavailable")