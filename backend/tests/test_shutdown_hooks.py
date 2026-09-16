from __future__ import annotations

import unittest

from backend.app.core import container, shutdown


class _DummyWorker:
    def __init__(self, name: str, calls: list[str]) -> None:
        self._name = name
        self._calls = calls
        self.stop_count = 0

    def stop(self, timeout: float = 0.0) -> None:
        del timeout
        self.stop_count += 1
        self._calls.append(self._name)


class ShutdownHooksTest(unittest.TestCase):
    def test_shutdown_workers_is_idempotent_and_stops_plc_before_push(self) -> None:
        calls: list[str] = []
        original_push_worker = container.push_worker
        original_plc_worker = container.plc_worker
        original_shutdown_flag = shutdown._shutdown_workers_stopped  # noqa: SLF001
        try:
            container.push_worker = _DummyWorker("push", calls)
            container.plc_worker = _DummyWorker("plc", calls)
            shutdown._shutdown_workers_stopped = False  # noqa: SLF001

            shutdown.shutdown_workers(reason="test")
            shutdown.shutdown_workers(reason="test-again")

            self.assertEqual(calls, ["plc", "push"])
            self.assertEqual(container.plc_worker.stop_count, 1)
            self.assertEqual(container.push_worker.stop_count, 1)
        finally:
            container.push_worker = original_push_worker
            container.plc_worker = original_plc_worker
            shutdown._shutdown_workers_stopped = original_shutdown_flag  # noqa: SLF001