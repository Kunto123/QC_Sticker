from __future__ import annotations

import threading
import time
import unittest
from typing import Callable
from unittest import mock

from client_tk.app.components.async_bridge import run_async


class _DummyWidget:
    def __init__(self) -> None:
        self.alive = True
        self.scheduled_callbacks: list[Callable[[], None]] = []

    def after(self, _delay_ms: int, callback):
        self.scheduled_callbacks.append(callback)
        return f"job-{len(self.scheduled_callbacks)}"

    def winfo_exists(self) -> bool:
        return self.alive


def _pump_widget(widget: _DummyWidget, timeout: float = 1.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        while widget.scheduled_callbacks:
            callback = widget.scheduled_callbacks.pop(0)
            callback()
        if not widget.scheduled_callbacks:
            time.sleep(0.01)
        else:
            continue
        if not widget.scheduled_callbacks:
            return
    raise AssertionError("Timed out while waiting for async callbacks")


class AsyncBridgeTest(unittest.TestCase):
    def test_run_async_delivers_result_on_main_thread(self) -> None:
        widget = _DummyWidget()
        callback_threads: list[threading.Thread] = []
        callback_results: list[tuple[str | None, Exception | None]] = []
        worker_started = threading.Event()
        worker_release = threading.Event()

        def worker() -> str:
            worker_started.set()
            worker_release.wait(1)
            return "ok"

        run_async(
            widget,
            worker,
            callback=lambda result, error: (
                callback_threads.append(threading.current_thread()),
                callback_results.append((result, error)),
            ),
        )

        self.assertTrue(worker_started.wait(1), "Worker did not start")
        self.assertFalse(callback_results)

        worker_release.set()
        _pump_widget(widget)

        self.assertEqual(callback_results, [("ok", None)])
        self.assertEqual(callback_threads, [threading.main_thread()])

    def test_run_async_delivers_exception_on_main_thread(self) -> None:
        widget = _DummyWidget()
        callback_threads: list[threading.Thread] = []
        callback_results: list[tuple[object, Exception | None]] = []

        def worker() -> str:
            raise RuntimeError("boom")

        run_async(
            widget,
            worker,
            callback=lambda result, error: (
                callback_threads.append(threading.current_thread()),
                callback_results.append((result, error)),
            ),
        )

        _pump_widget(widget)

        self.assertEqual(len(callback_results), 1)
        self.assertIsNone(callback_results[0][0])
        self.assertIsInstance(callback_results[0][1], RuntimeError)
        self.assertEqual(callback_threads, [threading.main_thread()])

    def test_run_async_logs_callback_exception(self) -> None:
        widget = _DummyWidget()

        def worker() -> str:
            return "ok"

        def callback(_result, _error):
            raise RuntimeError("callback boom")

        with mock.patch("traceback.print_exc") as print_exc:
            run_async(widget, worker, callback=callback)
            _pump_widget(widget)

        print_exc.assert_called_once()
