from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from client_tk.app.services.frame_upload import FrameUploadService


class FrameUploadServiceTest(unittest.TestCase):
    def test_upload_service_sends_frame_and_returns_result(self) -> None:
        service = FrameUploadService()
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        done = threading.Event()
        results: list[dict] = []
        errors: list[str] = []

        def get_frame():
            return frame

        def send_frame(image_b64: str) -> dict:
            self.assertTrue(image_b64)
            return {"ok": True}

        def on_result(payload: dict) -> None:
            results.append(payload)
            done.set()
            service.stop()

        def on_error(message: str) -> None:
            errors.append(message)
            done.set()
            service.stop()

        service.start(
            interval_ms=100,
            get_frame=get_frame,
            send_frame=send_frame,
            on_result=on_result,
            on_error=on_error,
        )

        self.assertTrue(done.wait(1.5), "Timed out waiting for frame upload result")
        service.stop()
        self.assertFalse(errors)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["ok"])

    def test_upload_interval_compensates_request_time(self) -> None:
        service = FrameUploadService()
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        done = threading.Event()
        errors: list[str] = []
        call_times: list[float] = []

        def get_frame():
            return frame

        def send_frame(_image_b64: str) -> dict:
            call_times.append(time.perf_counter())
            time.sleep(0.2)
            if len(call_times) >= 2:
                done.set()
                service.stop()
            return {"ok": True}

        def on_result(_payload: dict) -> None:
            pass

        def on_error(message: str) -> None:
            errors.append(message)
            done.set()
            service.stop()

        service.start(
            interval_ms=250,
            get_frame=get_frame,
            send_frame=send_frame,
            on_result=on_result,
            on_error=on_error,
        )

        self.assertTrue(done.wait(3.0), "Timed out waiting for second upload iteration")
        service.stop()
        self.assertFalse(errors)
        self.assertGreaterEqual(len(call_times), 2)

        period_s = call_times[1] - call_times[0]
        self.assertGreaterEqual(period_s, 0.2)
        # New pacing targets interval period. Old behavior would be roughly 0.45s (request + sleep).
        self.assertLess(period_s, 0.36)


if __name__ == "__main__":
    unittest.main()
