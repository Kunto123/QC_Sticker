from __future__ import annotations

import base64
import threading
import time
from collections.abc import Callable

import cv2

# Rolling window to compute average request_ms for adaptive quality.
_ADAPTIVE_WINDOW = 6
# If avg request_ms exceeds this threshold, reduce JPEG quality one step.
_OVERLOAD_THRESHOLD_MS = 200.0
# If avg request_ms drops below this threshold, restore quality one step.
_RECOVER_THRESHOLD_MS = 100.0
_QUALITY_STEP = 5
_QUALITY_MIN = 45
# Width to auto-downscale to when backend is consistently overloaded.
_ADAPTIVE_RESIZE_WIDTH = 480


class FrameUploadService:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._running = False
        self._last_client_timings: dict = {}
        self._timings_lock = threading.Lock()

    @property
    def last_client_timings(self) -> dict:
        with self._timings_lock:
            return dict(self._last_client_timings)

    def start(
        self,
        *,
        interval_ms: int,
        get_frame: Callable[[], object],
        send_frame: Callable[[str], dict],
        on_result: Callable[[dict], None],
        on_error: Callable[[str], None],
        jpeg_quality: int = 75,
        resize_width: int | None = None,
        adaptive: bool = True,
        use_png: bool = False,       # ← tambah
    ) -> None:
        self.stop()
        self._running = True
        target_interval_s = max(0.05, interval_ms / 1000.0)
        base_quality = max(40, min(100, jpeg_quality))
        base_resize = resize_width

        def _loop():
            current_quality = base_quality
            current_resize = base_resize
            request_ms_history: list[float] = []

            while self._running:
                loop_started = time.perf_counter()
                try:
                    t0 = time.perf_counter()
                    frame = get_frame()
                    capture_ms = (time.perf_counter() - t0) * 1000.0

                    if frame is None:
                        elapsed_s = time.perf_counter() - loop_started
                        sleep_s = target_interval_s - elapsed_s
                        if sleep_s > 0:
                            time.sleep(sleep_s)
                        continue

                    # Adaptive: kick in auto-resize when backend is overloaded.
                    effective_resize = current_resize
                    if adaptive and current_resize is None:
                        if (
                            len(request_ms_history) >= _ADAPTIVE_WINDOW
                            and sum(request_ms_history) / len(request_ms_history) > _OVERLOAD_THRESHOLD_MS * 1.5
                        ):
                            effective_resize = _ADAPTIVE_RESIZE_WIDTH

                    resize_ms = 0.0
                    if effective_resize is not None and frame.shape[1] > effective_resize:
                        scale = effective_resize / frame.shape[1]
                        new_h = int(frame.shape[0] * scale)
                        t0 = time.perf_counter()
                        frame = cv2.resize(frame, (effective_resize, new_h), interpolation=cv2.INTER_LINEAR)
                        resize_ms = (time.perf_counter() - t0) * 1000.0

                    t0 = time.perf_counter()
                    if use_png:
                        ok, encoded = cv2.imencode(".png", frame)
                    else:
                        encode_params = [cv2.IMWRITE_JPEG_QUALITY, current_quality]
                        ok, encoded = cv2.imencode(".jpg", frame, encode_params)
                    encode_ms = (time.perf_counter() - t0) * 1000.0

                    if not ok:
                        raise RuntimeError("Failed to encode frame.")

                    t0 = time.perf_counter()
                    image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
                    b64_ms = (time.perf_counter() - t0) * 1000.0

                    t0 = time.perf_counter()
                    result = send_frame(image_b64)
                    request_ms = (time.perf_counter() - t0) * 1000.0

                    # Adaptive quality adjustment.
                    if adaptive and not use_png:
                        request_ms_history.append(request_ms)
                        if len(request_ms_history) > _ADAPTIVE_WINDOW:
                            del request_ms_history[0]
                        avg_ms = sum(request_ms_history) / len(request_ms_history)
                        if avg_ms > _OVERLOAD_THRESHOLD_MS and current_quality > _QUALITY_MIN:
                            current_quality = max(_QUALITY_MIN, current_quality - _QUALITY_STEP)
                        elif avg_ms < _RECOVER_THRESHOLD_MS and current_quality < base_quality:
                            current_quality = min(base_quality, current_quality + _QUALITY_STEP)

                    client_timings = {
                        "capture_ms": round(capture_ms, 2),
                        "resize_ms": round(resize_ms, 2),
                        "encode_ms": round(encode_ms, 2),
                        "b64_ms": round(b64_ms, 2),
                        "request_ms": round(request_ms, 2),
                        "frame_width": frame.shape[1],
                        "frame_height": frame.shape[0],
                        "jpeg_quality": current_quality,
                        "payload_bytes": len(encoded),
                        "adaptive_quality": current_quality if adaptive else None,
                        "adaptive_resize": effective_resize if adaptive else None,
                    }
                    with self._timings_lock:
                        self._last_client_timings = client_timings

                    if isinstance(result, dict):
                        result.setdefault("client_timings", client_timings)

                    on_result(result)
                except Exception as exc:  # noqa: BLE001
                    on_error(str(exc))

                if not self._running:
                    break

                elapsed_s = time.perf_counter() - loop_started
                sleep_s = target_interval_s - elapsed_s
                if sleep_s > 0:
                    time.sleep(sleep_s)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.3)
        self._thread = None
        with self._timings_lock:
            self._last_client_timings = {}
