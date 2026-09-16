from __future__ import annotations

import os
import threading
import time

import cv2


class CameraCaptureService:
    def __init__(self) -> None:
        self._capture = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._frame = None
        self._camera_index = 0
        self._actual_settings: dict[str, float] = {}
        self._reconnecting = False
        self._reconnect_interval_s = 5.0
        self._last_frame_before_disconnect = None
        self._disconnect_notified = False
        self._status_callback = None
        self._epoch = 0  # generation token — incremented on each start()
        # Transient-drop tolerance: allow N brief retries before declaring disconnect
        self._max_consecutive_failures = int(os.getenv("QC_SUITE_CAMERA_MAX_CONSECUTIVE_FAILURES", "5"))
        self._transient_retry_delay_s = float(os.getenv("QC_SUITE_CAMERA_TRANSIENT_RETRY_S", "0.1"))
        self._consecutive_fail = 0

    def set_status_callback(self, callback) -> None:
        """Set callback for status updates: 'connected', 'reconnecting', 'error'."""
        self._status_callback = callback

    def _notify_status(self, status: str) -> None:
        if self._status_callback:
            try:
                self._status_callback(status)
            except Exception:
                pass

    def start(
        self,
        camera_index: int,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
    ) -> None:
        self.stop()
        with self._lock:
            self._epoch += 1  # new generation
        self._camera_index = int(camera_index)
        self._reconnecting = False
        self._disconnect_notified = False
        self._last_frame_before_disconnect = None
        self._consecutive_fail = 0
        self._open_camera(width, height, fps)
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="camera-capture"
        )
        self._thread.start()
        self._notify_status("connected")

    def _open_camera(self, width=None, height=None, fps=None) -> None:
        """Open camera with backend fallback."""
        self._capture = None
        backend_candidates = []
        if hasattr(cv2, "CAP_DSHOW"):
            backend_candidates.append(cv2.CAP_DSHOW)
        if hasattr(cv2, "CAP_MSMF"):
            backend_candidates.append(cv2.CAP_MSMF)
        backend_candidates.append(None)

        for backend in backend_candidates:
            capture = (
                cv2.VideoCapture(self._camera_index)
                if backend is None
                else cv2.VideoCapture(self._camera_index, backend)
            )
            if capture.isOpened():
                self._capture = capture
                break
            capture.release()

        if self._capture is None or not self._capture.isOpened():
            raise RuntimeError(f"Cannot open camera index {self._camera_index}")
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Configurable exposure
        _auto_exp = int(os.getenv("QC_SUITE_CAMERA_AUTO_EXPOSURE", "1"))
        if _auto_exp == 0:
            _exp_val = int(os.getenv("QC_SUITE_CAMERA_EXPOSURE_VALUE", "-6"))
            try:
                if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
                    self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                if hasattr(cv2, "CAP_PROP_EXPOSURE"):
                    self._capture.set(cv2.CAP_PROP_EXPOSURE, _exp_val)
            except Exception:
                pass
        if width is not None and int(width) > 0:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height is not None and int(height) > 0:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if fps is not None and float(fps) > 0:
            self._capture.set(cv2.CAP_PROP_FPS, float(fps))
        self._actual_settings = {
            "width": float(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0),
            "height": float(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0),
            "fps": float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0),
        }

    def _loop(self) -> None:
        """Main capture loop with auto-reconnect and epoch guard."""
        my_epoch = self._epoch  # capture epoch at thread start
        while True:
            # Epoch guard: bail if start() was called again (new generation)
            with self._lock:
                if self._epoch != my_epoch or not self._running:
                    return

            capture = self._capture
            if capture is None or not capture.isOpened():
                # Camera disconnected — try to reconnect
                with self._lock:
                    if self._epoch != my_epoch or not self._running:
                        return
                    if not self._reconnecting:
                        self._reconnecting = True
                        self._notify_status("reconnecting")
                        self._last_frame_before_disconnect = (
                            self._frame.copy() if self._frame is not None else None
                        )
                time.sleep(self._reconnect_interval_s)
                with self._lock:
                    if self._epoch != my_epoch or not self._running:
                        return
                try:
                    self._open_camera(
                        width=int(self._actual_settings.get("width") or 0) or None,
                        height=int(self._actual_settings.get("height") or 0) or None,
                        fps=self._actual_settings.get("fps") or None,
                    )
                    with self._lock:
                        if self._epoch != my_epoch or not self._running:
                            return
                        if self._capture is not None and self._capture.isOpened():
                            self._reconnecting = False
                            self._last_frame_before_disconnect = None
                            self._disconnect_notified = False
                            self._notify_status("connected")
                            continue
                except Exception:
                    pass
                continue

            ok, frame = capture.read()
            with self._lock:
                if self._epoch != my_epoch or not self._running:
                    return
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._consecutive_fail = 0  # reset on good frame
            else:
                # Frame read failed — could be transient (MSMF bandwidth) or real disconnect
                self._consecutive_fail += 1
                if self._consecutive_fail < self._max_consecutive_failures:
                    # Transient drop: brief retry without releasing camera
                    time.sleep(self._transient_retry_delay_s)
                    continue
                # Exceeded tolerance — treat as real disconnect
                with self._lock:
                    if self._epoch != my_epoch or not self._running:
                        return
                    if not self._reconnecting:
                        self._reconnecting = True
                        self._notify_status("reconnecting")
                        self._last_frame_before_disconnect = (
                            self._frame.copy() if self._frame is not None else None
                        )
                try:
                    capture.release()
                except Exception:
                    pass
                with self._lock:
                    if self._epoch != my_epoch or not self._running:
                        return
                    self._capture = None
                    self._consecutive_fail = 0
                time.sleep(self._reconnect_interval_s)

    def get_latest_frame(self):
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            if self._last_frame_before_disconnect is not None:
                return self._last_frame_before_disconnect.copy()
            return None

    def attempt_reconnect(self) -> bool:
        """Force an immediate reconnect attempt. Returns True if successful."""
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None
        self._reconnecting = False
        return False

    @property
    def actual_settings(self) -> dict[str, float]:
        return dict(self._actual_settings)

    @property
    def is_running(self) -> bool:
        return bool(self._running and self._capture is not None)

    @property
    def is_active(self) -> bool:
        """Return True while capture service is running (even during reconnect)."""
        return self._running

    @property
    def is_reconnecting(self) -> bool:
        return self._reconnecting

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._epoch += 1  # invalidate current generation
            capture = self._capture
            self._capture = None
        # Release capture outside lock to avoid blocking
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5.0)  # longer timeout; epoch guard ensures thread exits
        self._thread = None
        self._actual_settings = {}
        self._reconnecting = False
        self._last_frame_before_disconnect = None
        with self._lock:
            self._frame = None
