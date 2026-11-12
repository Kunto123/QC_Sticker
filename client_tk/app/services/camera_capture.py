from __future__ import annotations

import logging
import os
import threading
import time

import cv2

_logger = logging.getLogger(__name__)


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
        self._epoch = 0  # token generasi — bertambah tiap kali start()
        # Toleransi drop transient: izinkan N retry singkat sebelum menyatakan disconnect
        self._max_consecutive_failures = int(os.getenv("QC_SUITE_CAMERA_MAX_CONSECUTIVE_FAILURES", "5"))
        self._transient_retry_delay_s = float(os.getenv("QC_SUITE_CAMERA_TRANSIENT_RETRY_S", "0.1"))
        self._consecutive_fail = 0

    def set_status_callback(self, callback) -> None:
        """Set callback untuk update status: 'connected', 'reconnecting', 'error'."""
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
            self._epoch += 1  # generasi baru
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
        """Buka kamera dengan fallback backend."""
        self._capture = None
        # Urutan sama dengan cv2.VideoCapture(idx) biasa (yang dipakai live camera admin,
        # terbukti jalan), lalu DSHOW / MSMF eksplisit sebagai cadangan. Sebagian kamera USB
        # "terbuka" di DSHOW tapi mengirim frame hitam tanpa error apa pun, jadi tiap backend
        # diverifikasi dengan membaca frame nyata sebelum dipakai.
        backend_candidates = [None]
        if hasattr(cv2, "CAP_DSHOW"):
            backend_candidates.append(cv2.CAP_DSHOW)
        if hasattr(cv2, "CAP_MSMF"):
            backend_candidates.append(cv2.CAP_MSMF)

        fallback = None  # backend yang terbuka tapi frame-nya hitam / kosong
        fallback_name = ""
        chosen = None
        for backend in backend_candidates:
            name = "default" if backend is None else str(backend)
            capture = (
                cv2.VideoCapture(self._camera_index)
                if backend is None
                else cv2.VideoCapture(self._camera_index, backend)
            )
            if not capture.isOpened():
                capture.release()
                continue
            self._configure_capture(capture, width, height, fps)
            if self._capture_has_signal(capture):
                chosen, chosen_name = capture, name
                break
            _logger.warning(
                "[camera] index=%s backend=%s terbuka tapi frame hitam/kosong, coba backend lain",
                self._camera_index,
                name,
            )
            if fallback is None:
                fallback, fallback_name = capture, name
            else:
                capture.release()

        if chosen is None:
            chosen, chosen_name = fallback, fallback_name
            fallback = None
            if chosen is not None:
                _logger.warning(
                    "[camera] index=%s: semua backend memberi frame hitam; pakai backend=%s. "
                    "Cek exposure / kamera tertutup / driver.",
                    self._camera_index,
                    chosen_name,
                )
        if fallback is not None:
            fallback.release()

        if chosen is None or not chosen.isOpened():
            raise RuntimeError(f"Tidak bisa membuka kamera index {self._camera_index}")
        self._capture = chosen
        _logger.info("[camera] index=%s dibuka dengan backend=%s", self._camera_index, chosen_name)
        self._actual_settings = {
            "width": float(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0),
            "height": float(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0),
            "fps": float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0),
        }

    @staticmethod
    def _configure_capture(capture, width, height, fps) -> None:
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Exposure yang bisa dikonfigurasi
        _auto_exp = int(os.getenv("QC_SUITE_CAMERA_AUTO_EXPOSURE", "1"))
        if _auto_exp == 0:
            _exp_val = int(os.getenv("QC_SUITE_CAMERA_EXPOSURE_VALUE", "-6"))
            try:
                if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
                    capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                if hasattr(cv2, "CAP_PROP_EXPOSURE"):
                    capture.set(cv2.CAP_PROP_EXPOSURE, _exp_val)
            except Exception:
                pass
        if width is not None and int(width) > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height is not None and int(height) > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if fps is not None and float(fps) > 0:
            capture.set(cv2.CAP_PROP_FPS, float(fps))

    @staticmethod
    def _capture_has_signal(capture, attempts: int = 12, delay_s: float = 0.05) -> bool:
        """True kalau dalam beberapa read ada frame yang tidak hitam total."""
        for _ in range(attempts):
            ok, frame = capture.read()
            if ok and frame is not None and frame.size and int(frame.max()) > 8:
                return True
            time.sleep(delay_s)
        return False

    def _loop(self) -> None:
        """Loop capture utama dengan auto-reconnect dan epoch guard."""
        my_epoch = self._epoch  # catat epoch saat thread mulai
        while True:
            # Epoch guard: keluar kalau start() dipanggil lagi (generasi baru)
            with self._lock:
                if self._epoch != my_epoch or not self._running:
                    return

            capture = self._capture
            if capture is None or not capture.isOpened():
                # Kamera terputus — coba reconnect
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
                    self._consecutive_fail = 0  # reset kalau frame bagus
            else:
                # Read frame gagal — bisa transient (bandwidth MSMF) atau disconnect beneran
                self._consecutive_fail += 1
                if self._consecutive_fail < self._max_consecutive_failures:
                    # Drop transient: retry singkat tanpa melepas kamera
                    time.sleep(self._transient_retry_delay_s)
                    continue
                # Melebihi toleransi — anggap disconnect beneran
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

    @property
    def actual_settings(self) -> dict[str, float]:
        return dict(self._actual_settings)

    @property
    def is_active(self) -> bool:
        """Return True selama service capture berjalan (bahkan saat reconnect)."""
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
        # Release capture di luar lock supaya tidak blocking
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5.0)  # timeout lebih lama; epoch guard memastikan thread keluar
        self._thread = None
        self._actual_settings = {}
        self._reconnecting = False
        self._last_frame_before_disconnect = None
        with self._lock:
            self._frame = None
