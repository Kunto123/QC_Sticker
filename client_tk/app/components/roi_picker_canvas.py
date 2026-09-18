from __future__ import annotations

import tkinter as tk
from typing import Callable

import customtkinter as ctk

import cv2
import numpy as np
from PIL import Image, ImageTk

from client_tk.app.theme import BORDER, PANEL_BG, TEXT_PRIMARY, TEXT_SECONDARY


_COLOR_PART_READY = (50, 180, 255)   # BGR oranye
_COLOR_STICKER = (0, 200, 255)       # BGR kuning
_COLOR_CROSSHAIR = (255, 255, 255)   # BGR putih
_ARM = 18
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_roi_box(frame, x: int, y: int, w: int, h: int, color, label: str) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame,
        label,
        (x, max(16, y - 6)),
        _LABEL_FONT,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_crosshair(frame, px: int, py: int, color, label: str = "") -> None:
    cv2.line(frame, (px - _ARM, py), (px + _ARM, py), color, 2, cv2.LINE_AA)
    cv2.line(frame, (px, py - _ARM), (px, py + _ARM), color, 2, cv2.LINE_AA)
    cv2.circle(frame, (px, py), 5, color, 1, cv2.LINE_AA)
    if label:
        cv2.putText(
            frame,
            label,
            (px + _ARM + 4, py + 5),
            _LABEL_FONT,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )


class RoiPickerCanvas(ctk.CTkFrame):
    """Interactive canvas for visualising ROI boxes and picking expected center.

    Usage::

        picker = RoiPickerCanvas(parent, title="ROI Picker", size=(640, 360))
        picker.on_center_changed = lambda cx, cy: print(cx, cy)
        picker.load_image(bgr_frame)
        picker.set_rois(part_ready_roi={"x":0.1,"y":0.1,"w":0.1,"h":0.1},
                        sticker_roi={"x":0.14,"y":0.25,"w":0.73,"h":0.37})
        picker.set_expected_center(0.5, 0.5)
    """

    def __init__(self, master, title: str = "ROI Visual Picker", *, size: tuple[int, int] = (640, 360)):
        super().__init__(master, fg_color=PANEL_BG, corner_radius=14, border_width=1, border_color=BORDER)
        self._display_size = size
        self.configure(width=size[0], height=size[1] + 56)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        ctk.CTkLabel(self, text=title, font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=(10, 6),
        )
        self._source_frame: np.ndarray | None = None
        self._part_ready_roi: dict = {}
        self._sticker_roi: dict = {}
        self._active_roi_kind: str | None = None
        self._drag_mode: str | None = None
        self._drag_start: tuple[float, float] | None = None
        self._drag_start_roi: dict | None = None
        self._cx: float = 0.5
        self._cy: float = 0.5
        self.on_center_changed: Callable[[float, float], None] | None = None
        self.on_roi_changed: Callable[[str, dict], None] | None = None

        self._canvas = tk.Canvas(
            self,
            width=size[0],
            height=size[1],
            bg="#0f172a",
            cursor="crosshair",
            highlightthickness=0,
        )
        self._canvas.grid(row=1, column=0, sticky="nw", padx=10)
        self._canvas.bind("<Button-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Motion>", self._on_motion)
        self._canvas.bind("<Configure>", lambda _: self.redraw())

        self._hint = ctk.CTkLabel(
            self,
            text="Klik dalam area kuning (Sticker ROI) untuk set expected center.",
            font=("Segoe UI", 9),
            wraplength=size[0] - 12,
            text_color=TEXT_SECONDARY,
        )
        self._hint.grid(row=2, column=0, sticky="w", padx=10, pady=(6, 10))
        self._photo = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_image(self, bgr_frame: np.ndarray | None) -> None:
        self._source_frame = bgr_frame.copy() if bgr_frame is not None else None
        self.redraw()

    def set_rois(self, part_ready_roi: dict | None = None, sticker_roi: dict | None = None) -> None:
        if part_ready_roi is not None:
            self._part_ready_roi = self._normalize_roi(part_ready_roi)
        if sticker_roi is not None:
            self._sticker_roi = self._normalize_roi(sticker_roi)
        self.redraw()

    def set_active_roi(self, kind: str | None) -> None:
        if kind not in {"part_ready", "sticker", None}:
            raise ValueError("kind must be 'part_ready', 'sticker', or None")
        self._active_roi_kind = kind
        self.redraw()

    def get_roi(self, kind: str) -> dict:
        if kind == "part_ready":
            return dict(self._part_ready_roi)
        if kind == "sticker":
            return dict(self._sticker_roi)
        raise ValueError("kind must be 'part_ready' or 'sticker'")

    def set_expected_center(self, cx: float | None, cy: float | None) -> None:
        self._cx = float(cx) if cx is not None else 0.5
        self._cy = float(cy) if cy is not None else 0.5
        self.redraw()

    def clear(self) -> None:
        self._source_frame = None
        self._canvas.delete("all")
        self._photo = None
        self._canvas.configure(bg="#0f172a")

    # ------------------------------------------------------------------
    # Live Camera
    # ------------------------------------------------------------------

    def start_live_camera(self, camera_index: int = 0) -> None:
        """Start capturing from camera and displaying on canvas."""
        import threading
        if hasattr(self, "_cam_thread") and self._cam_thread.is_alive():
            self.stop_live_camera()
        self._cam_running = True
        self._cam = cv2.VideoCapture(camera_index)
        if not self._cam.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        self._cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
        self._cam_thread.start()

    def stop_live_camera(self) -> None:
        """Stop live camera feed."""
        self._cam_running = False
        if hasattr(self, "_cam_thread"):
            self._cam_thread.join(timeout=2.0)
        if hasattr(self, "_cam") and self._cam is not None:
            self._cam.release()
            self._cam = None

    def _cam_loop(self) -> None:
        """Background thread: read frames and schedule redraw."""
        while getattr(self, "_cam_running", False):
            ret, frame = self._cam.read()
            if not ret:
                continue
            self._source_frame = frame
            try:
                self.after(33, self.redraw)
            except tk.TclError:
                break

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def redraw(self) -> None:
        if not self.winfo_exists():
            return
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 8:
            cw = self._display_size[0]
        if ch < 8:
            ch = self._display_size[1]

        if self._source_frame is None:
            self._canvas.delete("all")
            self._canvas.create_text(
                cw // 2,
                ch // 2,
                text="Belum ada gambar.\nKlik 'Load Image' atau 'Load from Session'.",
                fill="#475569",
                font=("Segoe UI", 11),
                justify="center",
            )
            return

        frame = self._render_frame(cw, ch)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(pil)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def _render_frame(self, cw: int, ch: int) -> np.ndarray:
        src_h, src_w = self._source_frame.shape[:2]
        scale = min(cw / max(src_w, 1), ch / max(src_h, 1))
        dw = max(1, int(src_w * scale))
        dh = max(1, int(src_h * scale))
        frame = cv2.resize(self._source_frame, (dw, dh), interpolation=cv2.INTER_AREA)

        # Expand canvas to (cw, ch) with black padding
        canvas_frame = np.zeros((ch, cw, 3), dtype=np.uint8)
        off_x = (cw - dw) // 2
        off_y = (ch - dh) // 2
        canvas_frame[off_y:off_y + dh, off_x:off_x + dw] = frame

        def _roi_px(roi: dict) -> tuple[int, int, int, int]:
            rx = off_x + int(float(roi.get("x", 0.0)) * dw)
            ry = off_y + int(float(roi.get("y", 0.0)) * dh)
            rw = max(1, int(float(roi.get("w", 1.0)) * dw))
            rh = max(1, int(float(roi.get("h", 1.0)) * dh))
            return rx, ry, rw, rh

        if self._part_ready_roi:
            rx, ry, rw, rh = _roi_px(self._part_ready_roi)
            _draw_roi_box(canvas_frame, rx, ry, rw, rh, _COLOR_PART_READY, "Part Ready ROI")
            if self._active_roi_kind == "part_ready":
                self._draw_handles(canvas_frame, rx, ry, rw, rh, _COLOR_PART_READY)

        if self._sticker_roi:
            rx, ry, rw, rh = _roi_px(self._sticker_roi)
            _draw_roi_box(canvas_frame, rx, ry, rw, rh, _COLOR_STICKER, "Sticker ROI")
            if self._active_roi_kind == "sticker":
                self._draw_handles(canvas_frame, rx, ry, rw, rh, _COLOR_STICKER)
            exp_px = rx + int(self._cx * rw)
            exp_py = ry + int(self._cy * rh)
            _draw_crosshair(canvas_frame, exp_px, exp_py, _COLOR_CROSSHAIR, "EXP CTR")

        return canvas_frame

    # ------------------------------------------------------------------
    # Drag handlers
    # ------------------------------------------------------------------

    def _draw_handles(self, frame, x: int, y: int, w: int, h: int, color) -> None:
        for px, py in ((x, y), (x + w, y), (x, y + h), (x + w, y + h)):
            cv2.rectangle(frame, (px - 5, py - 5), (px + 5, py + 5), color, -1)
            cv2.rectangle(frame, (px - 5, py - 5), (px + 5, py + 5), (255, 255, 255), 1)

    def _image_layout(self) -> tuple[int, int, int, int] | None:
        if self._source_frame is None:
            return None
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 8:
            cw = self._display_size[0]
        if ch < 8:
            ch = self._display_size[1]
        src_h, src_w = self._source_frame.shape[:2]
        scale = min(cw / max(src_w, 1), ch / max(src_h, 1))
        dw = max(1, int(src_w * scale))
        dh = max(1, int(src_h * scale))
        off_x = (cw - dw) // 2
        off_y = (ch - dh) // 2
        return off_x, off_y, dw, dh

    def _event_to_norm(self, event: tk.Event) -> tuple[float, float] | None:
        layout = self._image_layout()
        if layout is None:
            return None
        off_x, off_y, dw, dh = layout
        nx = (float(event.x) - off_x) / max(dw, 1)
        ny = (float(event.y) - off_y) / max(dh, 1)
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    def _roi_for_kind(self, kind: str | None) -> dict:
        if kind == "part_ready":
            return self._part_ready_roi
        if kind == "sticker":
            return self._sticker_roi
        return {}

    def _set_roi_for_kind(self, kind: str, roi: dict, *, notify: bool = True) -> None:
        normalized = self._normalize_roi(roi)
        if kind == "part_ready":
            self._part_ready_roi = normalized
        elif kind == "sticker":
            self._sticker_roi = normalized
        else:
            return
        self.redraw()
        if notify and self.on_roi_changed:
            self.on_roi_changed(kind, dict(normalized))

    def _normalize_roi(self, roi: dict) -> dict:
        x = self._to_float(roi.get("x"), 0.0)
        y = self._to_float(roi.get("y"), 0.0)
        w = self._to_float(roi.get("w", roi.get("width")), 1.0)
        h = self._to_float(roi.get("h", roi.get("height")), 1.0)
        min_size = 0.01
        w = max(min_size, min(1.0, w))
        h = max(min_size, min(1.0, h))
        x = max(0.0, min(1.0 - w, x))
        y = max(0.0, min(1.0 - h, y))
        return {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)}

    def _to_float(self, value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _roi_to_display_rect(self, roi: dict) -> tuple[int, int, int, int] | None:
        layout = self._image_layout()
        if layout is None or not roi:
            return None
        off_x, off_y, dw, dh = layout
        x = off_x + int(float(roi.get("x", 0.0)) * dw)
        y = off_y + int(float(roi.get("y", 0.0)) * dh)
        w = max(1, int(float(roi.get("w", 1.0)) * dw))
        h = max(1, int(float(roi.get("h", 1.0)) * dh))
        return x, y, w, h

    def _hit_test(self, event: tk.Event, roi: dict) -> str | None:
        rect = self._roi_to_display_rect(roi)
        if rect is None:
            return None
        x, y, w, h = rect
        px, py = int(event.x), int(event.y)
        handles = {
            "nw": (x, y), "ne": (x + w, y),
            "sw": (x, y + h), "se": (x + w, y + h),
        }
        for mode, (hx, hy) in handles.items():
            if abs(px - hx) <= 8 and abs(py - hy) <= 8:
                return mode
        if x <= px <= x + w and y <= py <= y + h:
            return "move"
        return None

    def _on_press(self, event: tk.Event) -> None:
        if self._source_frame is None:
            return
        if self._active_roi_kind is None:
            self._on_expected_center_click(event)
            return
        roi = self._roi_for_kind(self._active_roi_kind)
        point = self._event_to_norm(event)
        if point is None:
            return
        mode = self._hit_test(event, roi)
        if mode is None:
            current = self._normalize_roi(roi or {"x": 0.2, "y": 0.2, "w": 0.25, "h": 0.25})
            mode = "move"
            roi = {
                **current,
                "x": point[0] - current["w"] / 2,
                "y": point[1] - current["h"] / 2,
            }
            self._set_roi_for_kind(self._active_roi_kind, roi)
            roi = self._roi_for_kind(self._active_roi_kind)
        self._drag_mode = mode
        self._drag_start = point
        self._drag_start_roi = dict(roi)

    def _on_drag(self, event: tk.Event) -> None:
        if not self._active_roi_kind or not self._drag_mode or self._drag_start is None or self._drag_start_roi is None:
            return
        point = self._event_to_norm(event)
        if point is None:
            return
        sx, sy = self._drag_start
        dx = point[0] - sx
        dy = point[1] - sy
        roi = dict(self._drag_start_roi)
        x = float(roi.get("x", 0.0))
        y = float(roi.get("y", 0.0))
        w = float(roi.get("w", 1.0))
        h = float(roi.get("h", 1.0))
        min_size = 0.01

        if self._drag_mode == "move":
            roi["x"] = x + dx
            roi["y"] = y + dy
        else:
            left = x
            top = y
            right = x + w
            bottom = y + h
            if "w" in self._drag_mode:
                left = max(0.0, min(right - min_size, left + dx))
            if "e" in self._drag_mode:
                right = min(1.0, max(left + min_size, right + dx))
            if "n" in self._drag_mode:
                top = max(0.0, min(bottom - min_size, top + dy))
            if "s" in self._drag_mode:
                bottom = min(1.0, max(top + min_size, bottom + dy))
            roi = {"x": left, "y": top, "w": right - left, "h": bottom - top}

        self._set_roi_for_kind(self._active_roi_kind, roi)

    def _on_release(self, _event: tk.Event) -> None:
        self._drag_mode = None
        self._drag_start = None
        self._drag_start_roi = None

    def _on_motion(self, event: tk.Event) -> None:
        if self._active_roi_kind is None:
            self._canvas.configure(cursor="crosshair")
            return
        mode = self._hit_test(event, self._roi_for_kind(self._active_roi_kind))
        cursors = {
            "move": "fleur",
            "nw": "size_nw_se",
            "se": "size_nw_se",
            "ne": "size_ne_sw",
            "sw": "size_ne_sw",
        }
        try:
            self._canvas.configure(cursor=cursors.get(mode, "crosshair"))
        except tk.TclError:
            self._canvas.configure(cursor="crosshair")

    def _on_expected_center_click(self, event: tk.Event) -> None:
        if self._source_frame is None or not self._sticker_roi:
            return
        rect = self._roi_to_display_rect(self._sticker_roi)
        if rect is None:
            return
        rx, ry, rw, rh = rect

        if not (rx <= event.x <= rx + rw and ry <= event.y <= ry + rh):
            self._hint.configure(text="Klik dalam area kuning (Sticker ROI) untuk set expected center.")
            return

        cx = max(0.0, min(1.0, (event.x - rx) / rw))
        cy = max(0.0, min(1.0, (event.y - ry) / rh))
        self._cx = round(cx, 4)
        self._cy = round(cy, 4)
        self._hint.configure(text=f"Expected center: x={self._cx:.4f}, y={self._cy:.4f}  (tersimpan ke field di bawah)")
        self.redraw()
        if self.on_center_changed:
            self.on_center_changed(self._cx, self._cy)
