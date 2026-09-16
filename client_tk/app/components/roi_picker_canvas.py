from __future__ import annotations

import math
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
# BGR hijau
_COLOR_DEFECT = (200, 50, 200)       # BGR ungu/magenta
_COLOR_COMPONENT = (0, 255, 0)       # BGR hijau untuk component ROI
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


def _draw_roi_box_rotated(frame, x: int, y: int, w: int, h: int,
                          rotation_deg: float, color, label: str) -> None:
    cx = x + w / 2.0
    cy = y + h / 2.0
    angle = math.radians(rotation_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    corners_rel = [
        (-w / 2, -h / 2), (w / 2, -h / 2),
        (w / 2,  h / 2),  (-w / 2,  h / 2),
    ]
    corners = []
    for dx, dy in corners_rel:
        px = int(cx + dx * cos_a - dy * sin_a)
        py = int(cy + dx * sin_a + dy * cos_a)
        corners.append((px, py))
    for i in range(4):
        cv2.line(frame, corners[i], corners[(i + 1) % 4],
                 color, 2, cv2.LINE_AA)
    cv2.putText(frame, label,
                (corners[0][0], max(16, corners[0][1] - 6)),
                _LABEL_FONT, 0.42, color, 1, cv2.LINE_AA)

def _rotated_corners(x: int, y: int, w: int, h: int,
                     rotation_deg: float) -> list[tuple[int, int]]:
    cx = x + w / 2.0
    cy = y + h / 2.0
    angle = math.radians(rotation_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    result = []
    for dx, dy in [(-w/2, -h/2), (w/2, -h/2), (-w/2, h/2), (w/2, h/2)]:
        result.append((int(cx + dx*cos_a - dy*sin_a), int(cy + dx*sin_a + dy*cos_a)))
    return result

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
        self._show_sticker: bool = True
        self._show_part_ready: bool = True
        self._component_rois: list[dict] = []
        self._defect_rois: list[dict] = []
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

        # Rotation slider row
        _rot_row = tk.Frame(self, bg=PANEL_BG)
        _rot_row.grid(row=3, column=0, sticky="w", padx=10, pady=(0, 8))
        tk.Label(_rot_row, text="Rotasi ROI:",
                 bg=PANEL_BG, fg=TEXT_SECONDARY,
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
        self._rotation_var = tk.StringVar(value="0.0")
        from tkinter import ttk
        self._rot_spinbox = ttk.Spinbox(
            _rot_row, from_=-180, to=180, increment=1,
            textvariable=self._rotation_var, width=7,
        )
        self._rot_spinbox.pack(side="left")
        tk.Label(_rot_row, text="°", bg=PANEL_BG,
                 fg=TEXT_SECONDARY,
                 font=("Segoe UI", 9)).pack(side="left", padx=(2, 0))
        self._rotation_var.trace_add(
            "write", lambda *_: self._on_rotation_changed())

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

    # ── Component ROI API ──

    def set_component_rois(self, rois: list[dict]) -> None:
        """Set component ROIs. Each dict: {name, x, y, w, h, rotation}."""
        self._component_rois = []
        for r in (rois or []):
            nr = self._normalize_roi(r)
            nr["name"] = r.get("name", "ROI")
            self._component_rois.append(nr)
        if self._active_roi_kind and self._active_roi_kind.startswith("component:"):
            idx = int(self._active_roi_kind.split(":")[1])
            if idx >= len(self._component_rois):
                self._active_roi_kind = None
        self.redraw()

    def add_component_roi(self, name: str = "ROI") -> int:
        """Add a default component ROI. Returns index."""
        self._component_rois.append({"name": name, "x": 0.1, "y": 0.1, "w": 0.3, "h": 0.3, "rotation": 0.0})
        self.redraw()
        return len(self._component_rois) - 1

    def remove_component_roi(self, idx: int) -> None:
        if 0 <= idx < len(self._component_rois):
            self._component_rois.pop(idx)
            self.redraw()

    def get_component_rois(self) -> list[dict]:
        return [dict(r) for r in self._component_rois]

    # ── Defect ROI API ──

    def set_defect_rois(self, rois: list[dict]) -> None:
        """Set defect ROIs. Each dict: {name, x, y, w, h, rotation}."""
        self._defect_rois = []
        for r in (rois or []):
            nr = self._normalize_roi(r)
            nr["name"] = r.get("name", "ROI")
            self._defect_rois.append(nr)
        if self._active_roi_kind and self._active_roi_kind.startswith("defect:"):
            idx = int(self._active_roi_kind.split(":")[1])
            if idx >= len(self._defect_rois):
                self._active_roi_kind = None
        self.redraw()

    def add_defect_roi(self, name: str = "ROI") -> int:
        """Add a default defect ROI. Returns index."""
        self._defect_rois.append({"name": name, "x": 0.1, "y": 0.1, "w": 0.3, "h": 0.3, "rotation": 0.0})
        self.redraw()
        return len(self._defect_rois) - 1

    def remove_defect_roi(self, idx: int) -> None:
        if 0 <= idx < len(self._defect_rois):
            self._defect_rois.pop(idx)
            self.redraw()

    def get_defect_rois(self) -> list[dict]:
        return [dict(r) for r in self._defect_rois]

    def set_sticker_visible(self, visible: bool) -> None:
        """Show/hide sticker ROI (used for component_count mode)."""
        self._show_sticker = visible
        self.redraw()

    def set_part_ready_visible(self, visible: bool) -> None:
        """Show/hide part ready ROI (used for component_count mode)."""
        self._show_part_ready = visible
        self.redraw()

    def _hit_test_component(self, x: int, y: int) -> int:
        """Return component ROI index at (x, y), or -1."""
        for i, roi in enumerate(reversed(self._component_rois)):
            roi_idx = len(self._component_rois) - 1 - i
            rx, ry = self._denormalize_point(roi, (x, y))
            if 0 <= rx <= 1 and 0 <= ry <= 1:
                return roi_idx
        return -1

    def set_active_roi(self, kind: str | None) -> None:
        _valid = {"part_ready", "sticker", None}
        if kind is not None and kind.startswith("component:"):
            try:
                idx = int(kind.split(":")[1])
                if 0 <= idx < len(getattr(self, "_component_rois", [])):
                    _valid.add(kind)
            except (ValueError, IndexError):
                pass
        if kind is not None and kind.startswith("defect:"):
            try:
                idx = int(kind.split(":")[1])
                if 0 <= idx < len(getattr(self, "_defect_rois", [])):
                    _valid.add(kind)
            except (ValueError, IndexError):
                pass
        if kind not in _valid:
            raise ValueError("kind must be 'part_ready', 'sticker', 'component:<idx>', 'defect:<idx>', or None")
        self._active_roi_kind = kind
        if kind is not None:
            rot = self._roi_for_kind(kind).get("rotation", 0.0)
            self._rotation_var.set(str(round(float(rot), 2)))
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

        if self._part_ready_roi and self._show_part_ready:
            rx, ry, rw, rh = _roi_px(self._part_ready_roi)
            _rot = float(self._part_ready_roi.get("rotation", 0))
            if abs(_rot) > 0.1:
                _draw_roi_box_rotated(canvas_frame, rx, ry, rw, rh,
                                      _rot, _COLOR_PART_READY, "Part Ready ROI")
            else:
                _draw_roi_box(canvas_frame, rx, ry, rw, rh,
                              _COLOR_PART_READY, "Part Ready ROI")
            if self._active_roi_kind == "part_ready":
                if abs(_rot) > 0.1:
                    for px, py in _rotated_corners(rx, ry, rw, rh, _rot):
                        cv2.rectangle(canvas_frame, (px-5, py-5), (px+5, py+5),
                                      _COLOR_PART_READY, -1)
                        cv2.rectangle(canvas_frame, (px-5, py-5), (px+5, py+5),
                                      (255, 255, 255), 1)
                else:
                    self._draw_handles(canvas_frame, rx, ry, rw, rh, _COLOR_PART_READY)

        if self._sticker_roi and self._show_sticker:
            rx, ry, rw, rh = _roi_px(self._sticker_roi)
            _rot = float(self._sticker_roi.get("rotation", 0))
            if abs(_rot) > 0.1:
                _draw_roi_box_rotated(canvas_frame, rx, ry, rw, rh,
                                      _rot, _COLOR_STICKER, "Sticker ROI")
            else:
                _draw_roi_box(canvas_frame, rx, ry, rw, rh,
                              _COLOR_STICKER, "Sticker ROI")
            if self._active_roi_kind == "sticker":
                if abs(_rot) > 0.1:
                    for px, py in _rotated_corners(rx, ry, rw, rh, _rot):
                        cv2.rectangle(canvas_frame, (px-5, py-5), (px+5, py+5),
                                      _COLOR_STICKER, -1)
                        cv2.rectangle(canvas_frame, (px-5, py-5), (px+5, py+5),
                                      (255, 255, 255), 1)
                else:
                    self._draw_handles(canvas_frame, rx, ry, rw, rh, _COLOR_STICKER)
            exp_px = rx + int(self._cx * rw)
            exp_py = ry + int(self._cy * rh)
            _draw_crosshair(canvas_frame, exp_px, exp_py, _COLOR_CROSSHAIR, "EXP CTR")

        # Draw component ROIs
        for ci, comp_roi in enumerate(getattr(self, '_component_rois', [])):
            if not comp_roi:
                continue
            rx, ry, rw, rh = _roi_px(comp_roi)
            _rot = float(comp_roi.get("rotation", 0))
            _name = comp_roi.get("name", f"ROI {ci}")
            _color = _COLOR_COMPONENT
            if abs(_rot) > 0.1:
                _draw_roi_box_rotated(canvas_frame, rx, ry, rw, rh, _rot, _color, _name)
            else:
                _draw_roi_box(canvas_frame, rx, ry, rw, rh, _color, _name)
            if self._active_roi_kind == f"component:{ci}":
                if abs(_rot) > 0.1:
                    for px, py in _rotated_corners(rx, ry, rw, rh, _rot):
                        cv2.rectangle(canvas_frame, (px-5, py-5), (px+5, py+5), _color, -1)
                        cv2.rectangle(canvas_frame, (px-5, py-5), (px+5, py+5), (255, 255, 255), 1)
                else:
                    self._draw_handles(canvas_frame, rx, ry, rw, rh, _color)

        # Draw defect ROIs
        for di, def_roi in enumerate(getattr(self, '_defect_rois', [])):
            if not def_roi:
                continue
            rx, ry, rw, rh = _roi_px(def_roi)
            _rot = float(def_roi.get("rotation", 0))
            _name = def_roi.get("name", f"ROI {di}")
            _color = _COLOR_DEFECT
            if abs(_rot) > 0.1:
                _draw_roi_box_rotated(canvas_frame, rx, ry, rw, rh, _rot, _color, _name)
            else:
                _draw_roi_box(canvas_frame, rx, ry, rw, rh, _color, _name)
            if self._active_roi_kind == f"defect:{di}":
                if abs(_rot) > 0.1:
                    for px, py in _rotated_corners(rx, ry, rw, rh, _rot):
                        cv2.rectangle(canvas_frame, (px-5, py-5), (px+5, py+5), _color, -1)
                        cv2.rectangle(canvas_frame, (px-5, py-5), (px+5, py+5), (255, 255, 255), 1)
                else:
                    self._draw_handles(canvas_frame, rx, ry, rw, rh, _color)

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
        if kind is not None and kind.startswith("component:"):
            try:
                idx = int(kind.split(":")[1])
                if 0 <= idx < len(self._component_rois):
                    return self._component_rois[idx]
            except (ValueError, IndexError):
                pass
        if kind is not None and kind.startswith("defect:"):
            try:
                idx = int(kind.split(":")[1])
                if 0 <= idx < len(self._defect_rois):
                    return self._defect_rois[idx]
            except (ValueError, IndexError):
                pass
        return {}

    def _set_roi_for_kind(self, kind: str, roi: dict, *, notify: bool = True) -> None:
        normalized = self._normalize_roi(roi)
        if kind == "part_ready":
            self._part_ready_roi = normalized
        elif kind == "sticker":
            self._sticker_roi = normalized
        elif kind is not None and kind.startswith("component:"):
            try:
                idx = int(kind.split(":")[1])
                if 0 <= idx < len(self._component_rois):
                    existing_name = self._component_rois[idx].get("name", "")
                    self._component_rois[idx] = {**normalized, "name": existing_name}
                else:
                    return
            except (ValueError, IndexError):
                return
        elif kind is not None and kind.startswith("defect:"):
            try:
                idx = int(kind.split(":")[1])
                if 0 <= idx < len(self._defect_rois):
                    existing_name = self._defect_rois[idx].get("name", "")
                    self._defect_rois[idx] = {**normalized, "name": existing_name}
                else:
                    return
            except (ValueError, IndexError):
                return
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
        return {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4),
                "rotation": round(self._to_float(roi.get("rotation"), 0.0), 2)}

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
        rotation = float(roi.get("rotation", 0.0))
        if abs(rotation) > 0.1:
            # Rotated: check handles at rotated corners, move inside rotated bbox
            corners = _rotated_corners(x, y, w, h, rotation)
            for name, (hx, hy) in zip(("nw", "ne", "sw", "se"), corners):
                if abs(px - hx) <= 8 and abs(py - hy) <= 8:
                    return name
            # Inside check: transform click ke koordinat un-rotated
            cx, cy = x + w / 2.0, y + h / 2.0
            angle = math.radians(-rotation)
            dx, dy = px - cx, py - cy
            rx = dx * math.cos(angle) - dy * math.sin(angle)
            ry = dx * math.sin(angle) + dy * math.cos(angle)
            if abs(rx) <= w / 2 and abs(ry) <= h / 2:
                return "move"
            return None
        # No rotation: original axis-aligned logic
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
            roi = {"x": left, "y": top, "w": right - left, "h": bottom - top,
                   "rotation": self._drag_start_roi.get("rotation", 0.0)}

        self._set_roi_for_kind(self._active_roi_kind, roi)

    def _on_release(self, _event: tk.Event) -> None:
        self._drag_mode = None
        self._drag_start = None
        self._drag_start_roi = None

    def _on_rotation_changed(self) -> None:
        if self._active_roi_kind is None:
            return
        try:
            rotation = float(self._rotation_var.get() or 0)
        except ValueError:
            return
        roi = dict(self._roi_for_kind(self._active_roi_kind))
        roi["rotation"] = rotation
        self._set_roi_for_kind(self._active_roi_kind, roi, notify=False)

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
