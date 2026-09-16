from __future__ import annotations

import base64
import tkinter as tk

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

from client_tk.app.theme import BORDER, PANEL_BG, TEXT_PRIMARY, TEXT_SECONDARY


class LiveView(ctk.CTkFrame):
    def __init__(self, master, title: str, *, size: tuple[int, int] | None = None):
        super().__init__(master, fg_color=PANEL_BG, corner_radius=16, border_width=1, border_color=BORDER)
        self._size = size or (360, 240)
        self._source_frame: np.ndarray | None = None
        self._redraw_job: str | None = None
        if size is not None:
            self.configure(width=size[0], height=size[1])
            self.pack_propagate(False)
            self.grid_propagate(False)
        self._title = ctk.CTkLabel(
            self,
            text=title,
            anchor="w",
            font=("Segoe UI", 12, "bold"),
            text_color=TEXT_PRIMARY,
        )
        self._title.pack(fill="x", padx=12, pady=(10, 0))
        self._label = ctk.CTkLabel(
            self,
            text="No frame",
            fg_color="#0f172a",
            text_color=TEXT_SECONDARY,
            anchor="center",
            font=("Segoe UI", 11),
        )
        self._label.pack(fill="both", expand=True, padx=10, pady=10)
        self.bind("<Configure>", self._queue_redraw, add="+")
        self._label.bind("<Configure>", self._queue_redraw, add="+")
        self._photo = None

    def update_bgr(self, frame) -> None:
        if frame is None:
            return
        if not self.winfo_exists() or not self._label.winfo_exists():
            return
        self._source_frame = frame.copy()
        self._render_source_frame()

    def _queue_redraw(self, _event=None) -> None:
        if self._source_frame is None or self._redraw_job is not None:
            return
        try:
            self._redraw_job = self.after_idle(self._render_source_frame)
        except tk.TclError:
            self._redraw_job = None

    def _render_source_frame(self) -> None:
        self._redraw_job = None
        if self._source_frame is None:
            return
        if not self.winfo_exists() or not self._label.winfo_exists():
            return
        target_width = self._label.winfo_width()
        target_height = self._label.winfo_height()
        if target_width <= 8:
            target_width = self._size[0]
        if target_height <= 8:
            target_height = self._size[1]
        src_h, src_w = self._source_frame.shape[:2]
        scale = min(target_width / max(1, src_w), target_height / max(1, src_h))
        if scale <= 0:
            return
        render_w = max(1, int(round(src_w * scale)))
        render_h = max(1, int(round(src_h * scale)))
        render_frame = self._source_frame
        if render_w != src_w or render_h != src_h:
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            render_frame = cv2.resize(render_frame, (render_w, render_h), interpolation=interpolation)
        rgb = cv2.cvtColor(render_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self._photo = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
        try:
            # Clear stale tk image handle first so text update does not fail on orphan pyimage refs.
            if hasattr(self._label, "_label"):
                self._label._label.configure(image="")
            self._label.configure(image=self._photo)
            self._label.configure(text="", fg_color=PANEL_BG)
        except tk.TclError:
            return

    def update_b64(self, image_b64: str | None) -> None:
        if not image_b64:
            return
        raw = base64.b64decode(image_b64)
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            self.update_bgr(frame)

    def reset(self) -> None:
        self._source_frame = None
        self._photo = None
        if not self.winfo_exists() or not self._label.winfo_exists():
            return
        try:
            if hasattr(self._label, "_label"):
                self._label._label.configure(image="")
            self._label.configure(image=None, text="No frame", fg_color="#0f172a")
        except tk.TclError:
            return
