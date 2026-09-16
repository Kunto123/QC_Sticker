from __future__ import annotations

import tkinter as tk
import weakref
from tkinter import ttk

import customtkinter as ctk

from client_tk.app.theme import ACCENT, ACCENT_HOVER, APP_BG

_SCROLLABLE_FRAMES: weakref.WeakSet["ScrollableFrame"] = weakref.WeakSet()
_SCROLL_DISPATCH_BOUND = False
_SCROLLED_EVENT = "<<ScrollableFrameScrolled>>"


def _dispatch_mousewheel(event) -> None:
    widget = getattr(event, "widget", None)
    if widget is None:
        return

    widget_path = str(widget)
    matching_frames = []
    for frame in list(_SCROLLABLE_FRAMES):
        if _frame_is_alive(frame) and frame._contains_widget_path(widget_path):
            matching_frames.append(frame)
    if not matching_frames:
        return

    matching_frames.sort(key=lambda frame: len(str(frame.body)), reverse=True)
    for frame in matching_frames:
        if frame._scroll_from_event(event):
            break


def _frame_is_alive(frame: "ScrollableFrame") -> bool:
    try:
        return bool(frame.winfo_exists())
    except tk.TclError:
        return False


class AutoHideScrollbar(ctk.CTkScrollbar):
    def __init__(self, master, **kwargs):
        orientation = kwargs.pop("orientation", kwargs.pop("orient", "vertical"))
        super().__init__(
            master,
            orientation=orientation,
            fg_color=APP_BG,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            corner_radius=8,
            width=12,
            **kwargs,
        )
        self._grid_kwargs: dict | None = None
        self._pack_kwargs: dict | None = None

    def grid(self, **kwargs):
        self._grid_kwargs = dict(kwargs)
        self._pack_kwargs = None
        return super().grid(**kwargs)

    def pack(self, **kwargs):
        self._pack_kwargs = dict(kwargs)
        self._grid_kwargs = None
        return super().pack(**kwargs)

    def set(self, first, last):
        super().set(first, last)


class ScrollableFrame(ttk.Frame):
    def __init__(self, master, *, padding: int = 0):
        super().__init__(master, padding=padding)

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, background=APP_BG)
        self.v_scrollbar = AutoHideScrollbar(self, orientation="vertical", command=self._on_scrollbar_scroll)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)

        self.body = tk.Frame(self, bg=APP_BG, highlightthickness=0, borderwidth=0)
        self._window_id = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        _SCROLLABLE_FRAMES.add(self)
        self._scrolled_event_job: str | None = None

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")

        self.body.bind("<Configure>", self._sync_scroll_region)
        self.canvas.bind("<Configure>", self._sync_body_width)
        global _SCROLL_DISPATCH_BOUND
        if not _SCROLL_DISPATCH_BOUND:
            self.winfo_toplevel().bind_all("<MouseWheel>", _dispatch_mousewheel, add="+")
            self.winfo_toplevel().bind_all("<Button-4>", _dispatch_mousewheel, add="+")
            self.winfo_toplevel().bind_all("<Button-5>", _dispatch_mousewheel, add="+")
            _SCROLL_DISPATCH_BOUND = True

    def _sync_scroll_region(self, _event=None) -> None:
        try:
            if not _frame_is_alive(self):
                return
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        except tk.TclError:
            return

    def _sync_body_width(self, event) -> None:
        try:
            if not _frame_is_alive(self):
                return
            self.canvas.itemconfigure(self._window_id, width=event.width)
            self._emit_scrolled_event()
        except tk.TclError:
            return

    def _on_scrollbar_scroll(self, *args) -> None:
        try:
            self.canvas.yview(*args)
            self._emit_scrolled_event()
        except tk.TclError:
            return

    def _emit_scrolled_event(self) -> None:
        if self._scrolled_event_job is not None or not _frame_is_alive(self):
            return
        try:
            self._scrolled_event_job = self.after(16, self._flush_scrolled_event)
        except tk.TclError:
            self._scrolled_event_job = None

    def _flush_scrolled_event(self) -> None:
        self._scrolled_event_job = None
        if not _frame_is_alive(self):
            return
        try:
            self.event_generate(_SCROLLED_EVENT, when="tail")
        except tk.TclError:
            return

    def _contains_widget_path(self, widget_path: str) -> bool:
        body_path = str(self.body)
        return widget_path == body_path or widget_path.startswith(f"{body_path}.")

    def _scroll_from_event(self, event) -> bool:
        if not self.v_scrollbar.winfo_manager():
            return False

        direction = 0
        event_num = getattr(event, "num", None)
        if event_num == 4:
            direction = -1
        elif event_num == 5:
            direction = 1
        else:
            delta = int(getattr(event, "delta", 0) or 0)
            if delta == 0:
                return False
            direction = -1 if delta > 0 else 1

        first, last = self.canvas.yview()
        if direction < 0 and first <= 0.0:
            return False
        if direction > 0 and last >= 1.0:
            return False

        self.canvas.yview_scroll(direction, "units")
        self._emit_scrolled_event()
        return True
