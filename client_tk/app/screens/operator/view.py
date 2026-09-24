from __future__ import annotations

import base64
import os as _os
import platform
import threading
from time import monotonic, time
import tkinter as tk
import uuid
from tkinter import messagebox, ttk

import customtkinter as ctk
import cv2
import numpy as np

from client_tk.app.components.async_bridge import run_async
from client_tk.app.components.counter_panel import BREAKDOWN_ORDER, CounterPanel
from client_tk.app.components.datapart_guard_popup import DatapartGuardPopup
from client_tk.app.components.live_view import LiveView
from client_tk.app.components.result_panel import ResultPanel
from client_tk.app.components.scrollable_frame import ScrollableFrame
from client_tk.app.config import (
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_OPERATOR_PREVIEW_FPS,
    DEFAULT_UPLOAD_INTERVAL_MS,
)
from client_tk.app.services.camera_capture import CameraCaptureService
from client_tk.app.services.frame_upload import FrameUploadService
from client_tk.app.theme import APP_BG, ACCENT_SOFT, BORDER, PANEL_ALT_BG, PANEL_BG, TEXT_PRIMARY, TEXT_SECONDARY, ACCENT, ACCENT_HOVER, TEXT_ON_ACCENT

import logging
_logger = logging.getLogger(__name__)


BADGE_COLORS = {
    "neutral": ("#475569", "#f8fafc"),
    "success": ("#166534", "#f0fdf4"),
    "danger": ("#991b1b", "#fef2f2"),
    "warning": ("#b45309", "#fffbeb"),
    "info": ("#1d4ed8", "#eff6ff"),
}
RESPONSIVE_BREAKPOINT = 1240
HEARTBEAT_INTERVAL_MS = 20_000
PLC_POLL_INTERVAL_MS = 2_000
DATAPART_GUARD_POLL_INTERVAL_MS = 3_000


AUTO_START_FRAME_WAIT_MS = 150
AUTO_START_MAX_ATTEMPTS = 40


class OperatorScreen(ctk.CTkFrame):
    def __init__(self, master, api_client, session_state):
        super().__init__(master, fg_color=APP_BG, corner_radius=0)
        self.api = api_client
        self.state = session_state
        self.capture = CameraCaptureService()
        self.uploader = FrameUploadService()  # Local frame pump via API client bridge.
        self._latest_payload: dict | None = None
        self._last_detections: list = []
        self._last_detections_ts: float = 0.0
        self._detection_holdover_s: float = 1.5
        self._latest_error: str | None = None
        self._lock = threading.Lock()
        self._after_id: str | None = None
        self._preview_after_id: str | None = None
        self._preview_interval_ms = max(15, int(round(1000.0 / DEFAULT_OPERATOR_PREVIEW_FPS)))
        self._heartbeat_after_id: str | None = None
        self._plc_poll_after_id: str | None = None
        self._closed = False
        self._machine_id = f"{platform.node() or 'workstation'}-{uuid.getnode():012x}"
        self._template_lookup: dict[str, dict] = {}
        self._template_detail_lookup: dict[int, dict] = {}
        self._template_version_detail_lookup: dict[int, dict] = {}
        self._is_compact_layout: bool | None = None
        self._is_preview_compact: bool | None = None
        self._auth_error_notified = False
        self._auto_start_done = False
        self._auto_start_after_id: str | None = None
        self._resize_debounce_after_id: str | None = None
        self._cached_overlay_frame: np.ndarray | None = None
        self._last_payload_seq: int = 0
        self._last_rendered_payload_seq: int = 0
        self._overlay_render_after_id: str | None = None
        self._overlay_render_interval_ms: int = 100  # throttle: max ~10 fps render
        self._overlay_thread: threading.Thread | None = None
        self._overlay_pending_frame = None
        self._overlay_pending_payload = None
        self._overlay_ready = threading.Event()
        self._skip_overlay_count: int = 0
        self._last_plc_template_cycle_event_id: int | None = None
        self._latest_plc_status: dict | None = None
        self._inference_running = False
        self._inference_thread: threading.Thread | None = None
        self._datapart_guard_poll_after_id: str | None = None
        self._datapart_guard_popup: DatapartGuardPopup | None = None

        self.camera_value = tk.StringVar(value="0")
        self.template_version_value = tk.StringVar()
        self.part_ready_roi_x_value = tk.StringVar(value="0.2")
        self.part_ready_roi_y_value = tk.StringVar(value="0.2")
        self.part_ready_roi_w_value = tk.StringVar(value="0.25")
        self.part_ready_roi_h_value = tk.StringVar(value="0.25")
        self.sticker_roi_x_value = tk.StringVar(value="0.2")
        self.sticker_roi_y_value = tk.StringVar(value="0.2")
        self.sticker_roi_w_value = tk.StringVar(value="0.6")
        self.sticker_roi_h_value = tk.StringVar(value="0.6")
        self.template_choice = tk.StringVar()
        self.display_source = tk.StringVar(value="Right View: Live Camera + ROIs")

        self.operator_context = tk.StringVar(value=f"Operator: {self.state.user.get('username') if self.state.user else '-'}")
        self.template_context = tk.StringVar(value="Template: -")
        self.active_preset_var = tk.StringVar(value="-")
        self.info_var = tk.StringVar(value="Idle. Pilih template atau deployment, lalu start camera.")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        self._build_top_bar()
        self._build_status_strip()
        self._build_content()
        self._load_template_choices()
        self._refresh_context_summary()
        self._update_status_badges()
        self.bind("<Configure>", self._on_resize)
        self.after_idle(self._apply_responsive_layout)
        # Preview frames are rendered by a dedicated camera tick; inference
        # results only update status/overlay payload.
        self._schedule_heartbeat(delay_ms=1_000)
        self._schedule_plc_poll(delay_ms=4_000)
        self._schedule_datapart_guard_poll(delay_ms=4_000)
        self.after_idle(self._auto_start_first_template)

    def _build_top_bar(self) -> None:
        self.top_bar = ctk.CTkFrame(self, fg_color=APP_BG, corner_radius=0)
        self.top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.top_bar.columnconfigure(0, weight=1)

        self.action_bar = ctk.CTkFrame(self.top_bar, fg_color=APP_BG, corner_radius=0)
        self.action_bar.grid(row=0, column=0, sticky="ew")
        self.action_buttons = [
            ctk.CTkButton(self.action_bar, text="Start", command=self._start_production, fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=TEXT_ON_ACCENT),
            ctk.CTkButton(self.action_bar, text="Stop", command=self._stop_production, fg_color="#7f1d1d", hover_color="#991b1b", text_color="#fef2f2"),
        ]

        self.active_preset_box = ctk.CTkFrame(
            self.top_bar, fg_color=PANEL_BG, corner_radius=14, border_width=1, border_color=BORDER
        )
        ctk.CTkLabel(
            self.active_preset_box,
            text="ACTIVE PRESET",
            font=("Segoe UI", 9, "bold"),
            text_color=TEXT_SECONDARY,
            anchor="center",
        ).pack(fill="x", padx=16, pady=(8, 0))
        self.active_preset_label = ctk.CTkLabel(
            self.active_preset_box,
            textvariable=self.active_preset_var,
            font=("Segoe UI", 22, "bold"),
            text_color="#60a5fa",
            anchor="center",
            justify="center",
            wraplength=420,
        )
        self.active_preset_label.pack(fill="x", padx=16, pady=(0, 8))

        self.template_box = ctk.CTkFrame(self.top_bar, fg_color=PANEL_BG, corner_radius=14, border_width=1, border_color=BORDER)
        self.template_box.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self.template_box, text="Template", font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 6))
        self.template_selector = ttk.Combobox(self.template_box, textvariable=self.template_choice, width=32, state="readonly")
        self.template_selector.grid(row=1, column=0, padx=(10, 6), pady=(0, 10), sticky="ew")
        self.template_selector.bind("<<ComboboxSelected>>", self._on_template_selected)
        ctk.CTkButton(self.template_box, text="Refresh", command=self._load_template_choices, fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=TEXT_ON_ACCENT).grid(row=1, column=1, padx=(0, 10), pady=(0, 10))

        self._layout_top_bar(compact=False)

    def _build_status_strip(self) -> None:
        self.status_frame = ctk.CTkFrame(self, fg_color=PANEL_BG, corner_radius=14, border_width=1, border_color=BORDER)
        self.status_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        # Header row with title + operator/template info
        header_frame = ctk.CTkFrame(self.status_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(10, 4))
        ctk.CTkLabel(header_frame, text="System Status", font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).pack(side="left")
        ctk.CTkLabel(header_frame, textvariable=self.operator_context, font=("Segoe UI", 10, "bold"), text_color=TEXT_SECONDARY).pack(side="right")
        ctk.CTkLabel(header_frame, textvariable=self.template_context, font=("Segoe UI", 10), text_color=TEXT_SECONDARY).pack(side="right", padx=(0, 12))

        self.status_badges_container = ctk.CTkFrame(self.status_frame, fg_color="transparent")
        self.status_badges_container.pack(fill="x", padx=10, pady=(0, 10))
        self.badges: dict[str, ctk.CTkLabel] = {}
        for key in ("EVENT", "PLC"):
            label = ctk.CTkLabel(
                self.status_badges_container,
                text=f"{key}: -",
                fg_color=BADGE_COLORS["neutral"][0],
                text_color=BADGE_COLORS["neutral"][1],
                font=("Segoe UI", 10, "bold"),
                corner_radius=999,
                height=28,
                padx=12,
            )
            self.badges[key] = label
        self._layout_status_strip(compact=False)

    def _build_content(self) -> None:
        self.content_scroller = ScrollableFrame(self)
        self.content_scroller.grid(row=2, column=0, sticky="nsew")
        self.content = self.content_scroller.body
        self.content.columnconfigure(0, weight=1)
        self.content.rowconfigure(0, weight=1)
        self.content.rowconfigure(1, weight=0)

        self.live_container = ttk.Frame(self.content)
        self.live_container.columnconfigure(0, weight=1)
        self.live_container.rowconfigure(0, weight=1)
        self.live_container.rowconfigure(1, weight=0)

        self.preview_strip = ttk.Frame(self.live_container)
        self.preview_strip.grid(row=0, column=0, sticky="nsew")
        self.preview_strip.columnconfigure(0, weight=1)
        self.preview_strip.rowconfigure(0, weight=1)

        self.main_view = LiveView(self.preview_strip, "ROI / ML Overlay", size=(1000, 560))
        self.main_view.grid(row=0, column=0, sticky="nsew")

        live_footer = ttk.Frame(self.live_container)
        live_footer.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        live_footer.columnconfigure(0, weight=1)
        ttk.Label(live_footer, textvariable=self.display_source, foreground="#475569").grid(row=0, column=0, sticky="w")
        ttk.Label(live_footer, textvariable=self.info_var, foreground="#475569", wraplength=700, justify="left").grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )

        self.sidebar_container = ttk.Frame(self.content, width=360)
        self.sidebar_container.grid_propagate(False)
        self.sidebar_container.columnconfigure(0, weight=1)
        self.sidebar_container.rowconfigure(0, weight=0)  # roi picker: fixed
        self.sidebar_container.rowconfigure(1, weight=0)  # counter: fixed
        self.sidebar_container.rowconfigure(2, weight=0)  # decision banner: fixed
        self.sidebar_container.rowconfigure(0, weight=0)  # counter: fixed
        self.sidebar_container.rowconfigure(1, weight=0)  # decision banner: fixed
        self.sidebar_container.rowconfigure(2, weight=1)  # scroll: fills rest

        # Fixed counter strip — always visible, outside scroll area
        self.counter_panel = CounterPanel(self.sidebar_container)
        self.counter_panel.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        # Decision status banner — fixed below counter, outside scroll area
        self.decision_status_frame = ctk.CTkFrame(
            self.sidebar_container, fg_color=PANEL_BG, corner_radius=16, border_width=1, border_color=BORDER
        )
        self.decision_status_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self.decision_banner = ctk.CTkLabel(
            self.decision_status_frame,
            text="WAITING",
            fg_color="#334155",
            text_color="#f8fafc",
            font=("Segoe UI", 24, "bold"),
            corner_radius=14,
            anchor="center",
            wraplength=320,
            justify="center",
        )
        self.decision_banner.pack(fill="x", padx=12, pady=(12, 6))
        self.decision_subtitle = ctk.CTkLabel(
            self.decision_status_frame,
            text="Menunggu event inspeksi pertama.",
            font=("Segoe UI", 10),
            wraplength=320,
            justify="left",
            text_color=TEXT_SECONDARY,
        )
        self.decision_subtitle.pack(anchor="w", padx=12, pady=(0, 6))

        self.plc_status_label = ctk.CTkLabel(
            self.decision_status_frame,
            text="",
            font=("Segoe UI", 9),
            text_color=TEXT_SECONDARY,
            wraplength=320,
            justify="left",
        )
        self.plc_status_label.pack(anchor="w", padx=12, pady=(0, 10))

        # Camera status label
        self.camera_status_label = ctk.CTkLabel(
            self.decision_status_frame,
            text="● Kamera",
            font=("Segoe UI", 9),
            text_color=TEXT_SECONDARY,
            wraplength=320,
            justify="left",
        )
        self.camera_status_label.pack(anchor="w", padx=12, pady=(0, 10))

        self.sidebar_scroller = ScrollableFrame(self.sidebar_container)
        self.sidebar_scroller.grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.sidebar_inner = self.sidebar_scroller.body
        self.sidebar_inner.columnconfigure(0, weight=1)

        # Reject breakdown — at top of scrollable area
        breakdown_frame = ctk.CTkFrame(
            self.sidebar_inner, fg_color=PANEL_BG, corner_radius=16, border_width=1, border_color=BORDER
        )
        breakdown_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(breakdown_frame, text="Reject Breakdown", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).pack(
            anchor="w", padx=12, pady=(12, 6)
        )
        breakdown_body = ctk.CTkFrame(breakdown_frame, fg_color=PANEL_ALT_BG, corner_radius=12, border_width=1, border_color=BORDER)
        breakdown_body.pack(fill="x", padx=12, pady=(0, 12))
        breakdown_body.grid_columnconfigure(1, weight=1)
        self.breakdown_labels: dict[str, ctk.CTkLabel] = {}
        for _idx, _key in enumerate(BREAKDOWN_ORDER):
            ctk.CTkLabel(breakdown_body, text=f"{_key}:", text_color=TEXT_PRIMARY).grid(
                row=_idx, column=0, sticky="w", padx=10, pady=3
            )
            _lbl = ctk.CTkLabel(breakdown_body, text="0", text_color=TEXT_SECONDARY)
            _lbl.grid(row=_idx, column=1, sticky="e", padx=10, pady=3)
            self.breakdown_labels[_key] = _lbl

        self.result_panel = ResultPanel(self.sidebar_inner)
        self.result_panel.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        events_frame = ctk.CTkFrame(self.sidebar_inner, fg_color=PANEL_BG, corner_radius=16, border_width=1, border_color=BORDER)
        events_frame.grid(row=2, column=0, sticky="nsew")
        self.sidebar_inner.rowconfigure(2, weight=1)
        ctk.CTkLabel(events_frame, text="Recent Events", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).pack(
            anchor="w", padx=12, pady=(12, 6)
        )
        list_wrapper = ctk.CTkFrame(events_frame, fg_color=PANEL_ALT_BG, corner_radius=10, border_width=1, border_color=BORDER)
        list_wrapper.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.recent_list = tk.Listbox(
            list_wrapper,
            height=12,
            bg=PANEL_ALT_BG,
            fg=TEXT_PRIMARY,
            selectbackground=ACCENT_SOFT,
            selectforeground=TEXT_ON_ACCENT,
            borderwidth=0,
            relief="flat",
            font=("Segoe UI", 9),
            highlightthickness=0,
        )
        self.recent_list.pack(fill="both", expand=True, padx=4, pady=4)

    def _apply_responsive_layout(self) -> None:
        if self._closed or not self.winfo_exists():
            return
        try:
            width = max(self.winfo_width(), self.winfo_toplevel().winfo_width())
            compact = width < RESPONSIVE_BREAKPOINT
            self._layout_top_bar(compact=compact)
            self._layout_status_strip(compact=compact)
        except tk.TclError:
            return
        if compact != self._is_compact_layout:
            self._is_compact_layout = compact

            self.live_container.grid_forget()
            self.sidebar_container.grid_forget()
            self.content.columnconfigure(0, weight=1)
            self.content.columnconfigure(1, weight=0)
            self.content.rowconfigure(0, weight=1)
            self.content.rowconfigure(1, weight=0)

            if compact:
                self.live_container.grid(row=0, column=0, sticky="nsew")
                self.sidebar_container.grid(row=1, column=0, sticky="ew", pady=(10, 0))
                self.sidebar_container.configure(width=0, height=300)
            else:
                self.content.columnconfigure(0, weight=1)
                self.content.columnconfigure(1, weight=0)
                self.live_container.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
                self.sidebar_container.grid(row=0, column=1, sticky="ns")
                self.sidebar_container.configure(width=380, height=1)

        self._apply_preview_layout()

    def _apply_preview_layout(self) -> None:
        if not self.winfo_exists() or not self.preview_strip.winfo_exists() or not self.main_view.winfo_exists():
            return
        self.preview_strip.columnconfigure(0, weight=1)
        self.preview_strip.rowconfigure(0, weight=1)
        self.main_view.grid(row=0, column=0, sticky="nsew")

    def _on_resize(self, _event=None) -> None:
        # Debounce: cancel pending layout, schedule new one 200ms later.
        # Prevents flicker from rapid successive resize events.
        if self._resize_debounce_after_id:
            try:
                self.after_cancel(self._resize_debounce_after_id)
            except tk.TclError:
                pass
        self._resize_debounce_after_id = self.after(200, self._apply_responsive_layout)

    def _clear_resize_debounce(self) -> None:
        """Cancel any pending resize debounce callback."""
        if self._resize_debounce_after_id:
            try:
                self.after_cancel(self._resize_debounce_after_id)
            except tk.TclError:
                pass
            self._resize_debounce_after_id = None

    def _layout_top_bar(self, *, compact: bool) -> None:
        for widget in self.action_bar.grid_slaves():
            widget.grid_forget()

        for column in range(6):
            self.action_bar.columnconfigure(column, weight=1 if compact else 0)
        self.action_bar.rowconfigure(0, weight=1)
        self.action_bar.rowconfigure(1, weight=1)

        self.top_bar.columnconfigure(0, weight=1 if compact else 0)
        self.top_bar.columnconfigure(1, weight=1)
        self.top_bar.columnconfigure(2, weight=0)

        if compact:
            self.action_bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
            self.active_preset_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
            self.template_box.grid(row=2, column=0, sticky="ew")
            self.template_box.columnconfigure(0, weight=1)
            self.template_box.columnconfigure(1, weight=0)
            for index, button in enumerate(self.action_buttons):
                row = 0 if index < 3 else 1
                column = index if index < 3 else index - 3
                button.grid(row=row, column=column, sticky="ew", padx=3, pady=3)
        else:
            self.action_bar.grid(row=0, column=0, sticky="w")
            self.active_preset_box.grid(row=0, column=1, sticky="ew", padx=12)
            self.template_box.grid(row=0, column=2, sticky="e")
            for index, button in enumerate(self.action_buttons):
                button.grid(row=0, column=index, sticky="w", padx=(0 if index == 0 else 6, 0))

    def _layout_status_strip(self, *, compact: bool) -> None:
        for widget in self.status_badges_container.grid_slaves():
            widget.grid_forget()

        keys = ["EVENT", "PLC"]
        columns = 2 if compact else 2
        rows = 1
        for column in range(columns):
            self.status_badges_container.columnconfigure(column, weight=1)
        for row in range(rows):
            self.status_badges_container.rowconfigure(row, weight=1)

        for index, key in enumerate(keys):
            row = index // columns
            column = index % columns
            self.badges[key].grid(row=row, column=column, sticky="ew", padx=4, pady=3)

    def _load_template_choices(self) -> None:
        try:
            items = self.api.list_templates()
        except Exception as exc:  # noqa: BLE001
            self.info_var.set(OperatorScreen._friendly_error(exc, "Gagal memuat template"))
            return
        values: list[str] = []
        self._template_lookup = {}
        self._template_detail_lookup = {}
        self._template_version_detail_lookup = {}
        for item in items:
            # Only hide explicitly inactive/retired templates.
            # Templates without lifecycle_status (legacy) are treated as active.
            is_active = bool(item.get("is_active", True))
            if not is_active:
                continue
            lifecycle = str(item.get("lifecycle_status") or "").strip().lower()
            if lifecycle in ("draft", "review", "retired"):
                continue
            label = f"{item['name']} | v{item.get('version_number')} | version_id={item.get('version_id')}"
            values.append(label)
            self._template_lookup[label] = item
        self.template_selector.configure(values=values)
        self._sync_template_selector()

    def _auto_start_first_template(self) -> None:
        if self._closed or self._auto_start_done:
            return
        self._auto_start_done = True
        values = list(self.template_selector.cget("values") or [])
        if not values:
            self.info_var.set("Tidak ada template tersedia untuk auto start.")
            return
        first_label = str(values[0])
        self.template_choice.set(first_label)
        self.info_var.set("Auto start: memilih template pertama.")
        self._on_template_selected()
        if not self._start_camera(show_errors=False):
            self.info_var.set("Auto start: kamera gagal dibuka. Start Camera bisa dilakukan manual.")
            return
        self._wait_for_auto_start_frame(attempt=0)

    def _wait_for_auto_start_frame(self, *, attempt: int) -> None:
        if self._closed or self.state.active_session:
            return
        if self.capture.get_latest_frame() is not None:
            self.info_var.set("Auto start: frame kamera siap, memulai session.")
            self._start_session()
            return
        if attempt >= AUTO_START_MAX_ATTEMPTS:
            self.info_var.set("Auto start: kamera belum menghasilkan frame. Start Session bisa dilakukan manual.")
            return
        self._auto_start_after_id = self.after(
            AUTO_START_FRAME_WAIT_MS,
            lambda: self._wait_for_auto_start_frame(attempt=attempt + 1),
        )

    def _fetch_template_detail(self, template_id: int) -> dict:
        template_id = int(template_id)
        cached = self._template_detail_lookup.get(template_id)
        if cached:
            return cached
        detail = self.api.get_template(template_id)
        self._template_detail_lookup[template_id] = detail
        return detail

    def _fetch_template_version_detail(self, version_id: int) -> dict:
        version_id = int(version_id)
        cached = self._template_version_detail_lookup.get(version_id)
        if cached:
            return cached
        detail = self.api.get_template_version(version_id)
        self._template_version_detail_lookup[version_id] = detail
        return detail

    def _roi_vars(self, kind: str) -> dict[str, tk.StringVar]:
        if kind == "part_ready":
            return {
                "x": self.part_ready_roi_x_value,
                "y": self.part_ready_roi_y_value,
                "w": self.part_ready_roi_w_value,
                "h": self.part_ready_roi_h_value,
            }
        return {
            "x": self.sticker_roi_x_value,
            "y": self.sticker_roi_y_value,
            "w": self.sticker_roi_w_value,
            "h": self.sticker_roi_h_value,
        }

    def _format_roi_value(self, value) -> str:
        try:
            return f"{float(value):.6g}"
        except (TypeError, ValueError):
            return str(value or "")

    def _set_roi_values(self, kind: str, roi: dict | None) -> None:
        values = self._roi_vars(kind)
        payload = roi or {}
        for key, variable in values.items():
            if key in payload:
                variable.set(self._format_roi_value(payload.get(key)))

    def _float_value(self, value: str, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _resolve_upload_interval_ms(self) -> int:
        fallback = max(50, int(DEFAULT_UPLOAD_INTERVAL_MS))
        detail = self.state.cache.get("selected_template_detail") if isinstance(self.state.cache, dict) else None
        if not isinstance(detail, dict):
            return fallback
        vision = detail.get("vision") if isinstance(detail.get("vision"), dict) else {}
        try:
            inference_fps = float(vision.get("inference_fps") or 0.0)
        except (TypeError, ValueError):
            return fallback
        if inference_fps <= 0.0:
            return fallback
        return max(50, int(round(1000.0 / inference_fps)))

    def _resolve_camera_settings(self) -> dict[str, int | float | None]:
        detail = self.state.cache.get("selected_template_detail") if isinstance(self.state.cache, dict) else None
        camera = detail.get("camera") if isinstance(detail, dict) and isinstance(detail.get("camera"), dict) else {}

        def _positive_int(value, fallback: int) -> int | None:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                parsed = int(fallback)
            return parsed if parsed > 0 else None

        def _positive_float(value, fallback: float) -> float | None:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                parsed = float(fallback)
            return parsed if parsed > 0 else None

        return {
            "width": _positive_int(camera.get("width"), DEFAULT_CAMERA_WIDTH),
            "height": _positive_int(camera.get("height"), DEFAULT_CAMERA_HEIGHT),
            "fps": _positive_float(camera.get("fps"), DEFAULT_CAMERA_FPS),
        }

    def _camera_settings_label(self) -> str:
        actual = self.capture.actual_settings
        width = int(actual.get("width") or 0)
        height = int(actual.get("height") or 0)
        fps = float(actual.get("fps") or 0.0)
        parts: list[str] = []
        if width and height:
            parts.append(f"{width}x{height}")
        if fps > 0:
            parts.append(f"{fps:.0f} fps")
        return " / ".join(parts)

    def _read_roi_payload(self, kind: str) -> dict[str, float]:
        variables = self._roi_vars(kind)
        return {
            key: self._float_value(variable.get(), default=(1.0 if key in {"w", "h"} else 0.0))
            for key, variable in variables.items()
        }

    def _roi_meta_payload(self, kind: str) -> dict[str, float]:
        roi = self._read_roi_payload(kind)
        return {
            "x": roi["x"],
            "y": roi["y"],
            "width": roi["w"],
            "height": roi["h"],
        }

    def _resolve_roi_rect(
        self,
        roi_meta: dict,
        frame_width: int,
        frame_height: int,
        *,
        source_width: int | None = None,
        source_height: int | None = None,
    ) -> tuple[int, int, int, int] | None:
        if not isinstance(roi_meta, dict):
            return None
        try:
            x = float(roi_meta.get("x", 0.0) or 0.0)
            y = float(roi_meta.get("y", 0.0) or 0.0)
            width_value = roi_meta.get("width", roi_meta.get("w", 0.0))
            height_value = roi_meta.get("height", roi_meta.get("h", 0.0))
            width = float(width_value or 0.0)
            height = float(height_value or 0.0)
        except (TypeError, ValueError):
            return None
        if width <= 0.0 or height <= 0.0:
            return None

        normalized = max(abs(x), abs(y), abs(width), abs(height)) <= 1.5
        if normalized:
            px = int(x * frame_width)
            py = int(y * frame_height)
            pw = int(width * frame_width)
            ph = int(height * frame_height)
        else:
            src_w = source_width or frame_width
            src_h = source_height or frame_height
            scale_x = frame_width / src_w if src_w else 1.0
            scale_y = frame_height / src_h if src_h else 1.0
            px = int(x * scale_x)
            py = int(y * scale_y)
            pw = int(width * scale_x)
            ph = int(height * scale_y)

        if pw <= 0 or ph <= 0:
            return None
        return px, py, pw, ph

    def _build_preview_overlay_payload(self, frame) -> dict:
        with self._lock:
            base_payload = dict(self._latest_payload) if isinstance(self._latest_payload, dict) else {}
        client_timings = dict(base_payload.get("client_timings") or {})
        if frame is not None:
            height, width = frame.shape[:2]
            client_timings.setdefault("frame_width", width)
            client_timings.setdefault("frame_height", height)
        return {
            "validation": base_payload.get("validation") or {},
            "part_ready": base_payload.get("part_ready") or {},
            "sticker_detection": base_payload.get("sticker_detection") or {},
            "detections": base_payload.get("detections") or [],
            "event_state": base_payload.get("event_state") or "idle",
            "part_ready_roi_meta": self._roi_meta_payload("part_ready"),
            "sticker_roi_meta": self._roi_meta_payload("sticker"),
            "client_timings": client_timings,
        }

    def _draw_roi_overlays(self, frame, payload: dict):
        if frame is None:
            return None
        overlay = frame.copy()
        disp_h, disp_w = overlay.shape[:2]
        client_timings = payload.get("client_timings") or {}
        sent_w = client_timings.get("frame_width")
        sent_h = client_timings.get("frame_height")

        part_ready_roi = payload.get("part_ready_roi_meta") or {}
        sticker_roi = payload.get("sticker_roi_meta") or {}

        part_ready_box = self._resolve_roi_rect(
            part_ready_roi,
            disp_w,
            disp_h,
            source_width=int(sent_w) if sent_w else None,
            source_height=int(sent_h) if sent_h else None,
        )
        if part_ready_box is not None:
            px, py, pw, ph = part_ready_box
            cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (50, 180, 255), 3)
            cv2.putText(overlay, "Part Ready ROI", (px, max(18, py - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 180, 255), 2, cv2.LINE_AA)

        sticker_box = self._resolve_roi_rect(
            sticker_roi,
            disp_w,
            disp_h,
            source_width=int(sent_w) if sent_w else None,
            source_height=int(sent_h) if sent_h else None,
        )
        if sticker_box is not None:
            sx, sy, sw, sh = sticker_box
            cv2.rectangle(overlay, (sx, sy), (sx + sw, sy + sh), (255, 200, 0), 3)
            cv2.putText(overlay, "Sticker ROI", (sx, max(18, sy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2, cv2.LINE_AA)

        return overlay

    def _validated_roi_payload(self, kind: str) -> dict[str, float]:
        variables = self._roi_vars(kind)
        payload: dict[str, float] = {}
        label = "Part Ready ROI" if kind == "part_ready" else "Sticker ROI"
        for key, variable in variables.items():
            raw = variable.get().strip()
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} field '{key}' harus numerik.") from exc
            if key in {"x", "y"} and not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} field '{key}' harus di rentang 0 sampai 1.")
            if key in {"w", "h"} and not 0.0 < value <= 1.0:
                raise ValueError(f"{label} field '{key}' harus lebih besar dari 0 dan maksimal 1.")
            payload[key] = value
        return payload

    def _build_full_frame_with_roi(self, frame, kind: str, *, label: str, color: tuple[int, int, int]):
        if frame is None:
            return None
        payload = self._read_roi_payload(kind)
        height, width = frame.shape[:2]
        x = max(0, min(width - 1, int(float(payload.get("x", 0.0)) * width)))
        y = max(0, min(height - 1, int(float(payload.get("y", 0.0)) * height)))
        roi_w = max(1, int(float(payload.get("w", 1.0)) * width))
        roi_h = max(1, int(float(payload.get("h", 1.0)) * height))
        x2 = min(width, x + roi_w)
        y2 = min(height, y + roi_h)
        annotated = frame.copy()
        cv2.rectangle(annotated, (x, y), (x2, y2), color, 2)
        cv2.putText(annotated,
            f"{label} | x={x} y={y} w={x2 - x} h={y2 - y}",
            (x, max(22, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return annotated

    def _build_local_detection_overlay(self, frame, payload: dict):
        """Render detection bboxes onto the live camera frame using payload data.

        Replicates the backend overlay composition locally so that the network
        round-trip cost of JPEG-encoding and transmitting the overlay image is
        eliminated entirely (stream response mode).
        """
        if frame is None:
            return None
        overlay = frame.copy()
        validation = payload.get("validation") or {}
        part_ready = payload.get("part_ready") or {}
        sticker_roi = payload.get("sticker_roi_meta") or {}
        part_ready_roi = payload.get("part_ready_roi_meta") or {}
        detections = payload.get("detections") or []
        _stale_detections = False
        if not detections and self._last_detections:
            _age = monotonic() - self._last_detections_ts
            if _age < self._detection_holdover_s:
                detections = self._last_detections
                _stale_detections = True
        event_state = payload.get("event_state") or "idle"

        # Backend computes roi_meta pixel coords from the (possibly downscaled) frame
        # that was uploaded. The display overlay is drawn on the full-resolution live
        # camera frame. Scale all backend pixel coordinates accordingly.
        client_timings = payload.get("client_timings") or {}
        sent_w = client_timings.get("frame_width")
        sent_h = client_timings.get("frame_height")
        disp_h, disp_w = frame.shape[:2]
        scale_x = disp_w / sent_w if sent_w else 1.0
        scale_y = disp_h / sent_h if sent_h else 1.0

        from shared.contracts.enums import DecisionCode
        decision = validation.get("decision")
        reject_reason = validation.get("reject_reason_code") or "OK"
        decision_color = (0, 180, 0) if decision == DecisionCode.ACCEPT.value else (0, 0, 220)

        self._render_sticker_overlay(overlay, frame, payload,
                                      sent_w, sent_h, disp_w, disp_h,
                                      scale_x, scale_y, _stale_detections)
        return overlay

    def _render_sticker_overlay(self, overlay, frame, payload,
                                 sent_w, sent_h, disp_w, disp_h,
                                 scale_x, scale_y, stale_detections) -> None:
        """Render sticker detection (original): part_ready ROI, sticker ROI, bboxes."""
        validation = payload.get("validation") or {}
        part_ready = payload.get("part_ready") or {}
        sticker_roi = payload.get("sticker_roi_meta") or {}
        part_ready_roi = payload.get("part_ready_roi_meta") or {}
        detections = payload.get("detections") or []
        event_state = payload.get("event_state") or "idle"
        decision = validation.get("decision")
        reject_reason = validation.get("reject_reason_code") or "OK"
        decision_color = (0, 180, 0) if decision == "ACCEPT" else (0, 0, 220)

        if stale_detections and not detections:
            detections = self._last_detections
            age = monotonic() - self._last_detections_ts
            if age >= self._detection_holdover_s:
                detections = []

        # Part ready ROI box (blue)
        pr_box = self._resolve_roi_rect(part_ready_roi, disp_w, disp_h,
                                          source_width=sent_w, source_height=sent_h)
        if pr_box:
            px, py, pw, ph = pr_box
            cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (50, 180, 255), 2)

        # Sticker ROI box (yellow)
        st_box = self._resolve_roi_rect(sticker_roi, disp_w, disp_h,
                                          source_width=sent_w, source_height=sent_h)
        sx = sy = sw = sh = 0
        if st_box:
            sx, sy, sw, sh = st_box
            cv2.rectangle(overlay, (sx, sy), (sx + sw, sy + sh), (255, 200, 0), 2)

        # Detection bboxes
        for det in detections:
            pos = det.get("position") or {}
            x1 = int(sx + float(pos.get("x1", 0)) * scale_x)
            y1 = int(sy + float(pos.get("y1", 0)) * scale_y)
            x2 = int(sx + float(pos.get("x2", 0)) * scale_x)
            y2 = int(sy + float(pos.get("y2", 0)) * scale_y)
            bc = (100, 180, 255) if stale_detections else (0, 255, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bc, 2)
            lbl = str(det.get("label") or "")
            conf = float(det.get("confidence") or 0.0)
            cv2.putText(overlay, f"{lbl} {conf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Crosshair
        if sw and sh:
            tmpl = {}; cxr, cyr = 0.5, 0.5
            exp_x, exp_y = int(sx + cxr * sw), int(sy + cyr * sh)
            arm = 18
            cv2.line(overlay, (exp_x - arm, exp_y), (exp_x + arm, exp_y), (0, 220, 255), 2, cv2.LINE_AA)
            cv2.line(overlay, (exp_x, exp_y - arm), (exp_x, exp_y + arm), (0, 220, 255), 2, cv2.LINE_AA)

        # Status text
        cv2.putText(overlay, f"{decision} / {reject_reason}", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, decision_color, 2, cv2.LINE_AA)
        pr_val = part_ready.get("part_ready")
        pr_ratio = part_ready.get("match_ratio", part_ready.get("part_ready_confidence", "-"))
        cv2.putText(overlay, f"part_ready={pr_val} ratio={pr_ratio}", (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"state={event_state}", (12, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def _update_roi_overlay_preview(self, frame) -> None:
        """Draw ROI boxes on live frame even before session starts."""
        if frame is None:
            self.main_view.reset()
            return
        
        annotated = self._build_full_frame_with_roi(
            frame.copy(), "part_ready", label="Part Ready ROI", color=(50, 180, 255)
        )
        if annotated is not None:
            annotated = self._build_full_frame_with_roi(
                annotated, "sticker", label="Sticker ROI", color=(255, 200, 0)
            )

        if annotated is not None:
            self._cached_overlay_frame = annotated.copy()
            self.main_view.update_bgr(annotated)
            self.display_source.set("Right View: Live Camera + ROIs (no session)")
        else:
            self._cached_overlay_frame = frame.copy()
            self.main_view.update_bgr(frame)
            self.display_source.set("Right View: Live Camera")

    def _preview_payload_for_frame(self, payload: dict, frame) -> dict:
        client_timings = dict(payload.get("client_timings") or {})
        if frame is not None:
            height, width = frame.shape[:2]
            client_timings.setdefault("frame_width", width)
            client_timings.setdefault("frame_height", height)
        return {
            **payload,
            "client_timings": client_timings,
            "part_ready_roi_meta": self._roi_meta_payload("part_ready"),
            "sticker_roi_meta": self._roi_meta_payload("sticker"),
        }

    def _build_preview_frame(self, frame):
        if frame is None:
            return None
        with self._lock:
            payload = dict(self._latest_payload) if isinstance(self._latest_payload, dict) else None
        if payload:
            try:
                overlay = self._build_local_detection_overlay(
                    frame,
                    self._preview_payload_for_frame(payload, frame),
                )
                if overlay is not None:
                    self.display_source.set("Right View: Live Camera + ROIs (overlay)")
                    return overlay
            except Exception:
                _logger.warning("Overlay rendering error in _build_preview_frame", exc_info=True)
        overlay = self._draw_roi_overlays(frame, self._build_preview_overlay_payload(frame))
        if overlay is not None:
            self.display_source.set("Right View: Live Camera + ROIs")
            return overlay
        self.display_source.set("Right View: Live Camera")
        return frame

    def _render_preview_frame(self) -> bool:
        frame = self.capture.get_latest_frame()
        if frame is None:
            return False
        display_frame = self._build_preview_frame(frame)
        if display_frame is None:
            return False
        self._cached_overlay_frame = display_frame
        self.main_view.update_bgr(display_frame)
        return True

    def _schedule_preview_tick(self, *, delay_ms: int | None = None) -> None:
        if self._closed:
            return
        if self._preview_after_id:
            try:
                self.after_cancel(self._preview_after_id)
            except tk.TclError:
                pass
            self._preview_after_id = None
        wait = self._preview_interval_ms if delay_ms is None else max(0, int(delay_ms))
        self._preview_after_id = self.after(wait, self._preview_tick)

    def _cancel_preview_tick(self) -> None:
        if self._preview_after_id:
            try:
                self.after_cancel(self._preview_after_id)
            except tk.TclError:
                pass
            self._preview_after_id = None

    def _preview_tick(self) -> None:
        self._preview_after_id = None
        if self._closed:
            return
        try:
            self._render_preview_frame()
        except tk.TclError as exc:
            self.state.latest_error = str(exc)
            self.info_var.set(f"UI render warning: {exc}")
        finally:
            # Keep preview alive as long as capture is active (even during reconnect)
            if not self._closed and self.capture.is_active:
                self._schedule_preview_tick()

    def _current_template_id(self) -> int | None:
        selected = self._template_lookup.get(self.template_choice.get().strip())
        if selected and selected.get("id"):
            return int(selected["id"])
        deployment = self.state.active_deployment or {}
        if deployment.get("template_id"):
            return int(deployment["template_id"])
        current_version = self.template_version_value.get().strip()
        for item in self._template_lookup.values():
            if str(item.get("version_id")) == current_version and item.get("id"):
                return int(item["id"])
        return None

    def _apply_template_detail(
        self,
        detail: dict | None,
        *,
        lock_version_id: int | None = None,
        keep_line_station: bool = False,
    ) -> None:
        if not detail:
            return
        version_id = lock_version_id or detail.get("version_id") or detail.get("current_version_id")
        if version_id:
            self.template_version_value.set(str(version_id))

        # Prefer explicit per-ROI fields.  Fall back to legacy shared "roi" only
        # when the explicit field is missing — never apply the same legacy "roi"
        # to both slots which causes sticker ROI to inherit part-ready geometry.
        part_ready_roi = detail.get("part_ready_roi") or {}
        sticker_roi = detail.get("sticker_roi") or {}
        legacy_roi = detail.get("roi") or {}
        if not part_ready_roi and legacy_roi:
            part_ready_roi = legacy_roi
        if not sticker_roi and legacy_roi:
            # Only use legacy roi for sticker when part_ready already has its
            # own explicit value — otherwise the two ROIs would be identical.
            if detail.get("part_ready_roi"):
                sticker_roi = legacy_roi

        self._set_roi_values("part_ready", part_ready_roi)
        self._set_roi_values("sticker", sticker_roi)
        camera_config = detail.get("camera") or {}
        if camera_config.get("camera_index") is not None:
            self.camera_value.set(str(camera_config["camera_index"]))
        sticker_config = detail.get("sticker") or {}
        self.state.cache["selected_template_detail"] = detail
        self._refresh_context_summary()

    def _sync_selected_template_detail(self) -> None:
        template_id = self._current_template_id()
        if not template_id:
            self.info_var.set("Belum ada template aktif untuk disinkronkan.")
            return
        try:
            detail = self._fetch_template_detail(template_id)
        except Exception as exc:  # noqa: BLE001
            self.info_var.set(OperatorScreen._friendly_error(exc, "Gagal sinkronisasi template"))
            return
        self._apply_template_detail(detail)
        self.info_var.set(f"Template ROI synced: {detail.get('name')} v{detail.get('version_number')}")

    def _sync_template_selector(self) -> None:
        current_version = str(self.template_version_value.get().strip() or "")
        if not current_version:
            return
        for label, item in self._template_lookup.items():
            if str(item.get("version_id")) == current_version:
                self.template_choice.set(label)
                return

    def _on_template_selected(self, _event=None) -> None:
        label = self.template_choice.get().strip()
        item = self._template_lookup.get(label)
        if not item:
            return
        session_was_running = self.state.active_session is not None
        previous_camera_index = self.camera_value.get().strip()
        self.state.active_deployment = None
        try:
            detail = self._fetch_template_detail(int(item["id"]))
        except Exception as exc:  # noqa: BLE001
            self.template_version_value.set(str(item.get("version_id") or ""))
            self.info_var.set(OperatorScreen._friendly_error(exc, "Gagal memuat detail template"))
        else:
            if session_was_running:
                self._stop_session()
            self._apply_template_detail(detail)
            camera_changed = self.camera_value.get().strip() != previous_camera_index
            if camera_changed:
                self._restart_camera_for_template_change()
            if session_was_running:
                self.info_var.set(f"Template changed: restarting session with {item.get('name')} v{item.get('version_number')}")
                self._restart_session_after_template_change()
            else:
                self.info_var.set(f"Template selected manually: {item.get('name')} v{item.get('version_number')}")
        self._refresh_context_summary()

    def _restart_camera_for_template_change(self) -> bool:
        self.capture.stop()
        self.main_view.reset()
        return self._start_camera(show_errors=False)

    def _restart_session_after_template_change(self) -> None:
        if self.capture.get_latest_frame() is not None:
            self._start_session()
            return
        if not self._start_camera(show_errors=False):
            self.info_var.set("Template changed, tapi kamera belum siap. Start Camera/Session bisa dilakukan manual.")
            return
        self._wait_for_auto_start_frame(attempt=0)

    def _selected_template_name(self) -> str | None:
        selected = self._template_lookup.get(self.template_choice.get().strip())
        if not selected:
            return None
        return str(selected.get("name") or "").strip() or None

    def _set_badge(self, key: str, value: str, tone: str = "neutral") -> None:
        if not hasattr(self, "badges") or key not in self.badges:
            return
        bg, fg = BADGE_COLORS.get(tone, BADGE_COLORS["neutral"])
        self.badges[key].configure(text=f"{key}: {value}", fg_color=bg, text_color=fg)

    def _on_camera_status(self, status: str) -> None:
        """Handle camera status updates: 'connected', 'reconnecting', 'error'."""
        if status == "connected":
            self.camera_status_label.configure(text="● Kamera Terhubung", text_color="green")
        elif status == "reconnecting":
            self.camera_status_label.configure(text="● Menghubungkan kembali...", text_color="orange")
        elif status == "error":
            self.camera_status_label.configure(text="● Kamera Terputus", text_color="red")

    @staticmethod
    def _friendly_error(exc: Exception, context: str = "") -> str:
        """Convert raw API/technical errors into user-friendly Indonesian messages."""
        raw = str(exc)
        # Strip HTTP status prefixes like "503: ", "401: ", etc.
        msg = raw
        for prefix in ("503:", "500:", "404:", "403:", "401:", "400:"):
            if msg.startswith(prefix):
                msg = msg[len(prefix):].strip()
                break
        # PLC disabled
        if "PLC worker is disabled" in msg or "QC_SUITE_PLC_ENABLED=0" in msg:
            return "PLC tidak aktif. Hubungi teknisi untuk mengaktifkan PLC."
        # PLC reconnect failed
        if "reconnect failed" in msg or "modbus" in msg.lower():
            return "Koneksi PLC terputus. Periksa kabel dan pastikan PLC menyala."
        # Session not found / stopped
        if "session not found" in msg.lower():
            return "Sesi inspeksi tidak ditemukan. Mulai sesi baru."
        if "idle timeout" in msg.lower():
            return "Sesi dihentikan otomatis karena tidak aktif. Mulai sesi baru."
        # Camera errors
        if "camera" in msg.lower() and ("failed" in msg.lower() or "error" in msg.lower()):
            return "Kamera gagal dibuka. Periksa koneksi kamera."
        # Auth errors
        if "unauthorized" in msg.lower() or "401" in raw:
            return "Sesi login berakhir. Silakan login ulang."
        if "forbidden" in msg.lower() or "403" in raw:
            return "Akses ditolak. Anda tidak memiliki izin untuk tindakan ini."
        # Connection errors
        if "connection" in msg.lower() or "refused" in msg.lower() or "timeout" in msg.lower():
            return "Tidak dapat terhubung ke server. Periksa jaringan atau restart aplikasi."
        # Fallback: truncate long messages
        if len(msg) > 80:
            msg = msg[:77] + "..."
        return f"{context}: {msg}" if context else msg

    def _refresh_context_summary(self) -> None:
        username = self.state.user.get("username") if self.state.user else "-"
        self.operator_context.set(f"Operator: {username}")
        selected_detail = self.state.cache.get("selected_template_detail") if isinstance(self.state.cache, dict) else None
        template_name = (
            (self.state.active_session or {}).get("template_name")
            or (self.state.active_deployment or {}).get("template_name")
            or (selected_detail or {}).get("name")
            or self._selected_template_name()
            or "-"
        )
        template_version = self.template_version_value.get().strip() or (self.state.active_session or {}).get("template_version_id") or "-"
        self.template_context.set(f"Template: {template_name} v{template_version}")
        self.active_preset_var.set(template_name)
        self._sync_template_selector()

    def _update_status_badges(self, payload: dict | None = None) -> None:
        token = getattr(self.state, "token", None)
        latest_error = getattr(self.state, "latest_error", None)

        event_state = str((payload or {}).get("event_state") or "idle").upper()
        event_tone = "success" if event_state == "DECISION_COMMITTED" else "info" if event_state not in {"IDLE", "COOLDOWN"} else "neutral"
        self._set_badge("EVENT", event_state, event_tone)

        clamp = (payload or {}).get("clamp") or {}
        clamp_status = str(clamp.get("status") or "").strip().upper()
        if clamp_status:
            plc_tone = (
                "success"
                if clamp_status in {"CLAMPED", "READY"}
                else "danger"
                if "TIMEOUT" in clamp_status or "ERROR" in clamp_status
                else "warning"
                if clamp_status in {"WAIT_FEEDBACK", "CLAMPING"}
                else "info"
            )
            self._set_badge("PLC", clamp_status[:12], plc_tone)

        phase = (payload or {}).get("phase") or {}
        phase_status = str(phase.get("status") or "").strip().upper()
        if phase_status and phase_status not in {"READY", "WAITING_PART_READY"}:
            phase_tone = "warning" if "DELAY" in phase_status or "WAIT" in phase_status else "info"
            self._set_badge("EVENT", phase_status[:12], phase_tone)

    def _sync_recent_events(self, payload: dict) -> None:
        items = payload.get("recent_events") or []
        self.recent_list.delete(0, "end")
        for item in items:
            timestamp = str(item.get("committed_at") or "-").replace("T", " ")[:19]
            decision = item.get("decision") or "-"
            reason = item.get("reject_reason_code") or "OK"
            part_name = item.get("part_name") or "-"
            self.recent_list.insert("end", f"{timestamp} | {decision} | {part_name} | {reason}")

    def _load_deployment(self) -> None:
        try:
            response = self.api.get_active_deployment()
        except Exception as exc:
            messagebox.showerror("Deployment", f"Failed to load deployment: {exc}")
            return
        deployment = response.get("deployment") if isinstance(response, dict) else None
        if not deployment:
            messagebox.showwarning("Deployment", "Tidak ada deployment aktif.")
            return
        self._apply_deployment_record(deployment, source="Deployment loaded")

    def _apply_deployment_record(self, deployment: dict, *, source: str) -> None:
        session_was_running = self.state.active_session is not None
        previous_camera_index = self.camera_value.get().strip()
        if session_was_running:
            self._stop_session()
        self.state.active_deployment = deployment

        deployment_version_id = int(deployment.get("template_version_id") or 0)
        self.template_version_value.set(str(deployment_version_id or ""))
        detail = None
        if deployment_version_id:
            try:
                detail = self._fetch_template_version_detail(deployment_version_id)
            except Exception as exc:  # noqa: BLE001
                self.info_var.set(f"Deployment loaded, tapi detail template version gagal dibaca: {exc}")

        if detail is None and deployment.get("template_id"):
            try:
                detail = self._fetch_template_detail(int(deployment["template_id"]))
            except Exception as exc:  # noqa: BLE001
                self.info_var.set(f"Deployment loaded, tapi detail template gagal dibaca: {exc}")

        if detail is not None:
            self._apply_template_detail(
                detail,
                lock_version_id=deployment_version_id or None,
                keep_line_station=True,
            )
        camera_changed = self.camera_value.get().strip() != previous_camera_index
        if camera_changed:
            self._restart_camera_for_template_change()
        if session_was_running:
            self.info_var.set(f"{source}: restarting session with {deployment.get('template_name')}")
            self._restart_session_after_template_change()
        else:
            self.info_var.set(f"{source}: {deployment.get('template_name')}")
        self._refresh_context_summary()
        self._update_status_badges()

    def _start_camera(self, *, show_errors: bool = True) -> bool:
        settings = self._resolve_camera_settings()
        try:
            self._cancel_preview_tick()
            self.capture.start(
                int(self.camera_value.get() or 0),
                width=settings["width"],
                height=settings["height"],
                fps=settings["fps"],
            )
        except Exception as exc:  # noqa: BLE001
            if show_errors:
                messagebox.showerror("Camera", str(exc))
            self.info_var.set(OperatorScreen._friendly_error(exc, "Kamera gagal"))
            self._update_status_badges()
            return False
        actual = self._camera_settings_label()
        suffix = f" ({actual})" if actual else ""
        self.info_var.set(f"Camera started{suffix}. Menunggu frame pertama.")
        self._schedule_preview_tick(delay_ms=0)
        self._update_status_badges()
        return True

    def _start_session(self) -> None:
        """Start a new inspection session (assumes camera is ready and template is selected)."""
        template_version_id = int(self.template_version_value.get() or 0)
        if not template_version_id:
            messagebox.showerror("Session", "Template version wajib diisi.")
            return
        try:
            part_ready_roi = self._validated_roi_payload("part_ready")
            sticker_roi = self._validated_roi_payload("sticker")
        except ValueError as exc:
            messagebox.showerror("Session", str(exc))
            return
        payload = self.api.create_session(
            {
                "client_id": str(self.state.user.get("id") if self.state.user else "client"),
                "camera_index": int(self.camera_value.get() or 0),
                "template_version_id": template_version_id,
            }
        )
        self.state.active_session = payload
        self.api.update_rois(
            payload["session_id"],
            part_ready_roi=part_ready_roi,
            sticker_roi=sticker_roi,
        )
        self.state.latest_result = None
        self.state.latest_error = None
        self.state.cache["part_ready"] = None
        self.state.cache["sticker_detection"] = None
        self.state.cache["last_committed_result"] = None
        self._auth_error_notified = False
        self.capture.set_status_callback(self._on_camera_status)
        with self._lock:
            self._latest_payload = None
            self._latest_error = None
        self.result_panel.reset()
        self.counter_panel.reset()
        self.decision_banner.configure(fg_color="#334155", text_color="#f8fafc", text="WAITING")
        self.decision_subtitle.configure(text="Menunggu event inspeksi pertama.")
        self.plc_status_label.configure(text="")
        for _lbl in self.breakdown_labels.values():
            _lbl.configure(text="0")
        self.recent_list.delete(0, "end")
        _current_frame = self.capture.get_latest_frame()
        if _current_frame is not None:
            self._show_cached_overlay_or_frame(_current_frame)
        else:
            self.main_view.reset()
        self._inference_running = True
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            args=(payload["session_id"],),
            name="qc-inference",
            daemon=True,
        )
        self._inference_thread.start()
        upload_interval_ms = self._resolve_upload_interval_ms()
        infer_fps_actual = round(1000.0 / upload_interval_ms, 1)
        preview_fps_actual = round(1000.0 / self._preview_interval_ms, 1)
        self.info_var.set(
            f"Session running: {payload['session_id']} "
            f"(inference @ {upload_interval_ms}ms / {infer_fps_actual} fps | preview @ {preview_fps_actual} fps)"
        )
        self._refresh_context_summary()
        self._update_status_badges()

    def _start_production(self) -> None:
        if not self.template_version_value.get().strip():
            self._auto_start_first_template()
        if self.capture.get_latest_frame() is None and not self._start_camera():
            return
        if self.state.active_session:
            self.info_var.set("Production inspection is already running.")
            return
        self._start_session()

    def _stop_production(self) -> None:
        self._stop_session()
        self._refresh_context_summary()
        self._update_status_badges()

    def _inference_loop(self, session_id: str) -> None:
        """Background thread: capture frame, upload, infer, then update UI."""
        import time as _time
        import cv2 as _cv2
        import base64 as _base64

        target_interval_s = max(0.05, self._resolve_upload_interval_ms() / 1000.0)
        while (
            self._inference_running
            and (self.state.active_session or {}).get("session_id") == session_id
        ):
            try:
                loop_start = _time.perf_counter()

                # Ambil frame dari camera
                t0 = _time.perf_counter()
                frame = self.capture.get_latest_frame()
                capture_ms = (_time.perf_counter() - t0) * 1000.0
                if frame is None:
                    _time.sleep(0.05)
                    continue

                # Skip inference saat camera reconnecting
                if self.capture.is_reconnecting:
                    _time.sleep(0.05)
                    continue

                # Upload ke backend
                active_session = self.state.active_session
                if active_session and active_session.get("session_id") == session_id:
                    _username = self.state.user.get("username") if self.state.user else None
                    _user_id = self.state.user.get("id") if self.state.user else None
                    t0 = _time.perf_counter()
                    if self.api._local_mode:
                        # Fast path: skip encode/base64/HTTP entirely for in-process backend
                        result = self.api.push_frame_local(
                            session_id,
                            frame,
                            username=_username,
                            user_id=_user_id,
                            response_mode="stream",
                        )
                        encode_ms = 0.0
                        b64_ms = 0.0
                        buffer = None
                    else:
                        # Remote path: encode → base64 → HTTP POST
                        _jpeg_quality = int(_os.getenv("QC_SUITE_JPEG_QUALITY", "85"))
                        _use_jpeg = int(_os.getenv("QC_SUITE_USE_JPEG", "1"))
                        t_enc = _time.perf_counter()
                        if _use_jpeg:
                            ok, buffer = _cv2.imencode(".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, _jpeg_quality])
                        else:
                            ok, buffer = _cv2.imencode(".png", frame)
                        encode_ms = (_time.perf_counter() - t_enc) * 1000.0
                        if not ok:
                            raise RuntimeError("Failed to encode frame.")
                        t_b64 = _time.perf_counter()
                        image_b64 = _base64.b64encode(buffer).decode("ascii")
                        b64_ms = (_time.perf_counter() - t_b64) * 1000.0
                        result = self.api.push_frame(
                            session_id,
                            image_b64,
                            response_mode="stream",
                        )
                    request_ms = (_time.perf_counter() - t0) * 1000.0
                    if isinstance(result, dict):
                        result.setdefault(
                            "client_timings",
                            {
                                "capture_ms": round(capture_ms, 2),
                                "encode_ms": round(encode_ms, 2),
                                "b64_ms": round(b64_ms, 2),
                                "request_ms": round(request_ms, 2),
                                "frame_width": frame.shape[1],
                                "frame_height": frame.shape[0],
                                "jpeg_quality": 0 if self.api._local_mode else int(_os.getenv("QC_SUITE_JPEG_QUALITY", "85")),
                                "payload_bytes": int(len(buffer)) if buffer is not None else 0,
                            },
                        )
                    if result and self._inference_running and (self.state.active_session or {}).get("session_id") == session_id:
                        # Update UI dari main thread
                        try:
                            self.after(0, lambda r=result: self._set_result(r))
                        except tk.TclError:
                            return

                # Sleep untuk maintain target inference fps
                elapsed = _time.perf_counter() - loop_start
                sleep_time = max(0.01, target_interval_s - elapsed)
                _time.sleep(sleep_time)

            except Exception as exc:
                if self._inference_running and (self.state.active_session or {}).get("session_id") == session_id:
                    try:
                        self.after(0, lambda e=str(exc): self._set_error(e))
                    except tk.TclError:
                        return
                else:
                    return
                _time.sleep(0.1)

    def _stop_session(self) -> None:
        stop_message = "Session stopped."
        if self.state.active_session:
            try:
                self.api.stop_session(self.state.active_session["session_id"])
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                if self._is_auth_error(message):
                    stop_message = "Session lokal dihentikan, tapi stop di backend gagal (401). Silakan login ulang."
                    if not self._auth_error_notified:
                        self._auth_error_notified = True
                        messagebox.showwarning("Session", "Sesi backend sudah tidak terotorisasi (401). Silakan login ulang.")
                else:
                    stop_message = f"Session lokal dihentikan. Warning backend: {message}"
        # Stop inference thread
        self._inference_running = False
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=2.0)
            self._inference_thread = None
        self.uploader.stop()
        self.state.active_session = None
        self.state.cache["part_ready"] = None
        self.state.cache["sticker_detection"] = None
        with self._lock:
            self._latest_payload = None
            self._latest_error = None
        self.info_var.set(stop_message)
        self._refresh_context_summary()
        self._update_status_badges()

    def _apply_roi(self) -> None:
        try:
            part_ready_roi = self._validated_roi_payload("part_ready")
            sticker_roi = self._validated_roi_payload("sticker")
        except ValueError as exc:
            messagebox.showerror("ROI", str(exc))
            return
        if not self.state.active_session:
            self.info_var.set("Dua ROI disimpan lokal. Akan diterapkan saat session aktif.")
            self._refresh_context_summary()
            return
        self.api.update_rois(
            self.state.active_session["session_id"],
            part_ready_roi=part_ready_roi,
            sticker_roi=sticker_roi,
        )
        self.info_var.set("Part-ready ROI dan sticker ROI updated.")

    @staticmethod
    def _human_readable_reason(code: str) -> str:
        """Map RejectReasonCode to human-readable Indonesian text."""
        _map = {
            "NOT_FOUND": "Sticker tidak ditemukan",
            "WRONG_TYPE": "Tipe sticker salah",
            "WRONG_TEXT": "Teks sticker salah",
            "LOW_ROI_CONF": "Confidence ROI rendah",
            "LOW_CLASS_CONF": "Confidence kelas rendah",
            "LOW_OCR_CONF": "Confidence OCR rendah",
            "OUT_OF_POSITION": "Posisi sticker tidak tepat",
            "OUT_OF_ANGLE": "Sticker miring",
            "ANCHOR_NOT_FOUND": "Anchor teks tidak ditemukan",
            "ANCHOR_MISMATCH": "Anchor teks tidak cocok",
            "PART_NOT_READY": "Part belum siap",
            "COMMIT_TIMEOUT": "Timeout — tidak ada ACCEPT",
            "ERROR": "Error sistem",
            "PLC_FAULT": "Koneksi PLC terputus",
        }
        return _map.get(code, code)

    def _update_result_info(self, payload: dict) -> None:
        timings = payload.get("timings") or {}
        total_ms = timings.get("total_ms")
        inference_ms = timings.get("inference_ms")
        client_timings = payload.get("client_timings") or {}
        timing_suffix = ""
        if isinstance(total_ms, (int, float)):
            timing_suffix = f" | backend={float(total_ms):.0f}ms"
        if isinstance(inference_ms, (int, float)):
            timing_suffix += f" infer={float(inference_ms):.0f}ms"
        req_ms = client_timings.get("request_ms")
        if isinstance(req_ms, (int, float)):
            timing_suffix += f" req={float(req_ms):.0f}ms"

        clamp = payload.get("clamp") or {}
        clamp_status = str(clamp.get("status") or "").strip()
        clamp_suffix = f" | clamp={clamp_status}" if clamp_status else ""
        phase = payload.get("phase") or {}
        phase_status = str(phase.get("status") or "").strip()
        phase_remaining = phase.get("remaining_ms")
        if isinstance(phase_remaining, (int, float)):
            phase_suffix = f" | phase={phase_status} {float(phase_remaining):.0f}ms"
        else:
            phase_suffix = f" | phase={phase_status}" if phase_status else ""

        if payload.get("count_committed"):
            committed = payload.get("last_committed_result") or {}
            validation = committed.get("validation") or {}
            part_ready = committed.get("part_ready") or {}
            detection = committed.get("sticker_detection") or {}
            self.info_var.set(
                f"Committed {validation.get('decision')} | "
                f"Part gate={'READY' if part_ready.get('part_ready') else 'BLOCK'} | "
                f"{validation.get('part_name') or '-'} | "
                f"{validation.get('reject_reason_code') or 'OK'} | "
                f"backend={detection.get('backend') or '-'} raw={detection.get('raw_detection_count') if detection.get('raw_detection_count') is not None else '-'}"
                f"{clamp_suffix}{phase_suffix}{timing_suffix}."
            )
            return

        live_validation = payload.get("validation") or {}
        live_part_ready = payload.get("part_ready") or {}
        live_detection = payload.get("sticker_detection") or {}
        self.info_var.set(
            f"Gate={'READY' if live_part_ready.get('part_ready') else 'BLOCK'} "
            f"(ratio {live_part_ready.get('match_ratio') if live_part_ready.get('match_ratio') is not None else '-'} | "
            f"pr_status={live_part_ready.get('status') or '-'}) | "
            f"Live decision: {live_validation.get('decision') or '-'} | "
            f"Detected: {live_validation.get('detected_class') or '-'} | "
            f"Reject: {live_validation.get('reject_reason_code') or 'OK'} | "
            f"backend={live_detection.get('backend') or '-'} reason={live_detection.get('reason') or '-'} raw={live_detection.get('raw_detection_count') if live_detection.get('raw_detection_count') is not None else '-'}"
            f"{clamp_suffix}{phase_suffix}{timing_suffix}"
        )

    def _set_result(self, payload: dict) -> None:
        """Handle inference result from backend. Update UI directly."""
        with self._lock:
            self._latest_payload = payload
        new_dets = payload.get("detections") or []
        if new_dets:
            self._last_detections = new_dets
            self._last_detections_ts = monotonic()

        # Update all UI elements directly (no separate poll needed)
        try:
            self.state.latest_result = payload
            self.state.latest_error = None
            self.state.cache["part_ready"] = payload.get("part_ready")
            self.state.cache["sticker_detection"] = payload.get("sticker_detection")
            self.state.cache["last_committed_result"] = payload.get("last_committed_result")
            self.result_panel.update_payload(payload)
            self.counter_panel.update_payload(payload)

            # Update decision banner
            _live_val = payload.get("validation") or {}
            _committed = payload.get("last_committed_result") or {}
            _committed_val = _committed.get("validation") or {}
            _disp_val = _committed_val or _live_val
            _decision = _disp_val.get("decision") or "WAITING"
            _d_palette = {"ACCEPT": ("#166534", "#f0fdf4"), "REJECT": ("#991b1b", "#fef2f2"), "WAITING": ("#334155", "#f8fafc")}
            _d_bg, _d_fg = _d_palette.get(_decision, ("#334155", "#f8fafc"))

            # Human-readable reason code
            _reason = _disp_val.get("reject_reason_code")
            _reason_hr = self._human_readable_reason(_reason) if _reason else ""

            _banner_text = _decision
            if _reason_hr:
                _banner_text += f" — {_reason_hr}"
            if _committed_val:
                _subtitle = "Menampilkan hasil committed terakhir"
            else:
                _subtitle = "Belum ada event committed. Banner mengikuti hasil live terbaru."

            self.decision_banner.configure(fg_color=_d_bg, text_color=_d_fg, text=_banner_text)
            self.decision_subtitle.configure(text=_subtitle)

            # Update reject breakdown
            _breakdown = (payload.get("counters") or {}).get("session_reject_breakdown") or {}
            for _key, _lbl in self.breakdown_labels.items():
                _lbl.configure(text=str(_breakdown.get(_key, 0)))
            self._sync_recent_events(payload)
            self._refresh_context_summary()
            self._update_status_badges(payload)
            self._update_result_info(payload)
            self._render_overlay_direct(payload)

        except tk.TclError:
            pass

    def _render_overlay_direct(self, payload: dict) -> None:
        """Render full overlay (ROI boxes + detections + decision) on live frame."""
        try:
            frame = self.capture.get_latest_frame()
            if frame is None:
                return
            # Enrich payload with client-side frame dimensions for scaling.
            # Use a COPY of client_timings to avoid mutating the original payload.
            client_timings = dict(payload.get("client_timings") or {})
            client_timings["frame_width"] = frame.shape[1]
            client_timings["frame_height"] = frame.shape[0]
            enriched = {**payload, "client_timings": client_timings}

            overlay = self._build_local_detection_overlay(frame, enriched)
            if overlay is not None:
                self._cached_overlay_frame = overlay
                self.main_view.update_bgr(overlay)
                self.display_source.set("Right View: Live Camera + ROIs (local)")

        except Exception:
            _logger.warning("Overlay rendering error in _render_overlay_direct", exc_info=True)

    def _show_cached_overlay_or_frame(self, frame) -> None:
        """Show cached overlay if available, otherwise show raw frame."""
        if self._cached_overlay_frame is not None:
            self.display_source.set("Right View: Live Camera + ROIs (cached)")
        elif frame is not None:
            self.display_source.set("Right View: Live Camera")

    @staticmethod
    def _is_auth_error(message: str | None) -> bool:
        text = str(message or "").strip().lower()
        return "401" in text or "unauthorized" in text

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._latest_error = message
        self.state.latest_error = message
        self.info_var.set(f"Error inspeksi: {message}")

    def _schedule_heartbeat(self, *, delay_ms: int | None = None) -> None:
        if self._closed:
            return
        if self._heartbeat_after_id:
            try:
                self.after_cancel(self._heartbeat_after_id)
            except tk.TclError:
                pass
        wait = HEARTBEAT_INTERVAL_MS if delay_ms is None else max(500, int(delay_ms))
        self._heartbeat_after_id = self.after(wait, self._send_heartbeat)

    def _send_heartbeat(self) -> None:
        self._heartbeat_after_id = None
        if self._closed:
            return
        if not getattr(self.state, "token", None):
            self._schedule_heartbeat()
            return

        def _load():
            return self.api.heartbeat(
                self._machine_id,
                client_version="client_tk",
            )

        run_async(self, _load, callback=lambda _result, _error: None)
        self._schedule_heartbeat()

    def _schedule_plc_poll(self, *, delay_ms: int | None = None) -> None:
        if self._closed:
            return
        if self._plc_poll_after_id:
            try:
                self.after_cancel(self._plc_poll_after_id)
            except tk.TclError:
                pass
        wait = PLC_POLL_INTERVAL_MS if delay_ms is None else max(500, int(delay_ms))
        self._plc_poll_after_id = self.after(wait, self._poll_plc_status)

    def _poll_plc_status(self) -> None:
        self._plc_poll_after_id = None
        if self._closed:
            return
        if not getattr(self.state, "token", None):
            self._schedule_plc_poll()
            return

        def _load():
            return self.api.plc_status()

        def _on_done(result, error):
            if self._closed:
                return
            if result and not error:
                self._handle_plc_template_cycle_event(result)
                self._update_plc_badge_from_status(result)

        run_async(self, _load, callback=_on_done)
        self._schedule_plc_poll()

    # ------------------------------------------------------------------
    # Datapart guard: every 5 accepted judgements pushed to SQL, further
    # judgements lock until a downstream MES process fills DatapartID for
    # all 5 (or a LEADERPI bypasses it). Polled independently of the frame
    # loop so it's still detected/cleared while the camera is paused.
    # ------------------------------------------------------------------

    def _schedule_datapart_guard_poll(self, *, delay_ms: int | None = None) -> None:
        if self._closed:
            return
        if self._datapart_guard_poll_after_id:
            try:
                self.after_cancel(self._datapart_guard_poll_after_id)
            except tk.TclError:
                pass
        wait = DATAPART_GUARD_POLL_INTERVAL_MS if delay_ms is None else max(500, int(delay_ms))
        self._datapart_guard_poll_after_id = self.after(wait, self._poll_datapart_guard_status)

    def _poll_datapart_guard_status(self) -> None:
        self._datapart_guard_poll_after_id = None
        if self._closed:
            return
        if not getattr(self.state, "token", None):
            self._schedule_datapart_guard_poll()
            return

        def _load():
            return self.api.get_datapart_guard_status()

        def _on_done(result, error):
            if self._closed:
                return
            if result and not error:
                if result.get("locked"):
                    self._show_datapart_guard_lock()
                else:
                    self._dismiss_datapart_guard_lock()
            self._schedule_datapart_guard_poll()

        run_async(self, _load, callback=_on_done)

    def _show_datapart_guard_lock(self) -> None:
        if self._datapart_guard_popup is not None:
            return  # already showing
        try:
            if self.capture.is_active:
                self.capture.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._datapart_guard_popup = DatapartGuardPopup(
                self, on_bypass_rfid=self._handle_datapart_guard_bypass
            )
        except Exception:  # noqa: BLE001
            self._datapart_guard_popup = None

    def _dismiss_datapart_guard_lock(self) -> None:
        if self._datapart_guard_popup is None:
            return
        try:
            self._datapart_guard_popup.grab_release()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._datapart_guard_popup.destroy()
        except Exception:  # noqa: BLE001
            pass
        self._datapart_guard_popup = None
        if self.state.active_session and not self._closed:
            self._start_camera(show_errors=False)

    def _handle_datapart_guard_bypass(self, rfid_uid: str) -> None:
        def _load():
            return self.api.override_datapart_guard(rfid_uid)

        def _on_done(result, error):
            if self._closed:
                return
            if error or not result:
                # Don't reuse _friendly_error here — its generic 401/403
                # handling ("sesi login berakhir" / "akses ditolak") would be
                # misleading for a card-scan flow. The backend's own message
                # ("Kartu RFID tidak dikenali.", "Kartu ini bukan LEADERPI.")
                # is already specific and in Indonesian — just strip the
                # "NNN: " status-code prefix that ApiClient adds.
                message = "Kartu RFID ditolak."
                if error is not None:
                    raw = str(error)
                    for prefix in ("400:", "401:", "403:"):
                        if raw.startswith(prefix):
                            raw = raw[len(prefix):].strip()
                            break
                    message = raw or message
                if self._datapart_guard_popup is not None:
                    self._datapart_guard_popup.set_bypass_error(message)
                return
            self._dismiss_datapart_guard_lock()

        run_async(self, _load, callback=_on_done)

    def _handle_plc_template_cycle_event(self, status: dict) -> None:
        raw_event_id = status.get("template_cycle_event_id")
        try:
            event_id = int(raw_event_id)
        except (TypeError, ValueError):
            return
        if self._last_plc_template_cycle_event_id is None:
            self._last_plc_template_cycle_event_id = event_id
            return
        if event_id <= self._last_plc_template_cycle_event_id:
            return
        self._last_plc_template_cycle_event_id = event_id
        # Cycle template dari dropdown (bukan deployment list)
        self._cycle_template_dropdown()

    def _cycle_template_dropdown(self) -> None:
        """Ganti template aktif ke template berikutnya di dropdown (wrap-around)."""
        # Ambil daftar template dari dropdown
        values = list(self.template_selector.cget("values") or [])
        if not values:
            # Dropdown kosong — coba reload
            self.info_var.set("IN2: memuat template...")
            self._load_template_choices()
            values = list(self.template_selector.cget("values") or [])
            if not values:
                self.info_var.set("IN2: tidak ada template tersedia.")
                return

        if len(values) <= 1:
            self.info_var.set("IN2: hanya satu template tersedia.")
            return

        # Cari index template saat ini
        current = self.template_choice.get().strip()
        current_index = 0
        for i, v in enumerate(values):
            if v == current:
                current_index = i
                break

        # Pilih template berikutnya (wrap-around)
        next_index = (current_index + 1) % len(values)
        next_label = values[next_index]

        # Set template choice dan trigger selected handler
        self.template_choice.set(next_label)
        self._on_template_selected()
        self.info_var.set(f"IN2 template switch: {next_label.split(' | ')[0]}")

    # Legacy methods — tidak dipakai lagi untuk IN2
    def _update_plc_badge_from_status(self, status: dict) -> None:
        if not status.get("enabled", True):
            self._set_badge("PLC", "DISABLED", "neutral")
            return
        if not status.get("running"):
            self._set_badge("PLC", "STOPPED", "danger")
            return
        connected = status.get("connected", False)
        clamp_engaged = status.get("clamp_engaged", False)
        plc_state = str(status.get("state") or "").strip().upper()
        if not connected:
            self._set_badge("PLC", "DISCONN", "danger")
        elif clamp_engaged or plc_state == "CLAMPED":
            self._set_badge("PLC", "ENGAGED", "warning")
        elif plc_state == "CLAMPING":
            self._set_badge("PLC", "CLAMPING", "warning")
        elif plc_state in {"REJECT_BUZZER", "ACCEPT_PULSE"}:
            self._set_badge("PLC", plc_state[:12], "warning")
        else:
            self._set_badge("PLC", "READY", "success")

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        self._cancel_preview_tick()
        if self._heartbeat_after_id:
            try:
                self.after_cancel(self._heartbeat_after_id)
            except tk.TclError:
                pass
            self._heartbeat_after_id = None
        if self._plc_poll_after_id:
            try:
                self.after_cancel(self._plc_poll_after_id)
            except tk.TclError:
                pass
            self._plc_poll_after_id = None
        if self._datapart_guard_poll_after_id:
            try:
                self.after_cancel(self._datapart_guard_poll_after_id)
            except tk.TclError:
                pass
            self._datapart_guard_poll_after_id = None
        if self._datapart_guard_popup is not None:
            try:
                self._datapart_guard_popup.destroy()
            except Exception:  # noqa: BLE001
                pass
            self._datapart_guard_popup = None
        if self._auto_start_after_id:
            try:
                self.after_cancel(self._auto_start_after_id)
            except tk.TclError:
                pass
            self._auto_start_after_id = None
        self._clear_resize_debounce()
        if self._overlay_render_after_id:
            try:
                self.after_cancel(self._overlay_render_after_id)
            except tk.TclError:
                pass
            self._overlay_render_after_id = None
        # Wait for background render thread to finish (max 1s)
        if self._overlay_thread is not None and self._overlay_thread.is_alive():
            self._overlay_thread.join(timeout=1.0)
        self._overlay_thread = None
        self._stop_session()
        self.capture.stop()

    def destroy(self) -> None:
        self.shutdown()
        super().destroy()
