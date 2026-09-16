from __future__ import annotations

import base64
from collections import OrderedDict
import copy
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import customtkinter as ctk
import cv2
import numpy as np

from backend.app.services.calibration import CalibrationService, MIN_CALIBRATION_PROFILE_PIXELS
from backend.app.core.model_catalog import list_base_models as catalog_list_base_models
from client_tk.app.api_client import ApiClient
from client_tk.app.components.async_bridge import run_async
from client_tk.app.components.annotation_canvas import AnnotationCanvas
from client_tk.app.components.roi_picker_canvas import RoiPickerCanvas
from client_tk.app.components.scrollable_frame import AutoHideScrollbar, ScrollableFrame
from client_tk.app.components.template_forms import JsonEditor, LabeledValuePanel
from client_tk.app.screens.admin.tabs.templates_tab import TemplatesTab
from client_tk.app.screens.admin.tabs.operators_tab import OperatorsTab
from client_tk.app.screens.admin.tabs.models_tab import ModelsTab
from client_tk.app.screens.admin.tabs.calibration_tab import CalibrationTab
from client_tk.app.screens.admin.tabs.results_tab import ResultsTab
from client_tk.app.screens.admin.tabs.machine_settings_tab import MachineSettingsTab
from client_tk.app.mode_utils import mode_label, mode_to_radio, normalize_mode, mode_from_template, validator_mode_for_payload
from client_tk.app.theme import (
    ACCENT,
    ACCENT_HOVER,
    APP_BG,
    BORDER,
    PANEL_ALT_BG,
    PANEL_BG,
    SHELL_BG,
    SUCCESS,
    SUCCESS_HOVER,
    TEXT_ON_ACCENT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from shared.contracts.augment import TRANSFORM_CATALOG as _TRANSFORM_CATALOG




RESPONSIVE_BREAKPOINT = 1180


def _safe_text(value: object, fallback: str = "-") -> str:
    text = str(value or "").strip()
    return text or fallback


def _format_timestamp(value: object) -> str:
    text = _safe_text(value)
    if text == "-":
        return text
    return text.replace("T", " ")[:19]


def _format_status(value: object) -> str:
    return "Active" if bool(value) else "Inactive"


def _float_or_default(value: object, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


class CompactStatCard(ctk.CTkFrame):
    def __init__(self, master, title: str, *, background: str, foreground: str):
        super().__init__(master, fg_color=background, corner_radius=8, border_width=1, border_color=BORDER)
        self.columnconfigure(0, weight=1)
        ctk.CTkLabel(self, text=title, text_color=foreground, font=("Segoe UI", 9, "bold")).grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=(8, 0),
        )
        self.value_label = ctk.CTkLabel(self, text="0", text_color=foreground, font=("Segoe UI", 16, "bold"))
        self.value_label.grid(row=1, column=0, sticky="w", padx=10)
        self.note_label = ctk.CTkLabel(self, text="", text_color=foreground, font=("Segoe UI", 8))
        self.note_label.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 8))

    def set_value(self, value: object, note: str = "") -> None:
        self.value_label.configure(text=str(value))
        self.note_label.configure(text=str(note or ""))


class AdminScreen(ctk.CTkFrame):
    """Production-facing Admin workspace.

    Admin only handles preset deployment, RFID operator onboarding, and monitoring.
    Full template/model/debug controls stay in the Engineer screen.
    """

    def __init__(self, master, api_client, session_state):
        super().__init__(master, fg_color=APP_BG, corner_radius=0)
        self.api = api_client
        self.state = session_state

        self._layout_compact: bool | None = None
        self._overview_cards_visible = True
        self._last_refresh: dict[str, str] = {}
        self._tab_scrollers: dict[str, ScrollableFrame] = {}

        self._deployments_cache: list[dict] = []
        self._templates_cache: list[dict] = []
        self._models_cache: list[dict] = []
        self._model_lookup: dict[str, dict] = {}
        self._template_model_lookup: dict[str, dict] = {}
        self._datasets_cache: list[dict] = []
        self._base_models_cache: list[dict] = []
        self._base_model_lookup: dict[str, dict] = {}
        self._training_jobs_cache: list[dict] = []
        self._augment_jobs_cache: list[dict] = []
        self._users_cache: list[dict] = []
        self._results_cache: list[dict] = []

        # Data tab state
        self._dataset_display_to_id: dict[str, str] = {}
        self._dataset_id_to_display: dict[str, str] = {}
        self._dataset_version_cache: list[dict] = []
        self._dataset_version_lookup: dict[str, dict] = {}
        self._annotation_files: list[dict] = []
        self._annotation_index: int | None = None
        self._annotation_dataset_id: str | None = None
        self._annotation_class_name: str = "object"
        self._annotation_manual_classes: list[str] = []
        self._annotation_selected_label_index: int | None = None
        self._annotation_cache: OrderedDict[tuple[str, str], dict] = OrderedDict()
        self._annotation_cache_bytes = 0
        self._annotation_cache_max_items = 8
        self._annotation_cache_max_bytes = 48 * 1024 * 1024
        self._annotation_load_sequence = 0
        self._annotation_files_refresh_sequence = 0
        self._train_dataset_display_to_id: dict[str, str] = {}
        self._train_dataset_id_to_display: dict[str, str] = {}

        # Training tab state
        self._training_jobs_all: list[dict] = []
        self._training_status_filter = None
        self._active_training_job_id: str | None = None
        self._augment_selected_transforms: list[str] = ["brightness", "blur"]
        self._augment_capabilities: dict | None = None
        self._flow_step_vars: list[tk.StringVar] = []

        # Calibration tab state
        self._calibration_profiles_cache: list[dict] = []
        self._calibration_selected_profile_id: str | None = None

        self.current_template_id: int | None = None
        self.current_template_version_id: int | None = None
        self._editing_deployment_id: int | None = None

        self.status_var = tk.StringVar(value="Admin ready.")
        self.refresh_time_var = tk.StringVar(value="")
        self.validator_mode_var = tk.StringVar(value="sticker")  # "sticker" or "component_count"
        self.preset_validator_mode_var = tk.StringVar(value="sticker")
        self.preset_component_rois: list = []
        self.preset_defect_rois: list = []
        self._calib_empty_mean: float = 0.0
        self._calib_part_mean: float = 0.0
        self._calib_part_std: float = 0.0
        self._calib_sticker_std: float = 0.0
        self._template_detail_cache: dict | None = None
        self.preset_model_classes_var = tk.StringVar()
        self.preset_name_var = tk.StringVar()
        self.preset_description_var = tk.StringVar()
        self.preset_model_choice_var = tk.StringVar()
        self.preset_model_path_var = tk.StringVar()
        self.preset_model_meta_path_var = tk.StringVar()
        self.preset_runtime_var = tk.StringVar(value="auto")
        self.preset_conf_threshold_var = tk.StringVar(value="0.25")
        self.preset_expected_class_var = tk.StringVar()
        self.preset_max_tilt_var = tk.StringVar(value="")
        self.preset_tilt_gate_var = tk.BooleanVar(value=False)
        self.preset_gap_threshold_var = tk.StringVar(value="0.85")
        self.preset_camera_index_var = tk.StringVar(value="0")
        self.preset_camera_rotation_var = tk.StringVar(value="0")
        self.preset_roi_choice_var = tk.StringVar(value="Sticker ROI")
        self.preset_part_ready_source_var = tk.StringVar(value="sensor")
        self.part_ready_roi_x_var = tk.StringVar(value="0.2")
        self.part_ready_roi_y_var = tk.StringVar(value="0.2")
        self.part_ready_roi_w_var = tk.StringVar(value="0.25")
        self.part_ready_roi_h_var = tk.StringVar(value="0.25")
        self._preset_roi_image_path: str = ""
        self.sticker_roi_x_var = tk.StringVar(value="0.2")
        self.sticker_roi_y_var = tk.StringVar(value="0.2")
        self.sticker_roi_w_var = tk.StringVar(value="0.6")
        self.sticker_roi_h_var = tk.StringVar(value="0.6")

        self.operator_username_var = tk.StringVar()
        self.operator_password_var = tk.StringVar()
        self.operator_role_var = tk.StringVar(value="operator")
        self.operator_edit_id: int | None = None
        self.operator_edit_username_var = tk.StringVar()
        self.operator_edit_role_var = tk.StringVar(value="operator")

        # Unified RFID bind state
        self.bind_target_user_id: int | None = None
        self.unified_rfid_var = tk.StringVar()

        self.training_dataset_var = tk.StringVar()
        self.training_base_model_var = tk.StringVar()
        self.training_device_var = tk.StringVar(value="auto")
        self.training_epochs_var = tk.StringVar(value="200")
        self._admin_train_mode_var = tk.StringVar(value="real")
        self.augment_dataset_var = tk.StringVar()
        self.model_import_path_var = tk.StringVar()

        self.monitor_context_var = tk.StringVar(value="Recent production activity.")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_header()
        self._build_tabs()
        self._build_status_bar()
        self.bind("<Configure>", self._on_resize)
        self.after_idle(self._apply_responsive_layout)

        self.refresh_all()
        # Guard: verify all preset_* vars are registered in FORM_DEFAULTS
        # Runs after tabs are built so all vars (including from _build_wizard) exist
        self._assert_form_defaults()

    # ------------------------------------------------------------------
    # Layout
    def _build_header(self) -> None:
        self.header = ctk.CTkFrame(self, fg_color=SHELL_BG, corner_radius=0)
        self.header.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 6))
        self.header.columnconfigure(0, weight=1)

        user = self.state.user or {}
        identity = f"{_safe_text(user.get('username'))} ({_safe_text(user.get('role'), 'admin')})"
        ctk.CTkLabel(
            self.header,
            text="Admin Production Setup",
            font=("Segoe UI", 14, "bold"),
            text_color=TEXT_PRIMARY,
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 0))
        ctk.CTkLabel(
            self.header,
            text=f"{identity} | Templates, models, operators, and production monitor.",
            text_color=TEXT_SECONDARY,
            font=("Segoe UI", 10),
        ).grid(row=1, column=0, sticky="w", padx=12, pady=(2, 8))
        ctk.CTkButton(
            self.header,
            text="Refresh",
            command=self.refresh_all,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=TEXT_ON_ACCENT,
            height=30,
            corner_radius=6,
        ).grid(row=0, column=1, rowspan=2, sticky="e", padx=12, pady=10)

        self.overview_cards_frame = ctk.CTkFrame(self.header, fg_color="transparent", corner_radius=0)
        self.overview_cards_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 10))
        for index in range(4):
            self.overview_cards_frame.columnconfigure(index, weight=1)
        self.admin_cards = {
            "presets": CompactStatCard(self.overview_cards_frame, "Active Templates", background="#0f172a", foreground="#f8fafc"),
            "operators": CompactStatCard(self.overview_cards_frame, "Operators", background="#134e4a", foreground="#ecfdf5"),
            "accept": CompactStatCard(self.overview_cards_frame, "Accept", background="#166534", foreground="#f0fdf4"),
            "reject": CompactStatCard(self.overview_cards_frame, "Reject", background="#991b1b", foreground="#fef2f2"),
        }
        self._layout_overview_cards(compact=False)

    def _build_tabs(self) -> None:
        notebook = ctk.CTkTabview(self, fg_color=APP_BG, corner_radius=0)
        notebook.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
        for tab_name in ("Templates", "Data", "Training", "Models", "Calibration", "Operators", "Monitor", "Machine Settings"):
            notebook.add(tab_name)

        original_tab = notebook.tab

        def _tabs() -> list[str]:
            return ["Templates", "Data", "Training", "Models", "Calibration", "Operators", "Monitor"]

        def _select(tab_id: str | None = None):
            if tab_id is None:
                return notebook.get()
            was_templates = (notebook.get() == "Templates")
            going_to_templates = (tab_id == "Templates")
            # Stop live camera when leaving Templates tab to prevent MSMF camera conflict
            if was_templates and not going_to_templates:
                picker = getattr(self, "preset_roi_picker", None)
                if picker is not None and getattr(picker, "_cam_running", False):
                    try:
                        picker.stop_live_camera()
                    except Exception:
                        pass
                    if hasattr(self, "_live_cam_btn"):
                        self._live_cam_btn.configure(text="Start Live Camera")
            notebook.set(tab_id)
            return tab_id

        def _tab(tab_id: str, option: str | None = None):
            if option == "text":
                return tab_id
            return original_tab(tab_id)

        notebook.tabs = _tabs  # type: ignore[attr-defined]
        notebook.select = _select  # type: ignore[attr-defined]
        notebook.tab = _tab  # type: ignore[attr-defined]

        self._notebook = notebook
        # Stop live camera when admin window loses focus (operator may start a session)
        self.bind("<FocusOut>", lambda _: self._stop_live_camera_if_running())
        self.presets_tab = notebook.tab("Templates")
        self.data_tab = notebook.tab("Data")
        self.training_tab = notebook.tab("Training")
        self.models_tab = notebook.tab("Models")
        self.calibration_tab = notebook.tab("Calibration")
        self.operators_tab = notebook.tab("Operators")
        self.monitor_tab = notebook.tab("Monitor")
        self.plc_settings_tab = notebook.tab("Machine Settings")

        self._build_presets_tab()
        self._build_data_tab()
        self._build_training_tab()
        self._build_models_tab()
        self._build_calibration_tab()
        self._build_operators_tab()
        self._build_monitor_tab()
        self._build_plc_settings_tab()

    def _build_status_bar(self) -> None:
        status_bar = ctk.CTkFrame(self, fg_color=APP_BG, corner_radius=0)
        status_bar.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))
        status_bar.columnconfigure(0, weight=1)
        ctk.CTkLabel(status_bar, textvariable=self.status_var, text_color=TEXT_SECONDARY).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(status_bar, textvariable=self.refresh_time_var, text_color=TEXT_SECONDARY, font=("Segoe UI", 9)).grid(row=0, column=1, sticky="e")

    def _make_scrollable_body(self, tab: ttk.Frame, key: str) -> tk.Frame:
        scroller = ScrollableFrame(tab)
        scroller.pack(fill="both", expand=True)
        self._tab_scrollers[key] = scroller
        scroller.body.columnconfigure(0, weight=1)
        scroller.body.rowconfigure(0, weight=1)
        return scroller.body

    def _build_presets_tab(self) -> None:
        self._templates_tab = TemplatesTab(self, self.presets_tab)

    def _build_operators_tab(self) -> None:
        OperatorsTab(self, self.operators_tab)

    def _build_monitor_tab(self) -> None:
        ResultsTab(self, self.monitor_tab)

    def _build_plc_settings_tab(self) -> None:
        MachineSettingsTab(self, self.plc_settings_tab)
    # ------------------------------------------------------------------
    # Refresh and render
    def refresh_all(self) -> None:
        self.refresh_presets()
        self.refresh_template_options()
        self.refresh_model_options()
        self.refresh_models_training()
        self.refresh_operators()
        self.refresh_monitor()
        self._admin_refresh_datasets()
        self._admin_refresh_training_jobs()
        self.refresh_model_options()
        self._admin_refresh_calibration()

    def refresh_presets(self) -> None:
        self._set_status("Loading presets...")

        def _load():
            return {
                "deployments": self.api.list_deployments(),
                "templates": self.api.list_templates(),
            }

        def _done(payload, error):
            if error:
                self._set_status(f"Preset load error: {error}")
                return
            data = payload or {}
            self._deployments_cache = list(data.get("deployments") or [])
            self._templates_cache = list(data.get("templates") or [])
            self._render_presets()
            self._update_overview_cards()
            self._record_refresh("Presets")
            active_count = sum(1 for item in self._deployments_cache if bool(item.get("is_active", True)))
            self._set_status(f"Loaded {len(self._templates_cache)} templates and {active_count} active deployments.")

        run_async(self, _load, callback=_done)

    def refresh_template_options(self) -> None:
        def _done(items, error):
            if error:
                return
            self._templates_cache = list(items or [])
            if hasattr(self, "preset_table"):
                self._render_presets()

        run_async(self, self.api.list_templates, callback=_done)

    def refresh_model_options(self) -> None:
        """Refresh model dropdown in template tab AND model list in models tab."""
        def _done(items, error):
            if error:
                self._template_model_lookup = {}
                self.preset_model_selector.configure(values=[])
                self._model_lookup = {}
                self._admin_model_list.delete(0, "end")
                return
            models = list(items or [])

            # ── Template dropdown lookup ──
            self._template_model_lookup = {}
            values: list[str] = []
            for item in models:
                raw_name = item.get("name") or item.get("path") or "model"
                raw_path = str(item.get("path") or "").strip()
                raw_runtime = str(item.get("runtime") or "ultralytics").strip()
                label = f"{item.get('id')} | {raw_name} | {raw_runtime}"
                if raw_path:
                    label = f"{label} | {raw_path}"
                values.append(label)
                self._template_model_lookup[label] = dict(item)
            self.preset_model_selector.configure(values=values)
            if values and not self.preset_model_choice_var.get().strip():
                self.preset_model_choice_var.set(values[0])
                self._on_preset_model_selected()

            # ── Models page listbox ──
            self._models_cache = models
            self._admin_model_list.delete(0, "end")
            self._model_lookup = {}
            for item in models:
                name = str(item.get("name") or "")
                path = str(item.get("path") or "")
                status = str(item.get("status") or "")
                runtime = str(item.get("runtime") or "ultralytics")
                display = f"{name} | {runtime} | {status} | {path}"
                self._admin_model_list.insert("end", display)
                self._model_lookup[display] = dict(item)

        run_async(self, self.api.list_models, callback=_done)

    def refresh_models_training(self) -> None:
        def _load():
            return {
                "models": self.api.list_models(),
                "datasets": self.api.list_datasets(),
                "base_models": self.api.list_base_models(),
                "training_jobs": self.api.list_training_jobs(),
                "augment_jobs": self.api.list_augment_jobs(),
            }

        def _done(payload, error):
            if error:
                self._set_status(f"Models/training load error: {error}")
                return
            data = payload or {}
            self._models_cache = list(data.get("models") or [])
            self._datasets_cache = list(data.get("datasets") or [])
            self._base_models_cache = list(data.get("base_models") or [])
            self._training_jobs_cache = list(data.get("training_jobs") or [])
            self._augment_jobs_cache = list(data.get("augment_jobs") or [])
            self._render_models_training()
            self._record_refresh("Models")

        run_async(self, _load, callback=_done)

    def _render_models_training(self) -> None:
        if hasattr(self, "models_table"):
            self._clear_tree(self.models_table)
            for item in self._models_cache:
                self.models_table.insert(
                    "",
                    "end",
                    iid=str(item.get("id")),
                    values=(
                        item.get("id"),
                        _safe_text(item.get("name")),
                        _safe_text(item.get("runtime")),
                        _safe_text(item.get("path")),
                    ),
                )
        if hasattr(self, "datasets_table"):
            self._clear_tree(self.datasets_table)
            for item in self._datasets_cache:
                dataset_id = _safe_text(item.get("id"))
                self.datasets_table.insert(
                    "",
                    "end",
                    iid=dataset_id,
                    values=(dataset_id, _safe_text(item.get("name")), _safe_text(item.get("status"), "ready")),
                )
        self._base_model_lookup = {}
        base_model_values: list[str] = []
        for item in self._base_models_cache:
            label = str(item.get("display_label") or item.get("display_name") or item.get("id") or "").strip()
            if not label:
                continue
            base_model_values.append(label)
            self._base_model_lookup[label] = dict(item)
        if hasattr(self, "_admin_train_base_model_combo"):
            self._admin_train_base_model_combo.configure(values=base_model_values)
            if base_model_values and self._admin_train_base_model_var.get().strip() not in base_model_values:
                self._admin_train_base_model_var.set(base_model_values[0])
        if hasattr(self, "training_base_model_var") and base_model_values:
            if self.training_base_model_var.get().strip() not in base_model_values:
                self.training_base_model_var.set(base_model_values[0])
        if hasattr(self, "training_jobs_table"):
            self._clear_tree(self.training_jobs_table)
            for item in self._training_jobs_cache:
                progress = item.get("progress_percent")
                self.training_jobs_table.insert(
                    "",
                    "end",
                    iid=str(item.get("id")),
                    values=(
                        _safe_text(item.get("id")),
                        _safe_text(item.get("dataset_id")),
                        _safe_text(item.get("status")),
                        f"{progress}%" if progress is not None else "-",
                        _safe_text(item.get("base_model_display_name") or item.get("base_model")),
                    ),
                )

    def refresh_operators(self) -> None:
        self._set_status("Loading operators...")

        def _done(items, error):
            if error:
                self._set_status(f"Operator load error: {error}")
                return
            self._users_cache = list(items or [])
            self._render_operators()
            self._update_overview_cards()
            self._record_refresh("Operators")

        run_async(self, self.api.list_users, callback=_done)

    def refresh_monitor(self) -> None:
        params = self._monitor_filters()
        self._set_status("Loading monitor...")

        def _load():
            return {
                "summary": self.api.dashboard_summary(params),
                "results": self.api.list_inspections(params),
            }

        def _done(result, error):
            if error:
                self._set_status(f"Monitor load error: {error}")
                return
            payload = result or {}
            self._results_cache = list(payload.get("results") or [])
            self._render_monitor_results()
            self._render_monitor_summary(payload.get("summary") or {})
            self._update_overview_cards()
            self._record_refresh("Monitor")
            self._set_status(f"Loaded {len(self._results_cache)} recent inspections.")

        run_async(self, _load, callback=_done)

    def _render_presets(self) -> None:
        self._clear_tree(self.preset_table)
        active_items = [item for item in self._deployments_cache if bool(item.get("is_active", True))]
        active_template_ids = {
            int(item.get("template_id") or 0)
            for item in active_items
            if int(item.get("template_id") or 0) > 0
        }
        if not active_items and not self._templates_cache:
            self.preset_table.insert("", "end", iid="__empty__", values=("-", "No presets.", "", "", ""))
            return
        for item in active_items:
            # Get latest template name from cache (falls back to deployment snapshot)
            template_id = int(item.get("template_id") or 0)
            latest_name = item.get("template_name") or ""
            if template_id > 0:
                tpl = next(
                    (t for t in self._templates_cache if int(t.get("id") or 0) == template_id),
                    None,
                )
                if tpl:
                    latest_name = tpl.get("name") or latest_name
            # Get validator mode from template (prefer top-level `mode` field)
            _vm = "sticker"
            if template_id > 0 and tpl:
                _vm = str(tpl.get("mode") or tpl.get("sticker", {}).get("validator_mode") or "ml_detection")
            _mode_label = mode_label(_vm)
            self.preset_table.insert(
                "",
                "end",
                iid=f"dep:{item.get('id')}",
                values=(
                    f"D{item.get('id')}",
                    _safe_text(latest_name),
                    _safe_text(item.get("template_version_id")),
                    _mode_label,
                    "ACTIVE",
                ),
            )
        for item in self._templates_cache:
            template_id = int(item.get("id") or 0)
            if template_id in active_template_ids:
                continue
            _vm = str(item.get("mode") or item.get("sticker", {}).get("validator_mode") or "ml_detection")
            _mode_label = mode_label(_vm)
            self.preset_table.insert(
                "",
                "end",
                iid=f"tpl:{template_id}",
                values=(
                    f"T{template_id}",
                    _safe_text(item.get("name")),
                    _safe_text(item.get("version_id") or item.get("current_version_id")),
                    _mode_label,
                    _safe_text(item.get("lifecycle_status") or _format_status(item.get("is_active", True))),
                ),
            )

    def _render_operators(self) -> None:
        self._clear_tree(self.users_table)
        users = list(self._users_cache)
        if not users:
            self.users_table.insert("", "end", iid="__empty__", values=("-", "No users.", "", "", ""))
            return
        for item in users:
            rfid_status = "Bound" if item.get("rfid_bound") else "Unbound"
            if item.get("rfid_uid_last4"):
                rfid_status = f"*{_safe_text(item.get('rfid_uid_last4'))}"
            self.users_table.insert(
                "",
                "end",
                iid=str(item.get("id")),
                values=(
                    item.get("id"),
                    _safe_text(item.get("username")),
                    _safe_text(item.get("role")),
                    _format_status(item.get("is_active", True)),
                    rfid_status,
                ),
            )

    # ------------------------------------------------------------------
    # User CRUD handlers
    # ------------------------------------------------------------------

    def _on_user_double_click(self, event) -> None:
        """Load selected user into the edit form."""
        sel = self.users_table.selection()
        if not sel:
            return
        user_id = int(sel[0])
        users = {int(u.get("id")): u for u in self._users_cache}
        user = users.get(user_id)
        if user is None:
            return
        self.operator_edit_id = user_id
        self.operator_edit_username_var.set(user.get("username", ""))
        self.operator_edit_role_var.set(str(user.get("role") or "operator").strip().lower())
        self.operator_form_title.configure(text=f"Edit User #{user_id}")
        self.operator_form_hint.configure(text="Change the role, optionally reset password, or delete this user.")
        self.operator_username_var.set(user.get("username", ""))
        self.operator_password_var.set("")
        self.operator_role_var.set(str(user.get("role") or "operator").strip().lower())
        self.operator_save_btn.configure(text="Save Changes")
        self.operator_cancel_btn.configure(state="normal")
        self.operator_delete_btn.configure(state="normal")

    def _on_cancel_edit(self) -> None:
        """Reset the form back to create mode."""
        self.operator_edit_id = None
        self.operator_username_var.set("")
        self.operator_password_var.set("")
        self.operator_role_var.set("operator")
        self.operator_form_title.configure(text="Add User")
        self.operator_form_hint.configure(text="Create a new user, then bind RFID below.")
        self.operator_save_btn.configure(text="Create User")
        self.operator_cancel_btn.configure(state="disabled")
        self.operator_delete_btn.configure(state="disabled")
        # Reset unified bind state
        self.bind_target_user_id = None
        self.bind_target_label.configure(text="Select a user from the list")
        self.unified_rfid_var.set("")
        self.bind_rfid_status.configure(text="")

    def _on_user_selected(self, event=None) -> None:
        """Single-click on table: set bind target user."""
        sel = self.users_table.selection()
        if not sel:
            return
        user_id = int(sel[0])
        users = {int(u.get("id")): u for u in self._users_cache}
        user = users.get(user_id)
        if user is None:
            return
        self.bind_target_user_id = user_id
        name = _safe_text(user.get("username"))
        role = _safe_text(user.get("role"))
        self.bind_target_label.configure(text=f"Target: {name} ({role}) #{user_id}")
        self.after_idle(self.unified_rfid_entry.focus_set)

    def _on_bind_rfid(self) -> None:
        """Bind scanned RFID to the selected user."""
        if self.bind_target_user_id is None:
            self.bind_rfid_status.configure(text="Select a user first.", text_color="orange")
            return
        rfid_uid = self.unified_rfid_var.get().strip()
        if not rfid_uid:
            self.bind_rfid_status.configure(text="Scan RFID card first.", text_color="orange")
            return
        try:
            self.api.bind_user_rfid(self.bind_target_user_id, rfid_uid)
        except Exception as exc:
            self.bind_rfid_status.configure(text=str(exc), text_color="red")
            return
        self.unified_rfid_var.set("")
        self.bind_rfid_status.configure(text="RFID bound successfully!", text_color="green")
        self.refresh_operators()

    def _on_clear_rfid(self) -> None:
        """Clear RFID binding from the selected user."""
        if self.bind_target_user_id is None:
            return
        if not messagebox.askyesno("Clear RFID", "Remove RFID binding from this user?"):
            return
        try:
            self.api.clear_user_rfid(self.bind_target_user_id)
        except Exception as exc:
            self.bind_rfid_status.configure(text=str(exc), text_color="red")
            return
        self.bind_rfid_status.configure(text="RFID cleared.", text_color=TEXT_SECONDARY)
        self.refresh_operators()

    def _on_save_user(self) -> None:
        """Create new user or update existing user's role/password."""
        username = self.operator_username_var.get().strip()
        if not username:
            messagebox.showerror("Users", "Username is required.")
            return
        password = self.operator_password_var.get().strip()
        if password and len(password) < 6:
            messagebox.showerror("Users", "Password must be at least 6 characters.")
            return

        if self.operator_edit_id is not None:
            # Edit mode — update role, and optionally reset password
            user_id = self.operator_edit_id
            new_role = self.operator_role_var.get().strip()
            try:
                self.api.change_user_role(user_id, new_role)
                if password:
                    self.api.reset_user_password(user_id, password)
            except Exception as exc:
                messagebox.showerror("Users", str(exc))
                return
            self._on_cancel_edit()
            self.refresh_operators()
            suffix = " and password reset" if password else ""
            self._set_status(f"User #{user_id} role changed to {new_role}{suffix}.")
        else:
            # Create mode — password required, no RFID here, user will bind below
            if not password:
                messagebox.showerror("Users", "Password is required.")
                return
            role = self.operator_role_var.get().strip()
            try:
                created = self.api.create_user({
                    "username": username,
                    "password": password,
                    "role": role,
                })
                user_id = int(created.get("id") or 0)
                if user_id <= 0:
                    raise ValueError("API did not return a valid user id.")
            except Exception as exc:
                messagebox.showerror("Users", str(exc))
                return
            self.operator_username_var.set("")
            self.operator_password_var.set("")
            self.operator_role_var.set("operator")
            self.refresh_operators()
            # Auto-select the newly created user in the table and focus RFID entry
            self.bind_target_user_id = user_id
            created_user = {int(u.get("id")): u for u in self._users_cache}.get(user_id, {})
            name = _safe_text(created_user.get("username") or username)
            self.bind_target_label.configure(text=f"Target: {name} ({role}) #{user_id}")
            self.after_idle(self.unified_rfid_entry.focus_set)
            self._set_status(f"User {username} created. Now scan RFID to bind.")

    def _on_delete_user(self) -> None:
        """Delete the selected user after confirmation."""
        if self.operator_edit_id is None:
            return
        user_id = self.operator_edit_id
        username = self.operator_username_var.get().strip()
        if not messagebox.askyesno(
            "Delete User",
            f"Permanently delete user '{username}' (#{user_id})?\n\nThis action cannot be undone.",
        ):
            return
        try:
            self.api.delete_user(user_id)
        except Exception as exc:
            messagebox.showerror("Users", str(exc))
            return
        self._on_cancel_edit()
        self.refresh_operators()
        self._set_status(f"User '{username}' (#{user_id}) deleted.")

    def _render_monitor_results(self) -> None:
        self._clear_tree(self.results_table)
        self._clear_tree(self.reject_table)
        if not self._results_cache:
            self.results_table.insert("", "end", iid="__empty__", values=("-", "No inspection data.", "", "", "", "", "", ""))
            self.monitor_context_var.set("No inspections found for current filter.")
            return
        for item in self._results_cache:
            decision = str(item.get("decision") or item.get("decision_code") or "").strip().upper()
            reason = item.get("reject_reason_code") or ("OK" if decision == "ACCEPT" else "-")
            # Extract mode from validation_details (data1/data2 convention or details field)
            _vd = item.get("validation_details") or {}
            _mode = _vd.get("mode", "sticker")
            _mode_label = mode_label(_mode)
            self.results_table.insert(
                "",
                "end",
                iid=str(item.get("id")),
                values=(
                    item.get("id"),
                    _format_timestamp(item.get("inspected_at")),
                    _safe_text(decision),
                    _mode_label,
                    _safe_text(item.get("part_name")),
                    _safe_text(item.get("line_id")),
                    _safe_text(item.get("station_id")),
                    _safe_text(item.get("push_status")),
                    _safe_text(reason),
                ),
            )
            if decision == "REJECT":
                self.reject_table.insert(
                    "",
                    "end",
                    iid=str(item.get("id")),
                    values=(
                        item.get("id"),
                        _safe_text(item.get("part_name")),
                        _safe_text(reason),
                        _format_timestamp(item.get("inspected_at")),
                    ),
                )
        self.monitor_context_var.set(f"Showing {len(self._results_cache)} recent inspections.")

    def _render_monitor_summary(self, summary: dict) -> None:
        total = int(summary.get("total") or summary.get("total_count") or summary.get("total_inspections") or len(self._results_cache) or 0)
        accept = int(summary.get("accept") or summary.get("accept_count") or summary.get("total_accept") or sum(1 for item in self._results_cache if str(item.get("decision") or "").upper() == "ACCEPT"))
        reject = int(summary.get("reject") or summary.get("reject_count") or summary.get("total_reject") or sum(1 for item in self._results_cache if str(item.get("decision") or "").upper() == "REJECT"))
        reject_rate = (reject / total * 100.0) if total else 0.0
        pending_pushes = sum(1 for item in self._results_cache if str(item.get("push_status") or "").lower() == "pending")
        failed_pushes = sum(1 for item in self._results_cache if str(item.get("push_status") or "").lower() == "failed")
        self.monitor_cards["total"].set_value(total)
        self.monitor_cards["accept"].set_value(accept)
        self.monitor_cards["reject"].set_value(reject)
        self.monitor_cards["reject_rate"].set_value(f"{reject_rate:.1f}%")
        self.monitor_cards["pending"].set_value(pending_pushes, "queued")
        self.monitor_cards["failed"].set_value(failed_pushes, "retry needed")
        self.admin_cards["accept"].set_value(accept)
        self.admin_cards["reject"].set_value(reject)

    # ── Centralized form defaults ──
    # All form state variables must be registered here. Adding a new preset_*
    # variable without registering it triggers _assert_form_defaults() at startup.
    FORM_DEFAULTS = {
        # tk variables
        "preset_name_var": "",
        "preset_description_var": "",
        "preset_validator_mode_var": "sticker",
        "preset_model_choice_var": "",
        "preset_model_path_var": "",
        "preset_model_meta_path_var": "",
        "preset_model_classes_var": "",
        "preset_runtime_var": "auto",
        "preset_conf_threshold_var": "0.25",
        "preset_expected_class_var": "",
        "preset_max_tilt_var": "",
        "preset_tilt_gate_var": False,
        "preset_gap_threshold_var": "0.85",
        "preset_part_ready_method_var": "gap_template_match",
        "preset_mean_max_var": "105.0",
        "preset_std_max_var": "35.0",
        "preset_min_match_ratio_var": "0.5",
        "preset_camera_index_var": "0",
        "preset_camera_rotation_var": "0",
        "preset_roi_choice_var": "Part Ready ROI",
        "preset_part_ready_source_var": "sensor",
        # complex types
        "preset_component_rois": lambda: [],
        "preset_defect_rois": lambda: [],
    }

    def _reset_preset_form(self) -> None:
        """Reset ALL form fields to defaults. Called before loading a new preset or on New Preset."""
        for key, default in self.FORM_DEFAULTS.items():
            obj = getattr(self, key, None)
            if obj is None:
                continue
            if isinstance(obj, tk.Variable):
                if isinstance(default, bool):
                    obj.set(bool(default))
                else:
                    obj.set(str(default))
            elif isinstance(obj, list):
                # Create a NEW list — do NOT .clear() the old one (widgets may share references)
                setattr(self, key, default() if callable(default) else list(default))
            elif hasattr(obj, "set"):
                obj.set(str(default))
        # Reset visual children
        self._preset_roi_image_path = ""
        if hasattr(self, "preset_roi_picker"):
            try:
                self.preset_roi_picker.clear()
                self.preset_roi_picker.set_component_rois([])
                self.preset_roi_picker.set_defect_rois([])
                self._sync_preset_roi_picker(part_ready_rotation=0.0, sticker_rotation=0.0)
            except Exception:
                pass
        # Reset ROI entry fields (prefix not preset_)
        for var_attr in ("part_ready_roi_x_var", "part_ready_roi_y_var",
                         "part_ready_roi_w_var", "part_ready_roi_h_var",
                         "sticker_roi_x_var", "sticker_roi_y_var",
                         "sticker_roi_w_var", "sticker_roi_h_var"):
            v = getattr(self, var_attr, None)
            if v is not None:
                defaults = {"x": "0.2", "y": "0.2", "w": "0.6", "h": "0.6"}
                key = var_attr.split("_")[-2]  # x, y, w, h
                default = defaults.get(key, "0.2") if "sticker" in var_attr else defaults.get(key, "0.2")
                if "part_ready" in var_attr:
                    default = defaults.get(key, "0.2") if key == "x" or key == "y" else defaults.get(key, "0.25")
                v.set(default)
        # Refresh editors
        if hasattr(self, "_templates_tab") and self._templates_tab:
            try:
                self._templates_tab._refresh_comp_roi_editor(self)
                self._templates_tab._refresh_defect_editor(self)
                self._templates_tab._update_roi_selector_dropdown(self)
            except Exception:
                pass
        # Clear gap ref status
        if hasattr(self, "gap_ref_status_label"):
            try:
                self.gap_ref_status_label.configure(
                    text="Referensi: belum dikonfigurasi", foreground="gray")
            except Exception:
                pass

    def _assert_form_defaults(self) -> None:
        """Assert all preset_* variables are registered in FORM_DEFAULTS.
        Call once at end of __init__ to catch missing registrations.
        """
        missing = []
        for name in dir(self):
            if not name.startswith("preset_"):
                continue
            if name in self.FORM_DEFAULTS:
                continue
            obj = getattr(self, name, None)
            if isinstance(obj, (tk.Variable, list, dict)):
                missing.append(name)
        if missing:
            raise AssertionError(
                f"Form variables not registered in FORM_DEFAULTS: {missing}. "
                "Add them to FORM_DEFAULTS to prevent state-leak bugs."
            )

    # ------------------------------------------------------------------
    # Preset behavior
    def reset_preset_wizard(self) -> None:
        self.current_template_id = None
        self.current_template_version_id = None
        self._editing_deployment_id = None
        self._template_detail_cache = None
        self._refresh_preset_action_button()
        if hasattr(self, "preset_table"):
            for item_id in self.preset_table.selection():
                self.preset_table.selection_remove(item_id)
            self.preset_table.focus("")
        self._reset_preset_form()
        self._set_status("Preset wizard reset.")

    def _on_preset_selected(self, _event=None) -> None:
        selected_kind, selected_id = self._selected_preset_row()
        if selected_kind is None or selected_id is None:
            return
        if selected_kind == "template":
            self._editing_deployment_id = None
            try:
                detail = self.api.get_template(selected_id)
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"Preset detail load failed: {exc}")
                return
            self._apply_preset_detail(detail, deployment=None)
            self._set_status(f"Loaded template {_safe_text(detail.get('name'))}.")
            return

        deployment = next((item for item in self._deployments_cache if int(item.get("id") or 0) == selected_id), None)
        if not deployment:
            return
        self._editing_deployment_id = selected_id

        template_id = int(deployment.get("template_id") or 0)
        version_id = int(deployment.get("template_version_id") or 0)
        # Always load current version for editing; deployment snapshot is only for display
        if template_id:
            try:
                detail = self.api.get_template(template_id)
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"Preset detail load failed: {exc}")
                return
        elif version_id:
            # Fallback: load version snapshot if no template_id
            try:
                detail = self.api.get_template_version(version_id)
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"Preset detail load failed: {exc}")
                return
        else:
            return
        self._apply_preset_detail(detail, deployment=deployment)
        self._set_status(f"Loaded preset {_safe_text(deployment.get('template_name'))}.")

    def _apply_preset_detail(self, detail: dict, *, deployment: dict | None = None) -> None:
        # Reset form first to purge any state from previous preset/mode
        self._reset_preset_form()
        # Cache the template detail for mode-switch freshness checks
        self._template_detail_cache = detail
        self.current_template_id = int(detail.get("id") or (deployment or {}).get("template_id") or 0) or None
        self.current_template_version_id = int(detail.get("version_id") or (deployment or {}).get("template_version_id") or 0) or None
        self.preset_name_var.set(str(detail.get("name") or (deployment or {}).get("template_name") or ""))
        self.preset_description_var.set(str(detail.get("description") or ""))
        sticker = detail.get("sticker") or {}
        part_ready = detail.get("part_ready") or {}
        # Restore validator mode — fires _refresh_comp_roi_editor trace
        # Read from new `mode` field first, fall back to legacy sticker.validator_mode
        _vm = mode_from_template(detail)
        self.preset_validator_mode_var.set(mode_to_radio(_vm))
        self.preset_expected_class_var.set(str(sticker.get("expected_class") or ""))
        self.preset_max_tilt_var.set("" if sticker.get("max_tilt_degrees") is None else str(sticker.get("max_tilt_degrees")))
        self.preset_tilt_gate_var.set(bool(sticker.get("tilt_gate_enabled", False)))
        self.preset_gap_threshold_var.set(str(part_ready.get("gap_match_threshold", 0.85)))
        # Part ready method and mean-std thresholds
        _method = str(part_ready.get("method") or "").strip() or "gap_template_match"
        self.preset_part_ready_method_var.set(_method)
        self.preset_mean_max_var.set(str(part_ready.get("mean_max", 105.0)))
        self.preset_std_max_var.set(str(part_ready.get("std_max", 35.0)))
        self.preset_min_match_ratio_var.set(str(part_ready.get("min_match_ratio", 0.5)))
        # Update gap ref status label
        _gap_ref_path = detail.get("gap_ref_path") or part_ready.get("gap_ref_path")
        _gap_ref_type = str(part_ready.get("gap_ref_type") or "").strip() or "raw"
        if _gap_ref_path:
            from pathlib import Path
            if not Path(_gap_ref_path).is_file():
                self.gap_ref_status_label.configure(
                    text="Referensi: file tidak ditemukan", foreground="orange")
            elif _gap_ref_type != "edge_map":
                self.gap_ref_status_label.configure(
                    text="⚠ Format lama — capture ulang diperlukan", foreground="orange")
            else:
                self.gap_ref_status_label.configure(
                    text="Referensi: edge map ✓", foreground="green")
        else:
            self.gap_ref_status_label.configure(
                text="Referensi: belum dikonfigurasi", foreground="gray")
        camera_cfg = detail.get("camera") or {}
        self.preset_camera_index_var.set(str(camera_cfg.get("camera_index", 0)))
        self.preset_camera_rotation_var.set(str(camera_cfg.get("rotation_degrees", 0)))
        self._refresh_preset_action_button()
        part_ready_roi = detail.get("part_ready_roi") or {}
        sticker_roi = detail.get("sticker_roi") or detail.get("roi") or {}
        self.part_ready_roi_x_var.set(str(part_ready_roi.get("x", 0.2)))
        self.part_ready_roi_y_var.set(str(part_ready_roi.get("y", 0.2)))
        self.part_ready_roi_w_var.set(str(part_ready_roi.get("w", 0.25)))
        self.part_ready_roi_h_var.set(str(part_ready_roi.get("h", 0.25)))
        self.sticker_roi_x_var.set(str(sticker_roi.get("x", 0.2)))
        self.sticker_roi_y_var.set(str(sticker_roi.get("y", 0.2)))
        self.sticker_roi_w_var.set(str(sticker_roi.get("w", 0.6)))
        self.sticker_roi_h_var.set(str(sticker_roi.get("h", 0.6)))
        self._sync_preset_roi_picker(
            part_ready_rotation=float(part_ready_roi.get("rotation", 0.0)),
            sticker_rotation=float(sticker_roi.get("rotation", 0.0)),
        )

        vision = detail.get("vision") or {}
        _crit = detail.get("criteria") or {}
        _vm = mode_from_template(detail)
        # Single model selector: load from criteria.default_model_path in defect mode,
        # from vision.model_path in sticker/counter mode
        if _vm == "defect":
            _model_for_selector = str(_crit.get("default_model_path") or vision.get("model_path") or "")
        else:
            _model_for_selector = str(vision.get("model_path") or "")
        self.preset_model_path_var.set(_model_for_selector)
        self.preset_model_meta_path_var.set(str(vision.get("model_meta_path") or ""))
        self.preset_conf_threshold_var.set(str(vision.get("conf_threshold", 0.25)))
        self.preset_runtime_var.set(str(vision.get("runtime") or "auto"))
        self._select_model_label_for_path(_model_for_selector)

        # ── Load component ROIs from detail (criteria first, top-level fallback) ──
        _crit = detail.get("criteria") or {}
        self.preset_component_rois = list(_crit.get("component_rois") or detail.get("component_rois") or [])
        # Load part_ready_source for counter mode (default "sensor" when absent)
        if _vm == "counter":
            self.preset_part_ready_source_var.set(str(_crit.get("part_ready_source") or "sensor"))
        # Additional fallback: if counter mode and ROIs still empty, try to reconstruct
        # from legacy template data. This guards against API responses that may omit criteria.
        if _vm == "counter" and not self.preset_component_rois:
            # Fallback 1: sticker.validator_mode == "component_count" indicates counter template
            _sticker_vm = str(sticker.get("validator_mode") or "").strip().lower()
            if _sticker_vm == "component_count" and detail.get("component_rois"):
                self.preset_component_rois = list(detail["component_rois"])
            # Fallback 2: check vision.classes for class list (minimal recovery)
            elif _sticker_vm == "component_count" and not self.preset_component_rois:
                # Template is counter mode but no component ROIs found — edge case
                pass  # leave empty, operator can re-add
        self.preset_defect_rois = list(_crit.get("rois") or [])
        if hasattr(self, "_defect_infer_mode_var"):
            self._defect_infer_mode_var.set(str(_crit.get("inference_mode", "whole_part")))

        # Sync component ROIs to the ROI picker canvas
        # Build ROI geometry dicts from the component ROI list
        _comp_rois_for_picker = []
        for cr in self.preset_component_rois:
            _roi = cr.get("roi") or {}
            _comp_rois_for_picker.append({
                "name": cr.get("name", "ROI"),
                "x": _roi.get("x", 0.0),
                "y": _roi.get("y", 0.0),
                "w": _roi.get("w", 1.0),
                "h": _roi.get("h", 1.0),
                "rotation": _roi.get("rotation", 0.0),
            })
        self.preset_roi_picker.set_component_rois(_comp_rois_for_picker)

        # Refresh the component ROI editor and defect editor widgets
        if hasattr(self, "_templates_tab") and self._templates_tab:
            self._templates_tab._refresh_comp_roi_editor(self)
            self._templates_tab._refresh_defect_editor(self)
            self._templates_tab._update_roi_selector_dropdown(self)

    def _on_preset_model_selected(self, _event=None) -> None:
        item = self._template_model_lookup.get(self.preset_model_choice_var.get().strip())
        if not item:
            return
        self.preset_model_path_var.set(str(item.get("path") or ""))
        self.preset_model_meta_path_var.set(str(item.get("meta_path") or ""))
        self.preset_runtime_var.set(str(item.get("runtime") or "auto"))

    def _select_model_label_for_path(self, model_path: str) -> None:
        normalized = str(model_path or "").strip().lower()
        if not normalized:
            return
        for label, item in self._template_model_lookup.items():
            if str(item.get("path") or "").strip().lower() == normalized:
                self.preset_model_choice_var.set(label)
                return

    # ── HSV color picker from image ──

    # Visual preset ROI picker

    def _preset_roi_kind(self) -> str | None:
        choice = self.preset_roi_choice_var.get().strip()
        if choice == "Part Ready ROI":
            return "part_ready"
        if choice == "Sticker ROI":
            return "sticker"
        if choice.startswith("Component:"):
            # Extract index from the dropdown position, not from name match
            # This avoids name mismatch issues
            try:
                values = self.preset_roi_selector.cget("values")
                for i, val in enumerate(values):
                    if val == choice:
                        # Dropdown: [Part Ready ROI(0), Component:A(1), ...]
                        # Component index = dropdown index - 1 (offset for Part Ready ROI)
                        comp_idx = i - 1
                        if 0 <= comp_idx < len(self.preset_component_rois):
                            return f"component:{comp_idx}"
            except Exception:
                pass
        if choice.startswith("Defect:"):
            try:
                values = self.preset_roi_selector.cget("values")
                for i, val in enumerate(values):
                    if val == choice:
                        # Dropdown: [Part Ready ROI(0), Defect:A(1), ...]
                        defect_idx = i - 1
                        if 0 <= defect_idx < len(self.preset_defect_rois):
                            return f"defect:{defect_idx}"
            except Exception:
                pass
        return None

    def _roi_payload_from_vars(
        self,
        x_var: tk.StringVar,
        y_var: tk.StringVar,
        w_var: tk.StringVar,
        h_var: tk.StringVar,
        *,
        defaults: dict[str, float],
    ) -> dict[str, float]:
        return {
            "x": _float_or_default(x_var.get(), defaults["x"]),
            "y": _float_or_default(y_var.get(), defaults["y"]),
            "w": _float_or_default(w_var.get(), defaults["w"]),
            "h": _float_or_default(h_var.get(), defaults["h"]),
        }

    def _sync_preset_roi_picker(self, *,
                                part_ready_rotation: float | None = None,
                                sticker_rotation: float | None = None) -> None:
        if not hasattr(self, "preset_roi_picker"):
            return
        mode = self.preset_validator_mode_var.get()
        if mode == "component_count":
            # In component mode, only sync active ROI kind, don't touch positions
            self.preset_roi_picker.set_active_roi(self._preset_roi_kind())
            return
        _pr_rot = (part_ready_rotation if part_ready_rotation is not None
                   else self.preset_roi_picker.get_roi("part_ready").get("rotation", 0.0))
        _st_rot = (sticker_rotation if sticker_rotation is not None
                   else self.preset_roi_picker.get_roi("sticker").get("rotation", 0.0))
        self.preset_roi_picker.set_rois(
            part_ready_roi={
                **self._roi_payload_from_vars(
                    self.part_ready_roi_x_var,
                    self.part_ready_roi_y_var,
                    self.part_ready_roi_w_var,
                    self.part_ready_roi_h_var,
                    defaults={"x": 0.2, "y": 0.2, "w": 0.25, "h": 0.25},
                ),
                "rotation": _pr_rot,
            },
            sticker_roi={
                **self._roi_payload_from_vars(
                    self.sticker_roi_x_var,
                    self.sticker_roi_y_var,
                    self.sticker_roi_w_var,
                    self.sticker_roi_h_var,
                    defaults={"x": 0.2, "y": 0.2, "w": 0.6, "h": 0.6},
                ),
                "rotation": _st_rot,
            },
        )
        self.preset_roi_picker.set_active_roi(self._preset_roi_kind())

    def _on_preset_roi_selected(self, _event=None) -> None:
        kind = self._preset_roi_kind()
        if kind is not None and kind.startswith("component:"):
            # Component mode: just set active ROI, don't overwrite positions
            self.preset_roi_picker.set_active_roi(kind)
        else:
            self._sync_preset_roi_picker()

    def _on_preset_roi_changed(self, kind: str, roi: dict) -> None:
        # Ignore component ROIs — they are handled by _on_comp_roi_picker_changed
        if kind is not None and kind.startswith("component:"):
            return
        target = (
            (self.part_ready_roi_x_var, self.part_ready_roi_y_var, self.part_ready_roi_w_var, self.part_ready_roi_h_var)
            if kind == "part_ready"
            else (self.sticker_roi_x_var, self.sticker_roi_y_var, self.sticker_roi_w_var, self.sticker_roi_h_var)
        )
        for key, var in zip(("x", "y", "w", "h"), target, strict=True):
            value = f"{float(roi.get(key, 0.0)):.4f}".rstrip("0").rstrip(".")
            var.set(value or "0")

    def _pick_preset_roi_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Pick ROI Reference Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")],
        )
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("ROI Picker", f"Gagal membaca gambar: {path}")
            return
        self._preset_roi_image_path = path
        self.preset_roi_picker.load_image(frame)
        self._sync_preset_roi_picker()
        self._set_status(f"ROI reference loaded: {Path(path).name}")

    def _capture_preset_roi_from_camera(self) -> None:
        try:
            from client_tk.app.services.camera_capture import CameraCaptureService
        except ImportError:
            messagebox.showerror("Camera", "Camera service not available.")
            return

        cam_idx = int(_float_or_default(self.preset_camera_index_var.get(), 0))
        cap_service = CameraCaptureService()
        try:
            cap_service.start(cam_idx)
            import time
            time.sleep(0.5)
            frame = cap_service.get_latest_frame()
            if frame is None:
                messagebox.showwarning("Camera", "Camera returned no frame in time. Try again.")
                return
            # Apply camera rotation from preset config
            try:
                import cv2
                import numpy as np
                _rot = float(_float_or_default(self.preset_camera_rotation_var.get(), 0))
                if _rot != 0.0:
                    h, w = frame.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, -_rot, 1.0)
                    cos_a = abs(M[0, 0])
                    sin_a = abs(M[0, 1])
                    new_w = int(h * sin_a + w * cos_a)
                    new_h = int(h * cos_a + w * sin_a)
                    M[0, 2] += (new_w - w) / 2
                    M[1, 2] += (new_h - h) / 2
                    frame = cv2.warpAffine(frame, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)
            except Exception:
                pass  # rotation failed, use original frame
            self._preset_roi_image_path = ""
            self.preset_roi_picker.load_image(frame.copy())
            self._sync_preset_roi_picker()
            self._set_status(f"Captured ROI reference from camera {cam_idx}.")
        except Exception as exc:
            messagebox.showerror("Camera", f"Failed to capture from camera {cam_idx}: {exc}")
        finally:
            try:
                cap_service.stop()
            except Exception:
                pass

    def _stop_live_camera_if_running(self) -> None:
        """Stop live camera if it's running (safety for MSMF camera conflict)."""
        picker = getattr(self, "preset_roi_picker", None)
        if picker is not None and getattr(picker, "_cam_running", False):
            try:
                picker.stop_live_camera()
            except Exception:
                pass
            if hasattr(self, "_live_cam_btn"):
                self._live_cam_btn.configure(text="Start Live Camera")

    def shutdown(self) -> None:
        """Cleanup on screen teardown: stop live camera if running."""
        self._stop_live_camera_if_running()

    def _toggle_live_camera(self) -> None:
        """Toggle live camera feed on the ROI picker canvas."""
        picker = getattr(self, "preset_roi_picker", None)
        if picker is None:
            messagebox.showwarning("Camera", "ROI picker not initialized.")
            return
        if getattr(picker, "_cam_running", False):
            picker.stop_live_camera()
            self._live_cam_btn.configure(text="Start Live Camera")
            self._set_status("Live camera stopped.")
        else:
            cam_idx = int(_float_or_default(self.preset_camera_index_var.get(), 0))
            try:
                picker.start_live_camera(cam_idx)
                self._live_cam_btn.configure(text="Stop Live Camera")
                self._set_status(f"Live camera {cam_idx} started. Drag ROIs on the live feed.")
            except Exception as exc:
                messagebox.showerror("Camera", f"Failed to start camera {cam_idx}: {exc}")

    def _reset_preset_roi(self) -> None:
        kind = self._preset_roi_kind()
        if kind == "part_ready":
            self.part_ready_roi_x_var.set("0.2")
            self.part_ready_roi_y_var.set("0.2")
            self.part_ready_roi_w_var.set("0.25")
            self.part_ready_roi_h_var.set("0.25")
            self._set_status("Part Ready ROI reset to default.")
        else:
            self.sticker_roi_x_var.set("0.2")
            self.sticker_roi_y_var.set("0.2")
            self.sticker_roi_w_var.set("0.6")
            self.sticker_roi_h_var.set("0.6")
            self._set_status("Sticker ROI reset to default.")
        self._sync_preset_roi_picker()

    def _refresh_preset_action_button(self) -> None:
        """Update preset action button text/command based on current_template_id."""
        if self.current_template_id:
            self._preset_action_btn.configure(
                text=f"Update Template #{self.current_template_id}",
                command=self.update_preset_only,
                fg_color="#3b82f6",
                hover_color="#2563eb",
                text_color="#ffffff",
            )
        else:
            self._preset_action_btn.configure(
                text="Save & Deploy Preset",
                command=self.save_and_deploy_preset,
                fg_color=ACCENT,
                hover_color=ACCENT_HOVER,
                text_color=TEXT_ON_ACCENT,
            )

    def update_preset_only(self) -> None:
        """Update current template parameters without creating a new version or deploying."""
        if not self.current_template_id:
            messagebox.showwarning("Preset", "No template loaded. Create a new preset first.")
            return
        try:
            payload = self._preset_payload()
        except ValueError as exc:
            messagebox.showerror("Preset", str(exc))
            return
        try:
            saved = self.api.update_template(self.current_template_id, payload, update_current_version=True)
            template_id = int(saved.get("id") or self.current_template_id or 0)
            version_id = int(saved.get("version_id") or saved.get("current_version_id") or 0)
            self.current_template_id = template_id
            self.current_template_version_id = version_id
            # Update template_name in existing active deployment records
            new_name = payload.get("name", "").strip()
            if new_name and hasattr(self, "_deployments_cache"):
                for dep in self._deployments_cache:
                    if int(dep.get("template_id") or 0) == template_id and dep.get("is_active"):
                        try:
                            self.api.update_deployment(
                                int(dep.get("id") or 0),
                                {"template_name": new_name},
                            )
                        except Exception:
                            pass
            # Reload exact version detail to wizard form so values reflect the update
            _reload_ok = True
            _reload_error = ""
            # Backup component ROIs before clearing (for roundtrip recovery)
            _backup_comp_rois = list(self.preset_component_rois) if self.preset_component_rois else []
            try:
                if version_id:
                    detail = self.api.get_template_version(version_id)
                else:
                    detail = self.api.get_template(template_id)
                self._apply_preset_detail(detail, deployment=None)
                # After reloading, if counter mode ROIs are still empty, restore from backup
                if not self.preset_component_rois and _backup_comp_rois:
                    if mode_from_template(detail) == "counter":
                        self.preset_component_rois = _backup_comp_rois
                        self.preset_roi_picker.set_component_rois([
                            {"name": cr.get("name", "ROI"), **cr.get("roi", {})}
                            for cr in _backup_comp_rois
                        ])
                        if hasattr(self, "_templates_tab") and self._templates_tab:
                            self._templates_tab._refresh_comp_roi_editor(self)
            except Exception as exc:
                _reload_ok = False
                _reload_error = str(exc)
            self.refresh_presets()
            if _reload_ok:
                self._set_status(f"Template #{template_id} v{version_id} updated.")
                messagebox.showinfo("Preset", f"Template #{template_id} updated successfully.")
            else:
                self._set_status(f"Template #{template_id} saved, but detail reload failed: {_reload_error}")
                messagebox.showwarning("Preset", f"Saved OK, but form reload failed: {_reload_error}")
        except Exception as exc:
            messagebox.showerror("Preset", str(exc))
    def save_and_deploy_preset(self) -> None:
        """Save current template as new version and deploy."""
        # Stop live camera if running to prevent thread leak
        _picker = getattr(self, "preset_roi_picker", None)
        if _picker is not None and getattr(_picker, "_cam_running", False):
            try:
                _picker.stop_live_camera()
            except Exception:
                pass
        try:
            payload = self._preset_payload()
        except ValueError as exc:
            messagebox.showerror("Preset", str(exc))
            return
        try:
            is_update = bool(self.current_template_id)
            if is_update:
                saved = self.api.update_template(self.current_template_id, payload)
            else:
                saved = self.api.create_template(payload)
            template_id = int(saved.get("id") or self.current_template_id or 0)
            version_id = int(saved.get("version_id") or saved.get("current_version_id") or 0)
            if not template_id or not version_id:
                raise ValueError("Saved preset did not return template id and version id.")
            # Auto-transition lifecycle: draft → review → approved → published
            for transition in ("review", "approved", "published"):
                try:
                    self.api.transition_template_lifecycle(template_id, transition, "Auto-transition on deploy")
                except Exception:
                    pass
            deployment = self.api.deploy_template(
                {
                    "template_id": template_id,
                    "template_version_id": version_id,
                }
            )
            # Update template_name in existing active deployment records
            # This keeps the name in sync when renaming a deployed template
            if is_update:
                new_name = payload.get("name", "").strip()
                if new_name and hasattr(self, "_deployments_cache"):
                    for dep in self._deployments_cache:
                        if int(dep.get("template_id") or 0) == template_id and dep.get("is_active"):
                            try:
                                self.api.update_deployment(
                                    int(dep.get("id") or 0),
                                    {"template_name": new_name},
                                )
                            except Exception:
                                pass
        except Exception as exc:
            messagebox.showerror("Preset", str(exc))
            return
        self.current_template_id = template_id
        self.current_template_version_id = version_id
        self._editing_deployment_id = int(deployment.get("id") or 0) or self._editing_deployment_id
        # Backup component ROIs before clearing (for save-roundtrip recovery)
        _backup_comp_rois = list(self.preset_component_rois) if self.preset_component_rois else []
        _backup_defect_rois = list(self.preset_defect_rois) if self.preset_defect_rois else []
        # Re-apply saved data to wizard so UI reflects updated mode/criteria
        try:
            reloaded = self.api.get_template(template_id) if template_id else None
            if reloaded:
                self._apply_preset_detail(reloaded, deployment=deployment)
                # After reloading, if counter mode ROIs are still empty, restore from backup
                _reloaded_mode = mode_from_template(reloaded)
                if _reloaded_mode == "counter" and not self.preset_component_rois and _backup_comp_rois:
                    self.preset_component_rois = _backup_comp_rois
                    self.preset_roi_picker.set_component_rois([
                        {"name": cr.get("name", "ROI"), **cr.get("roi", {})}
                        for cr in _backup_comp_rois
                    ])
                    if hasattr(self, "_templates_tab") and self._templates_tab:
                        self._templates_tab._refresh_comp_roi_editor(self)
                self._set_status(f"Preset deployed — mode: {reloaded.get('mode', 'sticker')}")
            else:
                self._set_status("Preset deployed.")
        except Exception:
            self._set_status("Preset deployed.")
        self.refresh_presets()
        messagebox.showinfo("Preset", "Preset saved and deployed.")



    def _get_existing_gap_ref_path(self) -> str | None:
        """Return existing gap_ref_path from current template detail, if any."""
        if not self.current_template_id:
            return None
        try:
            detail = self.api.get_template(self.current_template_id)
            if detail:
                return detail.get("gap_ref_path") or (detail.get("part_ready") or {}).get("gap_ref_path")
        except Exception:
            pass
        return None

    def _capture_part_ready_ref(self) -> None:
        """Capture reference gap patch from current camera frame."""
        if not self.current_template_id:
            messagebox.showwarning("Reference", "Pilih template terlebih dahulu.")
            return
        try:
            import cv2
            import base64
            from client_tk.app.services.camera_capture import CameraCaptureService
            cam = CameraCaptureService()
            cam_idx = int(self.preset_camera_index_var.get() or 0)
            cam.start(cam_idx)
            import time
            time.sleep(0.5)
            frame = cam.get_latest_frame()
            cam.stop()
            if frame is None:
                messagebox.showwarning("Reference", "Tidak ada frame dari kamera.")
                return
            # Apply camera rotation to match runtime behavior
            _rot = float(_float_or_default(self.preset_camera_rotation_var.get(), 0))
            if _rot != 0.0:
                h, w = frame.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, -_rot, 1.0)
                cos_a = abs(M[0, 0])
                sin_a = abs(M[0, 1])
                new_w = int(h * sin_a + w * cos_a)
                new_h = int(h * cos_a + w * sin_a)
                M[0, 2] += (new_w - w) / 2
                M[1, 2] += (new_h - h) / 2
                frame = cv2.warpAffine(frame, M, (new_w, new_h),
                                       borderMode=cv2.BORDER_REPLICATE)
            roi = {
                "x": int(float(self.part_ready_roi_x_var.get() or 0.2) * frame.shape[1]),
                "y": int(float(self.part_ready_roi_y_var.get() or 0.2) * frame.shape[0]),
                "w": int(float(self.part_ready_roi_w_var.get() or 0.25) * frame.shape[1]),
                "h": int(float(self.part_ready_roi_h_var.get() or 0.25) * frame.shape[0]),
            }
            _pr_roi = self.preset_roi_picker.get_roi("part_ready")
            roi["rotation"] = float(_pr_roi.get("rotation", 0.0))
            _, buf = cv2.imencode(".png", frame)
            frame_b64 = base64.b64encode(buf).decode("ascii")
            result = self.api.capture_part_ready_ref(self.current_template_id, frame_b64, roi)
            if result.get("saved"):
                self.gap_ref_status_label.configure(
                    text="Referensi: edge map ✓", foreground="green")
                preview_b64 = result.get("preview_b64")
                if preview_b64:
                    self._show_edge_map_preview(preview_b64)
                else:
                    messagebox.showinfo("Reference", "Edge map referensi berhasil disimpan.")
            else:
                messagebox.showerror("Reference", result.get("error", "Gagal menyimpan referensi."))
        except Exception as exc:
            messagebox.showerror("Reference", f"Capture failed: {exc}")

    def _show_edge_map_preview(self, preview_b64: str) -> None:
        """Dialog preview edge map yang baru disimpan sebagai master."""
        try:
            import base64 as _b64
            import numpy as np
            import cv2
            from PIL import Image, ImageTk

            raw = _b64.b64decode(preview_b64)
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                messagebox.showinfo("Reference", "Edge map disimpan (preview tidak tersedia).")
                return
            max_w, max_h = 640, 400
            h, w = img.shape[:2]
            scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            pil_img = Image.fromarray(img)
            photo = ImageTk.PhotoImage(pil_img)
            dlg = tk.Toplevel(self)
            dlg.title("Preview Edge Map Referensi")
            dlg.resizable(False, False)
            tk.Label(dlg, image=photo).pack(padx=10, pady=10)
            tk.Label(
                dlg,
                text="Ini yang disimpan sebagai master.\n"
                     "Pastikan tepi part & clamp terlihat jelas.",
                justify="center",
            ).pack(pady=(0, 6))
            tk.Button(dlg, text="OK", width=10, command=dlg.destroy).pack(pady=(0, 10))
            dlg._photo = photo
            dlg.grab_set()
            dlg.wait_window()
        except Exception as exc:
            messagebox.showinfo("Reference", f"Edge map disimpan. Preview error: {exc}")

    def _upload_part_ready_ref(self) -> None:
        """Upload reference patch image from file."""
        if not self.current_template_id:
            messagebox.showwarning("Reference", "Pilih template terlebih dahulu.")
            return
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="Pilih gambar referensi gap",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            result = self.api.upload_part_ready_ref(self.current_template_id, file_path)
            if result.get("saved"):
                self.gap_ref_status_label.configure(
                    text="Referensi: edge map ✓", foreground="green")
                messagebox.showinfo("Reference", "Referensi edge map berhasil diupload.")
            else:
                messagebox.showerror("Reference", result.get("error", "Gagal upload referensi."))
        except Exception as exc:
            messagebox.showerror("Reference", f"Upload failed: {exc}")


    def _preset_payload(self) -> dict:
        name = self.preset_name_var.get().strip()
        expected_class = self.preset_expected_class_var.get().strip()
        model_path = self.preset_model_path_var.get().strip()
        if not name:
            raise ValueError("Preset name is required.")
        mode = self.preset_validator_mode_var.get()
        # Model path is not required for defect mode
        if mode != "defect" and not model_path:
            raise ValueError("Model is required.")
        if mode == "component_count" or mode == "defect":
            expected_class = expected_class or ""
        else:
            if not expected_class:
                raise ValueError("Expected class is required.")
        max_tilt = None
        if self.preset_max_tilt_var.get().strip():
            max_tilt = _float_or_default(self.preset_max_tilt_var.get(), 5.0)

        # Build criteria dict based on mode
        _criteria = {}
        if mode == "sticker":
            _criteria = {
                "expected_class": expected_class,
                "enabled": True,
                "min_roi_confidence": 0.0,
                "min_class_confidence": None,
                "max_offset_x": 80,
                "max_offset_y": 80,
                "tilt_gate_enabled": bool(self.preset_tilt_gate_var.get()),
                "max_tilt_degrees": max_tilt,
                "expected_tilt_degrees": 0.0,
            }
        elif mode == "component_count":
            _criteria = {
                "component_rois": self.preset_component_rois,
                "part_ready_source": self.preset_part_ready_source_var.get(),
            }
        elif mode == "defect":
            # Single model selector writes to criteria.default_model_path for defect mode
            _defect_model = self.preset_model_path_var.get().strip() or None
            _criteria = {
                "rois": getattr(self, "preset_defect_rois", []),
                "default_model_path": _defect_model,
                "inference_mode": getattr(self, "_defect_infer_mode_var", tk.StringVar(value="whole_part")).get(),
                "aggregation": "p99",
            }

        return {
            "id": self.current_template_id,
            "version_id": self.current_template_version_id,
            "version_number": 1,
            "name": name,
            "description": self.preset_description_var.get().strip(),
            "is_active": True,
            "mode": mode,
            "criteria": _criteria,
            "camera": {
                "camera_index": int(_float_or_default(self.preset_camera_index_var.get(), 0)),
                "rotation_degrees": float(_float_or_default(self.preset_camera_rotation_var.get(), 0)),
                "width": None, "height": None, "fps": None,
            },
            "part_ready_roi": {
                "x": _float_or_default(self.part_ready_roi_x_var.get(), 0.2),
                "y": _float_or_default(self.part_ready_roi_y_var.get(), 0.2),
                "w": _float_or_default(self.part_ready_roi_w_var.get(), 0.25),
                "h": _float_or_default(self.part_ready_roi_h_var.get(), 0.25),
                "rotation": self.preset_roi_picker.get_roi("part_ready").get("rotation", 0.0),
            },
            "sticker_roi": {
                "x": _float_or_default(self.sticker_roi_x_var.get(), 0.2),
                "y": _float_or_default(self.sticker_roi_y_var.get(), 0.2),
                "w": _float_or_default(self.sticker_roi_w_var.get(), 0.6),
                "h": _float_or_default(self.sticker_roi_h_var.get(), 0.6),
                "rotation": self.preset_roi_picker.get_roi("sticker").get("rotation", 0.0),
            },
            "vision": {
                # Defect mode writes placeholder to vision.model_path; actual model->criteria.default_model_path
                "model_path": model_path if mode != "defect" else "models/dummy.pt",
                "model_meta_path": self.preset_model_meta_path_var.get().strip() or None,
                "runtime": self.preset_runtime_var.get().strip() or "auto",
                "conf_threshold": _float_or_default(self.preset_conf_threshold_var.get(), 0.15),
                "stream_fps": 10.0,
                "inference_fps": 4.0,
                "imgsz": 640,
                "classes": [c.strip() for c in self.preset_model_classes_var.get().split(",") if c.strip()] or [expected_class],
                "enable_ergonomic_check": False,
                "ergonomic_pose_model_path": None,
                "ergonomic_min_keypoint_conf": 0.35,
                "text_anchor_class": "text_anchor",
                "center_dot_class": "center_dot",
                "anchor_crop_padding_ratio": 0.08,
                "anchor_crop_scale": 2.0,
            },
            "part_ready": {
                "enabled": True,
                "method": self.preset_part_ready_method_var.get(),
                "gap_match_threshold": _float_or_default(self.preset_gap_threshold_var.get(), 0.85),
                "gap_ref_path": self._get_existing_gap_ref_path(),
                "stable_ms": 500,
                "release_ms": 300,
                "mean_max": _float_or_default(self.preset_mean_max_var.get(), 105.0),
                "std_max": _float_or_default(self.preset_std_max_var.get(), 35.0),
                "min_match_ratio": _float_or_default(self.preset_min_match_ratio_var.get(), 0.5),
            },
            "sticker": {
                "part_name": expected_class,
                "expected_class": expected_class,
                "enabled": True,
                "validator_mode": validator_mode_for_payload(mode),
                "min_roi_confidence": 0.0,
                "min_class_confidence": None,
                "max_offset_x": 80,
                "max_offset_y": 80,
                "expected_center_x": None,
                "expected_center_y": None,
                "expected_tilt_degrees": 0.0,
                "max_tilt_degrees": max_tilt,
                "tilt_gate_enabled": bool(self.preset_tilt_gate_var.get()),
                "commit_stable_frames": 1,
                "part_ready_settle_ms": None,
                "white_hsv_lower": [0, 0, 160],
                "white_hsv_upper": [180, 70, 255],
                "min_text_contour_area_ratio": 0.002,
            },
            "persistence": {"write_to_db": True},
            "metadata": {"preset_ui": "admin_simple"},
        }

    def deactivate_selected_preset(self) -> None:
        selected_kind, selected_id = self._selected_preset_row()
        if selected_kind is None or selected_id is None:
            return
        if selected_kind == "deployment":
            deployment_id = selected_id
            if not self._confirm_action("Deactivate Preset", f"Deactivate preset deployment #{deployment_id}?"):
                return
            try:
                self.api.deactivate_deployment(deployment_id)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Preset", str(exc))
                return
            self.reset_preset_wizard()
            self.refresh_presets()
            self._set_status(f"Preset deployment #{deployment_id} deactivated.")
            return

        if selected_kind == "template":
            template_id = selected_id
            if not self._confirm_action("Delete Template", f"Delete template #{template_id}?"):
                return
            try:
                self.api.delete_template(template_id)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Preset", str(exc))
                return
            self.reset_preset_wizard()
            self.refresh_presets()
            self._set_status(f"Template #{template_id} deleted.")
            return

        return

    def export_runtime_template(self) -> None:
        version_id = self.current_template_version_id
        if not version_id:
            messagebox.showwarning("Template", "Save or select a deployed template first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            initialfile="template.json",
            title="Export Runtime Template",
        )
        if not path:
            return
        try:
            payload = self.api.get_runtime_template(int(version_id))
            with open(path, "w", encoding="utf-8") as file_handle:
                import json

                json.dump(payload, file_handle, ensure_ascii=True, indent=2)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Template", str(exc))
            return
        self._set_status(f"template.json exported to {path}.")

    def _on_dataset_selected(self, _event=None) -> None:
        selection = self.datasets_table.selection() if hasattr(self, "datasets_table") else ()
        if not selection:
            return
        dataset_id = str(selection[0])
        if dataset_id.startswith("__"):
            return
        self.training_dataset_var.set(dataset_id)
        self.augment_dataset_var.set(dataset_id)

    def _selected_base_model(self) -> dict | None:
        selected = self.training_base_model_var.get().strip()
        if selected in self._base_model_lookup:
            return self._base_model_lookup[selected]
        selected_key = selected.lower()
        return next((item for item in self._base_models_cache if str(item.get("id") or "").lower() == selected_key), None)

    def start_simple_training(self) -> None:
        dataset_id = self.training_dataset_var.get().strip()
        base_model = self._selected_base_model()
        if not dataset_id:
            messagebox.showwarning("Training", "Select or enter a dataset first.")
            return
        if base_model is None:
            messagebox.showwarning("Training", "Select a base model first.")
            return
        try:
            payload = {
                "dataset_id": dataset_id,
                "base_model": base_model.get("id"),
                "base_model_family": base_model.get("family"),
                "base_model_variant": base_model.get("variant"),
                "base_model_display_name": base_model.get("display_name"),
                "base_model_weights_name": base_model.get("weights_name"),
                "device_mode": self.training_device_var.get().strip() or "auto",
                "epochs": int(float(self.training_epochs_var.get().strip() or "200")),
            }
            created = self.api.create_training_job(payload)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Training", str(exc))
            return
        self.refresh_models_training()
        self._set_status(f"Training job {created.get('id', '')} queued.")

    def create_basic_augment_job(self) -> None:
        dataset_id = self.training_dataset_var.get().strip() or self.augment_dataset_var.get().strip()
        if not dataset_id:
            messagebox.showwarning("Augment", "Select or enter a dataset first.")
            return
        try:
            created = self.api.create_augment_job(
                {"dataset_id": dataset_id, "transforms": ["brightness", "blur"], "multiplier": 2}
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Augment", str(exc))
            return
        self.refresh_models_training()
        self._set_status(f"Augment job {created.get('id', '')} created.")

    def use_selected_model_in_template(self) -> None:
        model_id = self._selected_treeview_id(self.models_table)
        if model_id is None:
            messagebox.showwarning("Models", "Select a model first.")
            return
        item = next((entry for entry in self._models_cache if int(entry.get("id") or 0) == model_id), None)
        if not item:
            return
        self.preset_model_path_var.set(str(item.get("path") or ""))
        self.preset_model_meta_path_var.set(str(item.get("meta_path") or ""))
        self.preset_runtime_var.set(str(item.get("runtime") or "auto"))
        self._select_model_label_for_path(str(item.get("path") or ""))
        self._notebook.set("Templates")
        self._set_status("Selected model copied into the template wizard.")

    def import_model_archive(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Model Archive", "*.zip"), ("All Files", "*.*")],
            title="Import Model Archive",
        )
        if not path:
            return
        try:
            self.api.import_model_archive(path, target_lifecycle="published", skip_validation=False, force_rename=False)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Models", str(exc))
            return
        self.refresh_model_options()
        self.refresh_models_training()
        self._set_status("Model archive imported.")

    # ------------------------------------------------------------------
    # Monitor behavior
    def _monitor_filters(self) -> dict[str, object]:
        params: dict[str, object] = {"limit": 100}
        return params

    def export_monitor_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialfile="inspections.csv",
            title="Export Inspection Results",
        )
        if not path:
            return
        try:
            csv_text = self.api.export_inspections_csv(self._monitor_filters())
            with open(path, "w", encoding="utf-8", newline="") as file_handle:
                file_handle.write(csv_text)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export CSV", str(exc))
            return
        self._set_status(f"CSV export saved to {path}.")

    def retry_visible_failed_pushes(self) -> None:
        retry_ids = [
            int(item.get("id") or 0)
            for item in self._results_cache
            if str(item.get("push_status") or "").lower() in {"failed", "pending"} and int(item.get("id") or 0) > 0
        ]
        if not retry_ids:
            messagebox.showinfo("Monitor", "No failed or pending pushes are visible.")
            return
        try:
            result = self.api.retry_failed_inspection_pushes(result_ids=retry_ids, limit=len(retry_ids))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Monitor", str(exc))
            return
        self.refresh_monitor()
        self._set_status(
            f"Retry attempted={result.get('attempted', len(retry_ids))}, succeeded={result.get('succeeded', 0)}."
        )

    def open_monitor_result(self, _event=None) -> None:
        result_id = self._selected_treeview_id(self.results_table)
        if result_id is None:
            return
        item = next((entry for entry in self._results_cache if int(entry.get("id") or 0) == result_id), None)
        if not item:
            return
        decision = str(item.get("decision") or item.get("decision_code") or "").strip().upper()
        self.monitor_summary.set_values(
            {
                "decision": decision,
                "reason": item.get("reject_reason_code") or ("OK" if decision == "ACCEPT" else "-"),
                "push_status": item.get("push_status"),
            }
        )

    # ------------------------------------------------------------------
    # Shared helpers
    def _entry(self, master, row: int, column: int, label: str, variable: tk.StringVar, *, columnspan: int = 1, columns: int = 4) -> ttk.Entry:
        ttk.Label(master, text=label).grid(row=row, column=column, sticky="w", padx=(12, 8), pady=5)
        entry = ttk.Entry(master, textvariable=variable)
        entry.grid(row=row, column=column + 1, columnspan=columnspan, sticky="ew", padx=(0, 12), pady=5)
        if columns:
            for index in range(columns):
                master.columnconfigure(index, weight=1 if index % 2 else 0)
        return entry

    def _grid_entry(self, master, row: int, column: int, label: str, widget) -> None:
        ttk.Label(master, text=label).grid(row=row, column=column, sticky="w", padx=(0, 4), pady=2)
        widget.grid(row=row, column=column + 1, sticky="ew", padx=(0, 8), pady=2)

    def _roi_entries(self, master, row: int, x_var: tk.StringVar, y_var: tk.StringVar, w_var: tk.StringVar, h_var: tk.StringVar) -> None:
        labels = (("x", x_var), ("y", y_var), ("w", w_var), ("h", h_var))
        for offset, (label, variable) in enumerate(labels):
            column = offset * 2
            ttk.Label(master, text=label).grid(row=row, column=column, sticky="w", padx=(12 if offset == 0 else 8, 4), pady=5)
            ttk.Entry(master, textvariable=variable, width=8).grid(row=row, column=column + 1, sticky="ew", padx=(0, 8), pady=5)

    def _build_table(self, master, columns: list[tuple[str, str, int, str]], *, row: int | None = None, height: int = 14) -> ttk.Treeview:
        shell = ttk.Frame(master)
        if row is None:
            shell.pack(fill="both", expand=True)
        else:
            shell.grid(row=row, column=0, sticky="nsew")
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)

        column_names = [item[0] for item in columns]
        tree = ttk.Treeview(shell, columns=column_names, show="headings", height=height, selectmode="browse")
        for name, heading, width, anchor in columns:
            tree.heading(name, text=heading)
            tree.column(name, width=width, minwidth=60, stretch=True, anchor=anchor)
        y_scroll = AutoHideScrollbar(shell, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=y_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        return tree

    def _build_action_row(self, master, buttons: list[tuple]) -> ctk.CTkFrame:
        row = ctk.CTkFrame(master, fg_color="transparent", corner_radius=0)
        left = ctk.CTkFrame(row, fg_color="transparent", corner_radius=0)
        right = ctk.CTkFrame(row, fg_color="transparent", corner_radius=0)
        left.pack(side="left")
        right.pack(side="right")
        for label, command, tone, side in buttons:
            target = left if side == "left" else right
            color = ACCENT if tone == "primary" else PANEL_ALT_BG
            hover = ACCENT_HOVER if tone == "primary" else BORDER
            ctk.CTkButton(
                target,
                text=label,
                command=command,
                fg_color=color,
                hover_color=hover,
                text_color=TEXT_ON_ACCENT if tone == "primary" else TEXT_PRIMARY,
                height=28,
                corner_radius=6,
            ).pack(side="left", padx=(0, 6))
        return row

    def _layout_overview_cards(self, *, compact: bool) -> None:
        try:
            if not self.winfo_exists() or not self.overview_cards_frame.winfo_exists():
                return
            slaves = self.overview_cards_frame.grid_slaves()
        except tk.TclError:
            return
        for widget in slaves:
            widget.grid_forget()
        columns = 2 if compact else 4
        for column in range(columns):
            self.overview_cards_frame.columnconfigure(column, weight=1)
        for index, key in enumerate(("presets", "operators", "accept", "reject")):
            self.admin_cards[key].grid(row=index // columns, column=index % columns, sticky="ew", padx=4, pady=4)

    def _layout_monitor_cards(self, *, compact: bool) -> None:
        try:
            if not self.winfo_exists() or not self.monitor_cards_frame.winfo_exists():
                return
            slaves = self.monitor_cards_frame.grid_slaves()
        except tk.TclError:
            return
        for widget in slaves:
            widget.grid_forget()
        columns = 2 if compact else 6
        for column in range(columns):
            self.monitor_cards_frame.columnconfigure(column, weight=1)
        for index, key in enumerate(("total", "accept", "reject", "reject_rate", "pending", "failed")):
            self.monitor_cards[key].grid(row=index // columns, column=index % columns, sticky="ew", padx=4, pady=4)

    def _clear_tree(self, tree: ttk.Treeview) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def _selected_treeview_id(self, tree: ttk.Treeview) -> int | None:
        selection = tree.selection()
        if not selection:
            focus = tree.focus()
            selection = (focus,) if focus else ()
        if not selection:
            return None
        raw = str(selection[0])
        if raw.startswith("__"):
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _selected_preset_row(self) -> tuple[str | None, int | None]:
        selection = self.preset_table.selection()
        if not selection:
            focus = self.preset_table.focus()
            selection = (focus,) if focus else ()
        if not selection:
            return None, None
        raw = str(selection[0])
        if raw.startswith("__"):
            return None, None
        if raw.startswith("dep:"):
            try:
                return "deployment", int(raw.split(":", 1)[1])
            except ValueError:
                return None, None
        if raw.startswith("tpl:"):
            try:
                return "template", int(raw.split(":", 1)[1])
            except ValueError:
                return None, None
        try:
            return "deployment", int(raw)
        except ValueError:
            return None, None

    def _confirm_action(self, title: str, message: str) -> bool:
        return bool(messagebox.askyesno(title, message))

    def _record_refresh(self, key: str) -> None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self._last_refresh[key] = timestamp
        self.refresh_time_var.set(f"Last {key}: {timestamp}")

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _update_overview_cards(self) -> None:
        active_presets = sum(1 for item in self._deployments_cache if bool(item.get("is_active", True)))
        total_users = len(self._users_cache)
        self.admin_cards["presets"].set_value(active_presets, "active")
        self.admin_cards["operators"].set_value(total_users, "total users")

    def _on_resize(self, _event=None) -> None:
        self.after_idle(self._apply_responsive_layout)

    def _apply_responsive_layout(self) -> None:
        try:
            if not self.winfo_exists():
                return
            width = max(self.winfo_width(), self.winfo_toplevel().winfo_width())
            height = self.winfo_height()
        except tk.TclError:
            return
        compact = width < RESPONSIVE_BREAKPOINT
        if compact != self._layout_compact:
            self._layout_compact = compact
            self._layout_overview_cards(compact=compact)
            self._layout_monitor_cards(compact=compact)

            for left, right in (
                (self.presets_left, self.presets_right),
                (self.operators_left, self.operators_right),
            ):
                left.grid_forget()
                right.grid_forget()
                if compact:
                    left.grid(row=0, column=0, columnspan=2, sticky="nsew")
                    right.grid(row=1, column=0, columnspan=2, sticky="nsew")
                else:
                    left.grid(row=0, column=0, sticky="nsew")
                    right.grid(row=0, column=1, sticky="nsew")

        self._overview_cards_visible = height >= 760
        try:
            if self._overview_cards_visible:
                self.overview_cards_frame.grid()
            else:
                self.overview_cards_frame.grid_remove()
        except tk.TclError:
            return

    # ==================================================================
    # Data Tab - Dataset management, upload, versioning
    # ==================================================================
    def _build_data_tab(self) -> None:
        self.data_tab.columnconfigure(0, weight=1)
        self.data_tab.rowconfigure(0, weight=1)
        self.data_tab.rowconfigure(1, weight=0)

        body = self._make_scrollable_body(self.data_tab, "Data")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=2)
        body.rowconfigure(1, weight=1)

        # Top section: dataset list + upload + versions (existing layout)
        top_shell = ttk.Frame(body)
        top_shell.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
        top_shell.columnconfigure(0, weight=3)
        top_shell.columnconfigure(1, weight=2)
        top_shell.rowconfigure(0, weight=1)

        left_panel = ttk.LabelFrame(top_shell, text="Datasets", padding=8)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(0, weight=1)

        # Dataset list
        self._admin_dataset_list = tk.Listbox(left_panel, height=8)
        self._admin_dataset_list.grid(row=0, column=0, sticky="nsew")
        self._admin_dataset_list.bind("<<ListboxSelect>>", self._on_admin_dataset_selected)

        ds_form = ttk.Frame(left_panel)
        ds_form.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._admin_ds_name_var = tk.StringVar()
        self._admin_ds_desc_var = tk.StringVar()
        ttk.Label(ds_form, text="Name").grid(row=0, column=0, sticky="w")
        ttk.Entry(ds_form, textvariable=self._admin_ds_name_var).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Label(ds_form, text="Desc").grid(row=1, column=0, sticky="w")
        ttk.Entry(ds_form, textvariable=self._admin_ds_desc_var).grid(row=1, column=1, sticky="ew", padx=(4, 0))
        ds_form.columnconfigure(1, weight=1)

        ds_btn = ttk.Frame(left_panel)
        ds_btn.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(ds_btn, text="Create", command=self._admin_create_dataset).pack(side="left")
        ttk.Button(ds_btn, text="Delete", command=self._admin_delete_dataset).pack(side="left", padx=4)
        ttk.Button(ds_btn, text="Refresh", command=self._admin_refresh_datasets).pack(side="left")

        # Upload + Image list + Class management in a combined frame below dataset list
        bottom_left = ttk.Frame(left_panel)
        bottom_left.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        bottom_left.columnconfigure(0, weight=1)
        left_panel.rowconfigure(3, weight=1)

        upload_frame = ttk.LabelFrame(bottom_left, text="Upload Images", padding=6)
        upload_frame.pack(fill="x")
        self._admin_upload_path_var = tk.StringVar(value="No files selected")
        ttk.Button(upload_frame, text="Choose Files", command=self._admin_choose_upload_files).pack(anchor="w")
        ttk.Label(upload_frame, textvariable=self._admin_upload_path_var, wraplength=300).pack(anchor="w", pady=(2, 0))
        ttk.Button(upload_frame, text="Upload", command=self._admin_upload_files).pack(anchor="w", pady=(4, 0))
        self._admin_upload_paths: list[str] = []

        # Image listbox + coverage + class management
        img_frame = ttk.LabelFrame(bottom_left, text="Images", padding=6)
        img_frame.pack(fill="both", expand=True, pady=(6, 0))
        img_frame.columnconfigure(0, weight=1)
        img_frame.rowconfigure(1, weight=1)

        img_header = ttk.Frame(img_frame)
        img_header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(img_header, text="Images", font=("Segoe UI", 9, "bold")).pack(side="left")
        self._admin_annot_coverage_var = tk.StringVar(value="Coverage: -")
        ttk.Label(img_header, textvariable=self._admin_annot_coverage_var, foreground="#64748b").pack(side="right")

        self._admin_annot_image_listbox = tk.Listbox(img_frame, height=10, exportselection=False, font=("Segoe UI", 9))
        self._admin_annot_image_listbox.grid(row=1, column=0, sticky="nsew")
        self._admin_annot_image_listbox.bind("<<ListboxSelect>>", self._on_admin_annot_image_list_selected)
        img_scroll = ttk.Scrollbar(img_frame, orient="vertical", command=self._admin_annot_image_listbox.yview)
        img_scroll.grid(row=1, column=1, sticky="ns")
        self._admin_annot_image_listbox.configure(yscrollcommand=img_scroll.set)

        # Class management
        class_frame = ttk.Frame(img_frame)
        class_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self._admin_annot_new_class_var = tk.StringVar()
        ttk.Entry(class_frame, textvariable=self._admin_annot_new_class_var, width=12).pack(side="left")
        ttk.Button(class_frame, text="Add", command=self._admin_add_annotation_class, width=5).pack(side="left", padx=(4, 0))
        ttk.Button(class_frame, text="Del", command=self._admin_remove_annotation_class, width=5).pack(side="left", padx=(4, 0))

        # Right panel: Dataset versions
        right_panel = ttk.LabelFrame(top_shell, text="Dataset Versions", padding=8)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)

        self._admin_version_list = tk.Listbox(right_panel, height=8)
        self._admin_version_list.grid(row=0, column=0, sticky="nsew")

        ver_form = ttk.Frame(right_panel)
        ver_form.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._admin_ver_name_var = tk.StringVar(value="Snapshot v1")
        self._admin_ver_desc_var = tk.StringVar(value="YOLO export snapshot")
        self._admin_ver_status_var = tk.StringVar(value="ready")
        self._admin_ver_train_var = tk.StringVar(value="0.7")
        self._admin_ver_valid_var = tk.StringVar(value="0.2")
        self._admin_ver_test_var = tk.StringVar(value="0.1")
        ttk.Label(ver_form, text="Name").grid(row=0, column=0, sticky="w")
        ttk.Entry(ver_form, textvariable=self._admin_ver_name_var).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Label(ver_form, text="Desc").grid(row=1, column=0, sticky="w")
        ttk.Entry(ver_form, textvariable=self._admin_ver_desc_var).grid(row=1, column=1, sticky="ew", padx=(4, 0))
        ttk.Label(ver_form, text="Status").grid(row=2, column=0, sticky="w")
        ttk.Combobox(ver_form, textvariable=self._admin_ver_status_var, values=["draft", "ready", "archived"], state="readonly").grid(row=2, column=1, sticky="ew", padx=(4, 0))
        ttk.Label(ver_form, text="Train/Valid/Test").grid(row=3, column=0, sticky="w")
        ratios = ttk.Frame(ver_form)
        ratios.grid(row=3, column=1, sticky="ew", padx=(4, 0))
        ttk.Entry(ratios, textvariable=self._admin_ver_train_var, width=5).pack(side="left")
        ttk.Entry(ratios, textvariable=self._admin_ver_valid_var, width=5).pack(side="left", padx=2)
        ttk.Entry(ratios, textvariable=self._admin_ver_test_var, width=5).pack(side="left")
        ver_form.columnconfigure(1, weight=1)

        ver_btn = ttk.Frame(right_panel)
        ver_btn.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(ver_btn, text="Create Version", command=self._admin_create_version).pack(side="left")
        ttk.Button(ver_btn, text="Export", command=self._admin_rebuild_export).pack(side="left", padx=4)
        ttk.Button(ver_btn, text="Refresh", command=self._admin_refresh_versions).pack(side="left")

        # Bottom section: Annotation workflow (full width)
        annot_shell = ttk.LabelFrame(body, text="Annotation Workflow", padding=6)
        annot_shell.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        annot_shell.columnconfigure(0, weight=1)
        annot_shell.rowconfigure(1, weight=1)

        # Toolbar
        toolbar = ttk.Frame(annot_shell)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        toolbar.columnconfigure(1, weight=1)
        toolbar.columnconfigure(3, weight=1)
        toolbar.columnconfigure(5, weight=1)
        toolbar.columnconfigure(7, weight=1)
        self._admin_annot_dataset_var = tk.StringVar(value="")
        self._admin_annot_image_var = tk.StringVar(value="")
        self._admin_annot_class_var = tk.StringVar(value="object")
        self._admin_annot_dataset_combo = ttk.Combobox(toolbar, textvariable=self._admin_annot_dataset_var, values=[], state="readonly")
        self._admin_annot_dataset_combo.bind("<<ComboboxSelected>>", self._on_admin_annot_dataset_selected)
        self._admin_annot_image_entry = ttk.Entry(toolbar, textvariable=self._admin_annot_image_var, state="readonly")
        self._admin_annot_shape = ttk.Combobox(toolbar, values=["bbox", "polygon"], state="readonly", width=8)
        self._admin_annot_shape.set("bbox")
        self._admin_annot_class_combo = ttk.Combobox(toolbar, textvariable=self._admin_annot_class_var, values=["object"], state="normal")
        self._grid_entry(toolbar, 0, 0, "Dataset", self._admin_annot_dataset_combo)
        self._grid_entry(toolbar, 0, 2, "Image", self._admin_annot_image_entry)
        self._grid_entry(toolbar, 0, 4, "Type", self._admin_annot_shape)
        self._grid_entry(toolbar, 0, 6, "Class", self._admin_annot_class_combo)

        # Action buttons
        action_bar = ttk.Frame(annot_shell)
        action_bar.grid(row=0, column=1, sticky="e", pady=(0, 4))
        ttk.Button(action_bar, text="Save", command=self._admin_save_current_annotation_interactive).pack(side="left")
        ttk.Button(action_bar, text="Apply Class", command=self._admin_apply_class_to_selected_annotation).pack(side="left", padx=(4, 0))
        ttk.Button(action_bar, text="Delete Label", command=self._admin_delete_selected_annotation).pack(side="left", padx=(4, 0))

        # Canvas + Nav row
        canvas_row = ttk.Frame(annot_shell)
        canvas_row.grid(row=1, column=0, columnspan=2, sticky="nsew")
        canvas_row.columnconfigure(0, weight=1)
        canvas_row.rowconfigure(0, weight=1)

        self._admin_annotation_canvas = AnnotationCanvas(canvas_row, title="Image Annotation")
        self._admin_annotation_canvas.grid(row=0, column=0, sticky="nsew")

        # Nav bar
        nav = ttk.Frame(annot_shell)
        nav.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(nav, text="\u2190 Prev", command=self.admin_previous_annotation_image).pack(side="left")
        self._admin_annotation_status_var = tk.StringVar(value="Select a dataset to start annotating.")
        ttk.Label(nav, textvariable=self._admin_annotation_status_var, foreground="#64748b").pack(side="left", padx=12)
        ttk.Button(nav, text="Next \u2192", command=self.admin_next_annotation_image).pack(side="right")

        # Bindings
        self._admin_annot_shape.bind("<<ComboboxSelected>>", lambda _e: self._admin_sync_annotation_mode())
        self._admin_annot_class_combo.bind("<<ComboboxSelected>>", self._on_admin_annot_class_input)
        self._admin_annot_class_combo.bind("<Return>", self._on_admin_annot_class_input)
        self._admin_annot_class_combo.bind("<FocusOut>", self._on_admin_annot_class_input)
        self._admin_annotation_canvas.set_mode("bbox")
        self._admin_annotation_canvas.on_labels_changed = self._on_admin_annot_labels_changed
        self._admin_annotation_canvas.on_selection_changed = self._on_admin_annot_selection_changed
        self._admin_annotation_canvas._canvas.bind("<KeyPress-s>", self._on_admin_annot_shortcut_save)
        self._admin_annotation_canvas._canvas.bind("<KeyPress-Left>", self._on_admin_annot_shortcut_prev)
        self._admin_annotation_canvas._canvas.bind("<KeyPress-Right>", self._on_admin_annot_shortcut_next)
        self._admin_annotation_canvas._canvas.bind("<KeyPress-b>", self._on_admin_annot_shortcut_bbox)
        self._admin_annotation_canvas._canvas.bind("<KeyPress-p>", self._on_admin_annot_shortcut_polygon)
        self._admin_annotation_canvas._canvas.bind("<KeyPress-Delete>", self._on_admin_annot_shortcut_delete)
        self._admin_annotation_canvas._canvas.bind("<KeyPress-BackSpace>", self._on_admin_annot_shortcut_delete)

    def _admin_refresh_datasets(self) -> None:
        def _load():
            return self.api.list_datasets()
        def _done(result, error):
            if error:
                messagebox.showerror("Dataset", str(error))
                return
            if not isinstance(result, list):
                return
            self._datasets_cache = result
            self._admin_dataset_list.delete(0, "end")
            self._dataset_display_to_id = {}
            self._dataset_id_to_display = {}
            for item in result:
                ds_id = str(item.get("id") or "")
                name = str(item.get("name") or "")
                images = int(item.get("image_count") or 0)
                ann = int(item.get("annotated_image_count") or 0)
                display = f"{name} | {ds_id} | {images}img {ann}ann"
                self._admin_dataset_list.insert("end", display)
                self._dataset_display_to_id[display] = ds_id
                self._dataset_id_to_display[ds_id] = display
            self._sync_admin_training_datasets()
            self._sync_admin_annotation_dataset_combo()
        run_async(self, _load, callback=_done)

    def _on_admin_dataset_selected(self, _event=None) -> None:
        dataset_id = self._admin_resolve_annotation_dataset_id()
        if dataset_id:
            display = self._dataset_id_to_display.get(dataset_id, "")
            if display:
                self._admin_annot_dataset_combo.set(display)
        self._admin_refresh_versions()
        self.admin_refresh_annotation_images()

    def _on_admin_annot_dataset_selected(self, _event=None) -> None:
        display = self._admin_annot_dataset_var.get().strip()
        dataset_id = self._dataset_display_to_id.get(display, "")
        if not dataset_id:
            return
        self._annotation_dataset_id = dataset_id
        self._admin_dataset_list.selection_clear(0, "end")
        for index in range(self._admin_dataset_list.size()):
            if self._admin_dataset_list.get(index) == display:
                self._admin_dataset_list.selection_set(index)
                self._admin_dataset_list.see(index)
                break
        self._admin_refresh_versions()
        self.admin_refresh_annotation_images()

    def _admin_create_dataset(self) -> None:
        name = self._admin_ds_name_var.get().strip()
        if not name:
            messagebox.showwarning("Dataset", "Name is required.")
            return
        desc = self._admin_ds_desc_var.get().strip()
        try:
            self.api.create_dataset({"name": name, "description": desc})
        except Exception as exc:
            messagebox.showerror("Dataset", str(exc))
            return
        self._admin_ds_name_var.set("")
        self._admin_ds_desc_var.set("")
        self._admin_refresh_datasets()

    def _admin_delete_dataset(self) -> None:
        sel = self._admin_dataset_list.curselection()
        if not sel:
            messagebox.showwarning("Dataset", "Select a dataset first.")
            return
        display = self._admin_dataset_list.get(sel[0])
        ds_id = self._dataset_display_to_id.get(display, "")
        if not ds_id:
            return
        if not messagebox.askyesno("Dataset", f"Delete dataset {ds_id}?"):
            return
        try:
            self.api.delete_dataset(ds_id)
        except Exception as exc:
            messagebox.showerror("Dataset", str(exc))
            return
        self._admin_refresh_datasets()

    def _admin_choose_upload_files(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if paths:
            self._admin_upload_paths = list(paths)
            self._admin_upload_path_var.set(f"{len(paths)} file(s) selected")

    def _admin_upload_files(self) -> None:
        if not self._admin_upload_paths:
            messagebox.showwarning("Upload", "Choose files first.")
            return
        sel = self._admin_dataset_list.curselection()
        if not sel:
            messagebox.showwarning("Upload", "Select a dataset first.")
            return
        display = self._admin_dataset_list.get(sel[0])
        ds_id = self._dataset_display_to_id.get(display, "")
        if not ds_id:
            return
        try:
            self.api.upload_dataset_files(ds_id, list(self._admin_upload_paths), target="images")
        except Exception as exc:
            messagebox.showerror("Upload", str(exc))
            return
        self._admin_upload_paths = []
        self._admin_upload_path_var.set("No files selected")
        self._admin_refresh_datasets()

    def _admin_refresh_versions(self) -> None:
        sel = self._admin_dataset_list.curselection()
        if not sel:
            return
        display = self._admin_dataset_list.get(sel[0])
        ds_id = self._dataset_display_to_id.get(display, "")
        if not ds_id:
            return
        def _load():
            return self.api.list_dataset_versions(ds_id)
        def _done(result, error):
            if error or not isinstance(result, list):
                return
            self._dataset_version_cache = result
            self._admin_version_list.delete(0, "end")
            self._dataset_version_lookup = {}
            for item in result:
                ver_id = str(item.get("id") or "")
                name = str(item.get("name") or "")
                status = str(item.get("status") or "")
                display_v = f"{name} | {ver_id} | {status}"
                self._admin_version_list.insert("end", display_v)
                self._dataset_version_lookup[display_v] = item
        run_async(self, _load, callback=_done)

    def _admin_create_version(self) -> None:
        sel = self._admin_dataset_list.curselection()
        if not sel:
            messagebox.showwarning("Version", "Select a dataset first.")
            return
        display = self._admin_dataset_list.get(sel[0])
        ds_id = self._dataset_display_to_id.get(display, "")
        if not ds_id:
            return
        try:
            train_f = float(self._admin_ver_train_var.get())
            valid_f = float(self._admin_ver_valid_var.get())
            test_f = float(self._admin_ver_test_var.get())
        except ValueError:
            messagebox.showwarning("Version", "Train/Valid/Test must be numbers.")
            return
        payload = {
            "name": self._admin_ver_name_var.get().strip() or "Snapshot v1",
            "description": self._admin_ver_desc_var.get().strip(),
            "status": self._admin_ver_status_var.get().strip(),
            "split_ratios": {"train": train_f, "valid": valid_f, "test": test_f},
        }
        try:
            self.api.create_dataset_version(ds_id, payload)
        except Exception as exc:
            messagebox.showerror("Version", str(exc))
            return
        self._admin_refresh_versions()

    def _admin_rebuild_export(self) -> None:
        sel = self._admin_version_list.curselection()
        if not sel:
            messagebox.showwarning("Export", "Select a version first.")
            return
        display = self._admin_version_list.get(sel[0])
        ver = self._dataset_version_lookup.get(display)
        if not ver:
            return
        ds_id = str(ver.get("dataset_id") or "")
        ver_id = str(ver.get("id") or "")
        if not ds_id or not ver_id:
            return
        try:
            self.api.export_dataset_version(ds_id, ver_id)
            messagebox.showinfo("Export", "Export started.")
        except Exception as exc:
            messagebox.showerror("Export", str(exc))

    # ==================================================================
    # Training Tab - Augment + Training jobs
    # ==================================================================
    def _build_training_tab(self) -> None:
        self.training_tab.columnconfigure(0, weight=1)
        self.training_tab.rowconfigure(0, weight=1)

        body = self._make_scrollable_body(self.training_tab, "Training")

        shell = ttk.Frame(body)
        shell.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)

        top = ttk.Frame(shell)
        top.grid(row=0, column=0, sticky="nsew")
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=2)

        aug_panel = ttk.LabelFrame(top, text="Augmentation", padding=8)
        aug_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        aug_form = ttk.Frame(aug_panel)
        aug_form.pack(fill="x")
        self._admin_aug_dataset_var = tk.StringVar()
        self._admin_aug_multiplier_var = tk.StringVar(value="2")
        ttk.Label(aug_form, text="Dataset ID").grid(row=0, column=0, sticky="w")
        self._admin_aug_dataset_combo = ttk.Combobox(aug_form, textvariable=self._admin_aug_dataset_var, state="readonly")
        self._admin_aug_dataset_combo.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self._admin_aug_dataset_combo.bind("<<ComboboxSelected>>", lambda _event: self._admin_update_augment_estimator())
        ttk.Label(aug_form, text="Multiplier").grid(row=1, column=0, sticky="w")
        ttk.Combobox(aug_form, textvariable=self._admin_aug_multiplier_var, values=[str(i) for i in range(1, 11)], state="readonly", width=6).grid(row=1, column=1, sticky="w", padx=(4, 0))
        aug_form.columnconfigure(1, weight=1)

        ttk.Label(aug_panel, text="Transforms:").pack(anchor="w", pady=(6, 2))
        self._admin_aug_transform_vars: dict[str, tk.BooleanVar] = {}
        photo_frame = ttk.LabelFrame(aug_panel, text="Photometric", padding=4)
        photo_frame.pack(fill="x", pady=(0, 4))
        geo_frame = ttk.LabelFrame(aug_panel, text="Geometric", padding=4)
        geo_frame.pack(fill="x")
        for t_name, t_info in _TRANSFORM_CATALOG.items():
            parent = photo_frame if t_info.category == "photometric" else geo_frame
            var = tk.BooleanVar(value=t_name in ("brightness", "blur"))
            self._admin_aug_transform_vars[t_name] = var
            ttk.Checkbutton(parent, text=t_name, variable=var).pack(anchor="w")

        aug_btn = ttk.Frame(aug_panel)
        aug_btn.pack(fill="x", pady=(6, 0))
        ttk.Button(aug_btn, text="Create Job", command=self._admin_create_augment_job).pack(side="left")
        ttk.Button(aug_btn, text="Refresh", command=self._admin_refresh_augment_jobs).pack(side="left", padx=4)

        self._admin_augment_jobs_list = tk.Listbox(aug_panel, height=8)
        self._admin_augment_jobs_list.pack(fill="both", expand=True, pady=(6, 0))

        # Augment output estimator
        self._admin_aug_estimator_var = tk.StringVar(value="Select dataset + transforms to see output estimate.")
        ttk.Label(aug_panel, textvariable=self._admin_aug_estimator_var, foreground="#475569", wraplength=280, justify="left").pack(anchor="w", pady=(4, 0))

        # Bind transform changes to update estimator
        self._admin_aug_multiplier_var.trace_add("write", lambda *_: self._admin_update_augment_estimator())
        for var in self._admin_aug_transform_vars.values():
            var.trace_add("write", lambda *_: self._admin_update_augment_estimator())

        train_panel = ttk.LabelFrame(top, text="Training", padding=8)
        train_panel.grid(row=0, column=1, sticky="nsew")

        flow_frame = ttk.Frame(train_panel)
        flow_frame.pack(fill="x", pady=(0, 6))
        self._flow_step_vars = []
        for col_idx, (title, desc) in enumerate([("1.Dataset", "Select"), ("2.Version", "Optional"), ("3.Model", "YOLO"), ("4.Params", "HParams"), ("5.Start", "Go")]):
            flow_frame.columnconfigure(col_idx, weight=1)
            sf = ttk.LabelFrame(flow_frame, text=title, padding=2)
            sf.grid(row=0, column=col_idx, sticky="nsew", padx=(0, 2))
            var = tk.StringVar(value="\u25cb")
            self._flow_step_vars.append(var)
            ttk.Label(sf, textvariable=var, font=("Segoe UI", 12)).pack()
            ttk.Label(sf, text=desc, font=("Segoe UI", 7), foreground="#64748b").pack()

        form = ttk.Frame(train_panel)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        # Training mode indicator
        mode_frame = ttk.Frame(form)
        mode_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Label(mode_frame, text="Mode:", font=("Segoe UI", 8), foreground="#64748b").pack(side="left")
        ttk.Label(mode_frame, textvariable=self._admin_train_mode_var, font=("Segoe UI", 8, "bold"), foreground="#166534").pack(side="left", padx=(4, 0))
        self._admin_train_dataset_var = tk.StringVar()
        self._admin_train_dataset_version_var = tk.StringVar()
        self._admin_train_base_model_var = tk.StringVar()
        self._admin_train_device_var = tk.StringVar(value="auto")
        self._admin_train_epochs_var = tk.StringVar(value="1")
        self._admin_train_imgsz_var = tk.StringVar(value="320")
        self._admin_train_batch_var = tk.StringVar(value="4")
        self._admin_train_patience_var = tk.StringVar(value="5")
        self._admin_train_workers_var = tk.StringVar(value="0")
        self._admin_train_cache_var = tk.BooleanVar(value=False)

        ttk.Label(form, text="Dataset").grid(row=1, column=0, sticky="w")
        self._admin_train_dataset_combo = ttk.Combobox(form, textvariable=self._admin_train_dataset_var, state="readonly")
        self._admin_train_dataset_combo.grid(row=1, column=1, sticky="ew", padx=(4, 0))
        self._admin_train_dataset_combo.bind("<<ComboboxSelected>>", self._on_admin_train_dataset_selected)

        ttk.Label(form, text="Version").grid(row=2, column=0, sticky="w")
        self._admin_train_dataset_version_combo = ttk.Combobox(form, textvariable=self._admin_train_dataset_version_var, state="readonly")
        self._admin_train_dataset_version_combo.grid(row=2, column=1, sticky="ew", padx=(4, 0))
        self._admin_train_dataset_version_combo.bind("<<ComboboxSelected>>", self._on_admin_train_dataset_selected)

        ttk.Label(form, text="Base Model").grid(row=3, column=0, sticky="w")
        self._admin_train_base_model_combo = ttk.Combobox(form, textvariable=self._admin_train_base_model_var, state="readonly")
        self._admin_train_base_model_combo.grid(row=3, column=1, sticky="ew", padx=(4, 0))

        ttk.Label(form, text="Device").grid(row=4, column=0, sticky="w")
        ttk.Combobox(form, textvariable=self._admin_train_device_var, values=["auto", "gpu", "cpu"], state="readonly").grid(row=4, column=1, sticky="ew", padx=(4, 0))

        self._admin_train_readiness_var = tk.StringVar(value="Select dataset to check readiness.")
        ttk.Label(train_panel, textvariable=self._admin_train_readiness_var, foreground="#475569", wraplength=400, justify="left").pack(anchor="w", pady=(2, 0))

        hp_frame = ttk.LabelFrame(train_panel, text="Hyperparameters", padding=6)
        hp_frame.pack(fill="x", pady=(6, 0))
        hp_frame.columnconfigure(1, weight=1)
        hp_frame.columnconfigure(3, weight=1)
        for row_idx, (label, var) in enumerate([("Epochs", self._admin_train_epochs_var), ("Img Size", self._admin_train_imgsz_var), ("Batch", self._admin_train_batch_var), ("Patience", self._admin_train_patience_var), ("Workers", self._admin_train_workers_var)]):
            col = (row_idx % 2) * 2
            ttk.Label(hp_frame, text=label).grid(row=row_idx // 2, column=col, sticky="w")
            ttk.Spinbox(hp_frame, textvariable=var, width=8).grid(row=row_idx // 2, column=col + 1, sticky="w", padx=(4, 8))
        ttk.Checkbutton(hp_frame, text="Cache", variable=self._admin_train_cache_var).grid(row=3, column=0, columnspan=2, sticky="w")

        btn_bar = ttk.Frame(train_panel)
        btn_bar.pack(fill="x", pady=(6, 0))
        ttk.Button(btn_bar, text="Start Training", command=self._admin_create_training_job).pack(side="left")
        ttk.Button(btn_bar, text="Cancel", command=self._admin_cancel_training_job).pack(side="left", padx=4)
        ttk.Button(btn_bar, text="Delete", command=self._admin_delete_training_job).pack(side="left", padx=4)
        ttk.Button(btn_bar, text="Refresh", command=self._admin_refresh_training_jobs).pack(side="left")

        bottom = ttk.LabelFrame(shell, text="Training Jobs", padding=6)
        bottom.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        bottom.columnconfigure(0, weight=1)
        bottom.rowconfigure(0, weight=1)

        self._admin_training_jobs_list = tk.Listbox(bottom)
        self._admin_training_jobs_list.grid(row=0, column=0, sticky="nsew")
        self._admin_training_jobs_list.bind("<<ListboxSelect>>", self._on_admin_training_job_selected)

        detail = ttk.Frame(bottom)
        detail.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._admin_train_status_var = tk.StringVar(value="-")
        self._admin_train_progress_var = tk.StringVar(value="-")
        self._admin_train_message_var = tk.StringVar(value="-")
        ttk.Label(detail, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(detail, textvariable=self._admin_train_status_var).grid(row=0, column=1, sticky="w", padx=(4, 0))
        ttk.Label(detail, text="Progress:").grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Label(detail, textvariable=self._admin_train_progress_var).grid(row=0, column=3, sticky="w", padx=(4, 0))
        ttk.Label(detail, text="Message:").grid(row=1, column=0, sticky="w")
        ttk.Label(detail, textvariable=self._admin_train_message_var, wraplength=500, foreground="#475569").grid(row=1, column=1, columnspan=3, sticky="w", padx=(4, 0))

        self._admin_train_progress_bar = ttk.Progressbar(bottom, orient="horizontal", mode="determinate", maximum=100)
        self._admin_train_progress_bar.grid(row=2, column=0, sticky="ew", pady=(4, 0))

    def _sync_admin_training_datasets(self) -> None:
        values = []
        display_to_id = {}
        id_to_display = {}
        for item in self._datasets_cache:
            ds_id = str(item.get("id") or "")
            name = str(item.get("name") or "")
            images = int(item.get("image_count") or 0)
            ann = int(item.get("annotated_image_count") or 0)
            aug = int(item.get("augmented_count") or 0)
            display = f"{name} | {ds_id} | {images}img {ann}ann +{aug}aug"
            values.append(display)
            display_to_id[display] = ds_id
            id_to_display[ds_id] = display
        self._admin_train_display_to_id = display_to_id
        self._admin_train_id_to_display = id_to_display
        self._admin_train_dataset_combo["values"] = values
        self._admin_aug_dataset_combo["values"] = values
        self._admin_train_version_display_to_id: dict[str, str] = {}

    def _sync_admin_dataset_versions(self, ds_id: str, images: int, ann: int) -> None:
        """Load dataset versions and populate the version dropdown."""
        if images == 0 or ann == 0:
            self._admin_train_readiness_var.set(f"NOT READY - {images} images, {ann} annotations. Complete upload + annotation first.")
            self._admin_train_dataset_version_combo["values"] = []
            self._admin_train_version_display_to_id = {}
            self._admin_train_dataset_version_var.set("")
            return
        # Fetch versions for this dataset
        try:
            versions = self.api.list_dataset_versions(ds_id)
        except Exception:
            versions = []
        ver_values: list[str] = []
        ver_display_to_id: dict[str, str] = {}
        for v in versions:
            v_id = str(v.get("id") or "").strip()
            v_num = str(v.get("version_number") or "?").strip()
            v_status = str(v.get("status") or "unknown").strip()
            v_label = f"v{v_num} ({v_status})"
            ver_values.append(v_label)
            ver_display_to_id[v_label] = v_id
        self._admin_train_version_display_to_id = ver_display_to_id
        self._admin_train_dataset_version_combo["values"] = ver_values
        if ver_values:
            self._admin_train_dataset_version_var.set(ver_values[0])
            self._admin_train_readiness_var.set(f"READY - {images} images, {ann} annotations, {len(ver_values)} version(s)")
        else:
            self._admin_train_dataset_version_var.set("")
            self._admin_train_readiness_var.set(f"READY - {images} images, {ann} annotations. NO VERSION — create one in Data tab first.")

    def _on_admin_train_dataset_selected(self, _event=None) -> None:
        display = self._admin_train_dataset_var.get().strip()
        ds_id = self._admin_train_display_to_id.get(display, display)
        if not ds_id:
            return
        ds = next((item for item in self._datasets_cache if str(item.get("id") or "") == ds_id), None)
        if not ds:
            self._admin_train_readiness_var.set("Dataset not found.")
            return
        images = int(ds.get("image_count") or 0)
        ann = int(ds.get("annotated_image_count") or 0)
        # Sync dataset version dropdown
        self._sync_admin_dataset_versions(ds_id, images, ann)
        self._admin_aug_dataset_var.set(display)
        self._admin_update_augment_estimator()
        self._admin_update_flow_steps()

    def _admin_update_flow_steps(self) -> None:
        if not self._flow_step_vars:
            return
        ds = self._admin_train_dataset_var.get().strip()
        self._flow_step_vars[0].set("\u2713" if ds else "\u25cb")
        ver = self._admin_train_dataset_version_var.get().strip()
        self._flow_step_vars[1].set("\u2713" if ver else "\u25cb")
        bm = self._admin_train_base_model_var.get().strip()
        self._flow_step_vars[2].set("\u2713" if bm else "\u25cb")
        ep = self._admin_train_epochs_var.get().strip()
        self._flow_step_vars[3].set("\u2713" if ep else "\u25cb")
        self._flow_step_vars[4].set("\u2713" if self._active_training_job_id else "\u25cb")

    def _admin_create_augment_job(self) -> None:
        display = self._admin_aug_dataset_var.get().strip()
        ds_id = self._admin_train_display_to_id.get(display, display)
        if not ds_id:
            messagebox.showwarning("Augment", "Select dataset first.")
            return
        transforms = [name for name, var in self._admin_aug_transform_vars.items() if var.get()]
        if not transforms:
            messagebox.showwarning("Augment", "Select at least one transform.")
            return
        multiplier = int(self._admin_aug_multiplier_var.get() or "2")
        try:
            self.api.create_augment_job(
                {
                    "dataset_id": ds_id,
                    "transforms": transforms,
                    "multiplier": multiplier,
                }
            )
        except Exception as exc:
            messagebox.showerror("Augment", str(exc))
            return
        self._admin_refresh_augment_jobs()

    def _admin_refresh_augment_jobs(self) -> None:
        def _load():
            return self.api.list_augment_jobs()
        def _done(result, error):
            if error or not isinstance(result, list):
                return
            self._augment_jobs_cache = result
            self._admin_augment_jobs_list.delete(0, "end")
            for item in result:
                self._admin_augment_jobs_list.insert("end", f"{item.get('id')} | {item.get('status')} | {item.get('transforms')}")
        run_async(self, _load, callback=_done)

    def _admin_create_training_job(self) -> None:
        display = self._admin_train_dataset_var.get().strip()
        ds_id = self._admin_train_display_to_id.get(display, display)
        if not ds_id:
            messagebox.showwarning("Training", "Select dataset first.")
            return
        ds = next((item for item in self._datasets_cache if str(item.get("id") or "") == ds_id), None)
        if ds:
            images = int(ds.get("image_count") or 0)
            ann = int(ds.get("annotated_image_count") or 0)
            if images == 0 or ann == 0:
                messagebox.showwarning("Training", f"Dataset not ready. Images: {images}, Annotations: {ann}")
                return
        # Check dataset version selection
        version_display = self._admin_train_dataset_version_var.get().strip()
        ds_version_id = self._admin_train_version_display_to_id.get(version_display, version_display) if version_display else None
        bm_label = self._admin_train_base_model_var.get().strip()
        if not bm_label:
            messagebox.showwarning("Training", "Select base model first.")
            return
        if not ds_version_id:
            messagebox.showwarning("Training", "Select a dataset version. Go to Data tab → create a version first.")
            return
        # Resolve base model: combo has display labels, worker needs the catalog id
        bm_spec = self._base_model_lookup.get(bm_label)
        bm_id = bm_spec["id"] if bm_spec else bm_label
        try:
            epochs = int(self._admin_train_epochs_var.get())
            imgsz = int(self._admin_train_imgsz_var.get())
            batch = int(self._admin_train_batch_var.get())
            patience = int(self._admin_train_patience_var.get())
            workers = int(self._admin_train_workers_var.get())
        except ValueError:
            messagebox.showwarning("Training", "Hyperparameters must be integers.")
            return
        payload = {
            "dataset_id": ds_id,
            "dataset_version_id": ds_version_id,
            "base_model": bm_id,
            "device_mode": self._admin_train_device_var.get().strip() or "auto",
            "epochs": epochs, "imgsz": imgsz, "batch": batch,
            "patience": patience, "workers": workers,
            "cache": self._admin_train_cache_var.get(),
        }
        try:
            created = self.api.create_training_job(payload)
        except Exception as exc:
            messagebox.showerror("Training", str(exc))
            return
        if isinstance(created, dict):
            self._active_training_job_id = str(created.get("id") or "").strip()
        self._admin_update_flow_steps()
        self._admin_refresh_training_jobs()

    def _admin_cancel_training_job(self) -> None:
        sel = self._admin_training_jobs_list.curselection()
        if not sel:
            return
        job_id = self._admin_training_jobs_list.get(sel[0]).split(" | ")[0]
        try:
            self.api.cancel_training_job(job_id)
        except Exception as exc:
            messagebox.showerror("Training", str(exc))
        self._admin_refresh_training_jobs()

    def _admin_delete_training_job(self) -> None:
        sel = self._admin_training_jobs_list.curselection()
        if not sel:
            return
        job_id = self._admin_training_jobs_list.get(sel[0]).split(" | ")[0]
        if not messagebox.askyesno("Training", f"Delete training job {job_id}?"):
            return
        try:
            self.api.delete_training_job(job_id)
        except Exception as exc:
            messagebox.showerror("Training", str(exc))
        self._admin_refresh_training_jobs()

    def _admin_refresh_training_jobs(self) -> None:
        def _load():
            return self.api.list_training_jobs()
        def _done(result, error):
            if error or not isinstance(result, list):
                return
            self._training_jobs_all = result
            self._admin_training_jobs_list.delete(0, "end")
            for item in result:
                status = str(item.get("status") or "")
                progress = str(item.get("progress_percent") or 0)
                self._admin_training_jobs_list.insert("end", f"{item.get('id')} | {status} | {progress}%")
            self._admin_update_flow_steps()
        run_async(self, _load, callback=_done)

    def _on_admin_training_job_selected(self, _event=None) -> None:
        sel = self._admin_training_jobs_list.curselection()
        if not sel:
            return
        job_id = self._admin_training_jobs_list.get(sel[0]).split(" | ")[0]
        job = next((j for j in self._training_jobs_all if str(j.get("id") or "") == job_id), None)
        if not job:
            return
        self._admin_train_status_var.set(str(job.get("status") or "-"))
        self._admin_train_progress_var.set(f"{job.get('progress_percent', 0)}%")
        self._admin_train_message_var.set(str(job.get("progress_message") or "-"))
        self._admin_train_progress_bar["value"] = int(job.get("progress_percent") or 0)

    # ==================================================================
    # Models Tab - Model registry + export/import
    # ==================================================================
    def _build_models_tab(self) -> None:
        ModelsTab(self, self.models_tab)

    def _on_admin_model_selected(self, _event=None) -> None:
        sel = self._admin_model_list.curselection()
        if not sel:
            return
        display = self._admin_model_list.get(sel[0])
        model = self._model_lookup.get(display, {})
        if model:
            detail = f"Name: {model.get('name')}\nPath: {model.get('path')}\nRuntime: {model.get('runtime', 'ultralytics')}\nStatus: {model.get('status')}\nCreated: {model.get('created_at')}"
            self._admin_model_detail_var.set(detail)

    def _admin_choose_import_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[
                ("All Model Files", "*.pt *.tflite *.onnx *.xml"),
                ("PyTorch Model", "*.pt"),
                ("TFLite Model", "*.tflite"),
                ("ONNX Model", "*.onnx"),
                ("OpenVINO IR Model", "*.xml"),
            ]
        )
        if path:
            self._admin_import_path = path
            self._admin_import_path_var.set(Path(path).name)
            # Auto-detect format from extension
            ext = Path(path).suffix.lower()
            if ext == ".tflite":
                self._admin_import_format_var.set("tflite")
            elif ext == ".onnx":
                self._admin_import_format_var.set("onnx")
            elif ext == ".xml":
                self._admin_import_format_var.set("openvino")
            elif ext == ".pt":
                self._admin_import_format_var.set("pt")
            # Auto-fill model name if empty
            if not self._admin_import_name_var.get().strip():
                self._admin_import_name_var.set(Path(path).stem)

    def _admin_import_model(self) -> None:
        if not self._admin_import_path:
            messagebox.showwarning("Upload", "Choose a file first.")
            return
        name = self._admin_import_name_var.get().strip()
        if not name:
            messagebox.showwarning("Upload", "Enter a model name.")
            return
        # Detect format from extension or user selection
        fmt = self._admin_import_format_var.get()
        ext = Path(self._admin_import_path).suffix.lower()
        if fmt == "auto":
            if ext == ".tflite":
                runtime = "tflite"
            elif ext == ".onnx":
                runtime = "onnx"
            elif ext == ".xml":
                runtime = "openvino"
            else:
                runtime = "ultralytics"
        elif fmt == "tflite":
            runtime = "tflite"
        elif fmt == "onnx":
            runtime = "onnx"
        elif fmt == "openvino":
            runtime = "openvino"
        else:
            runtime = "ultralytics"
        # OpenVINO: upload .bin companion file alongside .xml
        companion_b64: str | None = None
        companion_file_name: str | None = None
        if runtime == "openvino":
            bin_path = Path(self._admin_import_path).with_suffix(".bin")
            if bin_path.exists():
                companion_file_name = bin_path.name
                companion_b64 = base64.b64encode(bin_path.read_bytes()).decode("ascii")
            else:
                messagebox.showwarning(
                    "OpenVINO Upload",
                    f"File .bin companion tidak ditemukan di:\n{bin_path}\n\n"
                    "Model .xml tetap diupload, tapi inference akan gagal tanpa .bin. "
                    "Pastikan file .bin ada di folder models.",
                )
        try:
            payload = {
                "name": name,
                "file_name": Path(self._admin_import_path).name,
                "content_b64": base64.b64encode(Path(self._admin_import_path).read_bytes()).decode("ascii"),
                "runtime": runtime,
            }
            if companion_b64:
                payload["companion_file_name"] = companion_file_name
                payload["companion_b64"] = companion_b64
            self.api.upload_model_file(payload)
        except Exception as exc:
            messagebox.showerror("Upload", str(exc))
            return
        self._admin_import_path = ""
        self._admin_import_path_var.set("No file selected")
        self._admin_import_name_var.set("")
        self.refresh_model_options()
        messagebox.showinfo("Upload", f"Model '{name}' uploaded ({runtime}).")

    def _admin_export_model(self) -> None:
        sel = self._admin_model_list.curselection()
        if not sel:
            messagebox.showwarning("Export", "Select a model first.")
            return
        display = self._admin_model_list.get(sel[0])
        model = self._model_lookup.get(display, {})
        model_id = model.get("id")
        if not model_id:
            return
        try:
            self.api.export_model_archive(int(model_id))
            messagebox.showinfo("Export", "Export started.")
        except Exception as exc:
            messagebox.showerror("Export", str(exc))

    def _admin_delete_model(self) -> None:
        sel = self._admin_model_list.curselection()
        if not sel:
            messagebox.showwarning("Delete", "Select a model first.")
            return
        display = self._admin_model_list.get(sel[0])
        model = self._model_lookup.get(display, {})
        model_id = model.get("id")
        if not model_id:
            return
        if not messagebox.askyesno("Delete", f"Delete model {model.get('name')}?"):
            return
        try:
            self.api.delete_model(int(model_id))
            self.refresh_model_options()
            self._admin_model_detail_var.set("Select a model to view details.")
            messagebox.showinfo("Delete", f"Model '{model.get('name')}' deleted.")
        except Exception as exc:
            messagebox.showerror("Delete", str(exc))

    # ==================================================================
    # Calibration Tab - Color calibration profiles
    # ==================================================================
    def _build_calibration_tab(self) -> None:
        CalibrationTab(self, self.calibration_tab)

    def _admin_refresh_calibration(self) -> None:
        def _load():
            return self.api.list_calibration_profiles()
        def _done(result, error):
            if error or not isinstance(result, list):
                return
            self._calibration_profiles_cache = result
            self._admin_cal_list.delete(0, "end")
            for item in result:
                self._admin_cal_list.insert("end", f"{item.get('name')} | {item.get('id')}")
        run_async(self, _load, callback=_done)

    def _admin_choose_cal_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self._admin_cal_image_path = path
            self._admin_cal_image_var.set(Path(path).name)

    def _admin_create_calibration(self) -> None:
        name = self._admin_cal_name_var.get().strip()
        if not name:
            messagebox.showwarning("Calibration", "Name is required.")
            return
        if not self._admin_cal_image_path:
            messagebox.showwarning("Calibration", "Choose a calibration image first.")
            return
        try:
            content = Path(self._admin_cal_image_path).read_bytes()
            b64 = base64.b64encode(content).decode("ascii")
            self.api.create_calibration_profile(name, self._admin_cal_desc_var.get().strip(), b64, Path(self._admin_cal_image_path).name)
        except Exception as exc:
            messagebox.showerror("Calibration", str(exc))
            return
        self._admin_cal_name_var.set("")
        self._admin_cal_desc_var.set("")
        self._admin_cal_image_path = ""
        self._admin_cal_image_var.set("No image selected")
        self._admin_refresh_calibration()

    def _admin_delete_calibration(self) -> None:
        sel = self._admin_cal_list.curselection()
        if not sel:
            messagebox.showwarning("Calibration", "Select a profile first.")
            return
        item_str = self._admin_cal_list.get(sel[0])
        cal_id = item_str.split(" | ")[-1] if " | " in item_str else ""
        if not cal_id:
            return
        if not messagebox.askyesno("Calibration", f"Delete profile {cal_id}?"):
            return
        try:
            self.api.delete_calibration_profile(cal_id)
        except Exception as exc:
            messagebox.showerror("Calibration", str(exc))
        self._admin_refresh_calibration()

    # ==================================================================
    # Annotation Methods
    # ==================================================================

    def _admin_resolve_annotation_dataset_id(self) -> str | None:
        sel = self._admin_dataset_list.curselection()
        if not sel:
            return self._annotation_dataset_id
        display = self._admin_dataset_list.get(sel[0])
        return self._dataset_display_to_id.get(display, self._annotation_dataset_id)

    def _admin_annot_cache_key(self, dataset_id: str | None, image_name: str | None) -> tuple[str, str] | None:
        if not dataset_id or not image_name:
            return None
        return (str(dataset_id), str(image_name))

    def _admin_annot_cache_entry_size(self, asset: dict) -> int:
        frame = asset.get("frame")
        if frame is None:
            return 0
        return frame.nbytes + 256

    def _admin_annot_cache_get(self, key: tuple[str, str] | None) -> dict | None:
        if key is None:
            return None
        return self._annotation_cache.get(key)

    def _admin_annot_cache_store(self, key: tuple[str, str] | None, asset: dict) -> None:
        if key is None:
            return
        while len(self._annotation_cache) >= self._annotation_cache_max_items:
            _, old = self._annotation_cache.popitem(last=False)
            self._annotation_cache_bytes -= self._admin_annot_cache_entry_size(old)
        size = self._admin_annot_cache_entry_size(asset)
        while self._annotation_cache and self._annotation_cache_bytes + size > self._annotation_cache_max_bytes:
            _, old = self._annotation_cache.popitem(last=False)
            self._annotation_cache_bytes -= self._admin_annot_cache_entry_size(old)
        self._annotation_cache[key] = asset
        self._annotation_cache_bytes += size

    def _admin_annot_cache_update_labels(self, dataset_id: str | None, image_name: str | None, labels: list[dict]) -> None:
        key = self._admin_annot_cache_key(dataset_id, image_name)
        entry = self._admin_annot_cache_get(key)
        if entry is not None:
            entry["labels"] = copy.deepcopy(labels)

    def admin_refresh_annotation_images(self) -> None:
        dataset_id = self._admin_resolve_annotation_dataset_id()
        self._annotation_files = []
        self._annotation_index = None
        self._admin_annot_image_listbox.delete(0, "end")
        if not dataset_id:
            self._admin_reset_annotation_state()
            return
        self._annotation_dataset_id = dataset_id
        self._annotation_files_refresh_sequence += 1
        refresh_seq = self._annotation_files_refresh_sequence

        def _load():
            return self.api.list_dataset_files(dataset_id, "images")

        def _done(result, error):
            if refresh_seq != self._annotation_files_refresh_sequence:
                return
            if not self.winfo_exists():
                return
            if self._admin_resolve_annotation_dataset_id() != dataset_id:
                return
            if error:
                messagebox.showerror("Annotate", str(error))
                self._admin_reset_annotation_state()
                return
            if not isinstance(result, list):
                self._admin_reset_annotation_state()
                return
            self._annotation_files = result
            self._admin_annot_image_listbox.delete(0, "end")
            for item in result:
                name = str(item.get("name") or "").strip()
                if name:
                    self._admin_annot_image_listbox.insert("end", name)
            if not result:
                self._admin_annot_image_var.set("-")
                self._admin_annotation_canvas.clear()
                self._admin_annotation_status_var.set("Dataset has no images to annotate.")
                self._admin_update_annot_nav_state()
                self._admin_annot_coverage_var.set("Coverage: -")
                return
            self._admin_update_annotation_coverage()

        run_async(self, _load, callback=_done)

    def _admin_reset_annotation_state(self) -> None:
        self._annotation_index = None
        self._annotation_files = []
        try:
            self._admin_annot_image_listbox.delete(0, "end")
            self._admin_annot_image_var.set("-")
            self._admin_annot_coverage_var.set("Coverage: -")
            if hasattr(self, "_admin_annotation_canvas"):
                self._admin_annotation_canvas.clear()
                self._admin_annotation_status_var.set("Select a dataset to start annotating.")
        except tk.TclError:
            pass

    def _on_admin_annot_image_list_selected(self, _event=None) -> None:
        sel = self._admin_annot_image_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx == self._annotation_index:
            return
        self._admin_load_annotation_for_index(idx)

    def _admin_annot_image_index(self, image_name: str | None = None) -> int | None:
        name = str(image_name or self._admin_annot_image_var.get() or "").strip()
        if not name or name == "-":
            return None
        for idx, item in enumerate(self._annotation_files):
            if str(item.get("name") or "").strip() == name:
                return idx
        return None

    def _admin_annot_widgets_alive(self) -> bool:
        try:
            return self._admin_annotation_canvas.winfo_exists()
        except Exception:
            return False

    def _admin_fetch_annotation_asset(self, dataset_id: str, image_name: str, image_path: str) -> dict:
        base_url = str(getattr(self.state, "base_url", None) or getattr(self.api, "base_url", "") or "").strip()
        worker_api = ApiClient(base_url)
        worker_api.set_token(getattr(self.state, "token", None))
        frame = None
        loaded_source = ""
        try:
            image_bytes = worker_api.download_dataset_image(dataset_id, image_name)
        except Exception:
            image_bytes = b""
        if image_bytes:
            raw = np.frombuffer(image_bytes, np.uint8)
            if raw.size > 0:
                frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if frame is not None:
                    loaded_source = "backend"
        if frame is None and image_path:
            path = Path(image_path)
            if path.exists():
                try:
                    raw_bytes = path.read_bytes()
                except OSError:
                    raw_bytes = b""
                if raw_bytes:
                    raw = np.frombuffer(raw_bytes, np.uint8)
                    if raw.size > 0:
                        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                        if frame is not None:
                            loaded_source = "local"
        if frame is None:
            raise ValueError(f"{image_name or 'image'} could not be loaded")
        labels_payload: list[dict] = []
        try:
            payload = worker_api.get_annotation(dataset_id, image_name)
        except Exception:
            payload = {"labels": []}
        labels = payload.get("labels") if isinstance(payload, dict) else []
        if isinstance(labels, list):
            labels_payload = [copy.deepcopy(item) for item in labels if isinstance(item, dict)]
        return {
            "dataset_id": dataset_id, "image_name": image_name,
            "image_path": image_path, "frame": frame,
            "labels": labels_payload, "loaded_source": loaded_source,
        }

    def _admin_apply_annotation_asset(self, asset: dict, *, base_status: str, source_label: str) -> None:
        frame = asset.get("frame")
        if frame is None:
            return
        image_name = str(asset.get("image_name") or "").strip()
        labels_payload = asset.get("labels") if isinstance(asset.get("labels"), list) else []
        self._admin_annotation_canvas.load_bgr(frame, image_name=image_name, redraw=False)
        self._admin_annotation_canvas.set_image_name(image_name)
        self._admin_annotation_canvas.set_class_name(self._annotation_class_name, redraw=False)
        self._admin_annotation_canvas.set_labels(labels_payload, redraw=False)
        self._admin_annotation_canvas.redraw()
        self._admin_annot_image_var.set(image_name)
        self._admin_annotation_status_var.set(f"{base_status} | loaded via {source_label}")
        self._admin_update_annotation_coverage()
        self._admin_update_annot_nav_state()
        # Sync image listbox
        if self._admin_annot_image_listbox.size() > 0:
            self._admin_annot_image_listbox.selection_clear(0, "end")
            self._admin_annot_image_listbox.selection_set(self._annotation_index)
            self._admin_annot_image_listbox.see(self._annotation_index)

    def _admin_load_annotation_for_index(self, index: int, *, save_current: bool = True) -> None:
        if index < 0 or index >= len(self._annotation_files):
            return
        if save_current and self._annotation_index is not None and not self._admin_save_current_annotation(silent=True):
            self._admin_annotation_status_var.set("Autosave failed. Stay on current image.")
            return
        item = self._annotation_files[index]
        dataset_id = self._admin_resolve_annotation_dataset_id()
        image_name = str(item.get("name") or "").strip()
        image_path = str(item.get("path") or "").strip()
        self._annotation_index = index
        base_status = f"{index + 1} / {len(self._annotation_files)}"
        cache_key = self._admin_annot_cache_key(dataset_id, image_name)
        cached = self._admin_annot_cache_get(cache_key)
        self._annotation_load_sequence += 1
        load_seq = self._annotation_load_sequence
        if cached is not None:
            self._admin_apply_annotation_asset(cached, base_status=base_status, source_label="cache")
            return
        self._admin_annotation_canvas.clear()
        self._admin_annotation_status_var.set(f"{base_status} | loading...")
        if not dataset_id or not image_name:
            self._admin_annotation_status_var.set(f"{image_name or 'image'} could not be loaded")
            return

        def _load():
            return self._admin_fetch_annotation_asset(dataset_id, image_name, image_path)

        def _done(result, error):
            if load_seq != self._annotation_load_sequence:
                return
            if not self._admin_annot_widgets_alive():
                return
            if self._annotation_dataset_id != dataset_id:
                return
            if self._annotation_index != index:
                return
            if error:
                self._admin_annotation_status_var.set(f"{image_name or 'image'} could not be loaded")
                return
            if not isinstance(result, dict):
                self._admin_annotation_status_var.set(f"{image_name or 'image'} could not be loaded")
                return
            self._admin_annot_cache_store(cache_key, result)
            source_label = str(result.get("loaded_source") or "").strip() or "loaded"
            self._admin_apply_annotation_asset(result, base_status=base_status, source_label=source_label)

        run_async(self, _load, callback=_done)

    def _admin_update_annot_nav_state(self) -> None:
        has_files = bool(self._annotation_files)
        is_first = self._annotation_index == 0 if has_files else True
        is_last = self._annotation_index == len(self._annotation_files) - 1 if has_files else True

    def _on_admin_annot_labels_changed(self, _labels: list[dict]) -> None:
        self._admin_save_current_annotation(silent=True)

    def _admin_save_current_annotation(self, *, silent: bool = False) -> bool:
        dataset_id = self._admin_resolve_annotation_dataset_id()
        image_name = self._admin_annot_image_var.get().strip()
        if not dataset_id or not image_name or image_name == "-":
            return False
        try:
            payload = self._admin_annotation_canvas.get_labels()
            self.api.save_annotation(dataset_id, image_name, payload)
        except Exception as exc:
            self._admin_annotation_status_var.set(f"Autosave failed: {image_name}")
            if not silent:
                messagebox.showerror("Annotate", str(exc))
            return False
        self._admin_annot_cache_update_labels(dataset_id, image_name, payload)
        self._admin_annotation_status_var.set(f"Saved {image_name}")
        # Update coverage
        self._admin_update_annotation_coverage()
        return True

    def _admin_save_current_annotation_interactive(self) -> None:
        dataset_id = self._admin_resolve_annotation_dataset_id()
        image_name = self._admin_annot_image_var.get().strip()
        if not dataset_id or not image_name or image_name == "-":
            messagebox.showwarning("Annotate", "Select a dataset and image first.")
            return
        self._admin_save_current_annotation(silent=False)

    def _admin_update_annotation_coverage(self) -> None:
        total = len(self._annotation_files)
        if total == 0:
            self._admin_annot_coverage_var.set("Coverage: -")
            return
        server_annotated = 0
        if self._annotation_dataset_id:
            ds = next((item for item in self._datasets_cache if str(item.get("id") or "") == self._annotation_dataset_id), None)
            if ds:
                server_annotated = int(ds.get("annotated_image_count") or 0)
        cached_annotated = 0
        for item in self._annotation_files:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            cache_key = self._admin_annot_cache_key(self._annotation_dataset_id, name)
            cached = self._admin_annot_cache_get(cache_key)
            if cached is not None:
                if cached.get("labels"):
                    cached_annotated += 1
        annotated = min(total, max(server_annotated, cached_annotated))
        self._admin_annot_coverage_var.set(f"Coverage: {annotated}/{total} ({annotated*100//total if total else 0}%)")

    def _sync_admin_annotation_dataset_combo(self) -> None:
        values = list(self._dataset_id_to_display.values())
        self._admin_annot_dataset_combo["values"] = values
        if self._annotation_dataset_id:
            display = self._dataset_id_to_display.get(self._annotation_dataset_id, "")
            if display:
                self._admin_annot_dataset_combo.set(display)


    def admin_previous_annotation_image(self) -> None:
        if self._annotation_index is None:
            return
        self._admin_load_annotation_for_index(max(0, self._annotation_index - 1))

    def admin_next_annotation_image(self) -> None:
        if self._annotation_index is None:
            return
        self._admin_load_annotation_for_index(min(len(self._annotation_files) - 1, self._annotation_index + 1))

    def _admin_add_annotation_class(self) -> None:
        class_name = str(self._admin_annot_new_class_var.get() or "").strip()
        if not class_name:
            return
        if class_name not in self._annotation_manual_classes:
            self._annotation_manual_classes.append(class_name)
        all_classes = ["object"] + self._annotation_manual_classes
        if class_name not in all_classes:
            all_classes.append(class_name)
        self._admin_annot_class_combo["values"] = all_classes
        self._admin_annot_class_combo.set(class_name)
        self._admin_annot_new_class_var.set("")
        self._on_admin_annot_class_input()

    def _admin_remove_annotation_class(self) -> None:
        class_name = str(self._admin_annot_class_combo.get() or "").strip()
        if not class_name or class_name == "object":
            return
        if class_name in self._annotation_manual_classes:
            self._annotation_manual_classes.remove(class_name)
        all_classes = ["object"] + self._annotation_manual_classes
        self._admin_annot_class_combo["values"] = all_classes
        self._admin_annot_class_combo.set("object")
        self._on_admin_annot_class_input()

    def _admin_sync_annotation_mode(self, *, redraw: bool = True) -> None:
        if not hasattr(self, "_admin_annotation_canvas"):
            return
        self._admin_annotation_canvas.set_mode(self._admin_annot_shape.get(), redraw=redraw)

    def _on_admin_annot_selection_changed(self, label: dict | None, index: int | None) -> None:
        self._annotation_selected_label_index = index
        if label:
            class_name = str(label.get("class_name") or label.get("class") or "object").strip()
            if class_name and class_name != self._admin_annot_class_var.get():
                self._admin_annot_class_combo.set(class_name)
                self._annotation_class_name = class_name

    def _on_admin_annot_class_input(self, _event=None) -> None:
        class_name = str(self._admin_annot_class_combo.get() or "").strip()
        if class_name:
            self._annotation_class_name = class_name
            if hasattr(self, "_admin_annotation_canvas"):
                self._admin_annotation_canvas.set_class_name(class_name)
            if self._annotation_selected_label_index is not None:
                if self._admin_annotation_canvas.set_selected_label_class_name(class_name):
                    self._admin_save_current_annotation(silent=True)

    def _admin_apply_class_to_selected_annotation(self) -> None:
        class_name = str(self._admin_annot_class_combo.get() or "").strip()
        if not class_name or self._annotation_selected_label_index is None:
            return
        if self._admin_annotation_canvas.set_selected_label_class_name(class_name):
            self._admin_save_current_annotation(silent=True)

    def _admin_delete_selected_annotation(self) -> None:
        self._admin_annotation_canvas.delete_selected_label()

    # Keyboard shortcuts
    def _on_admin_annot_shortcut_save(self, _event=None) -> None:
        self._admin_save_current_annotation_interactive()
        return "break"

    def _on_admin_annot_shortcut_prev(self, _event=None) -> None:
        self.admin_previous_annotation_image()
        return "break"

    def _on_admin_annot_shortcut_next(self, _event=None) -> None:
        self.admin_next_annotation_image()
        return "break"

    def _on_admin_annot_shortcut_bbox(self, _event=None) -> None:
        self._admin_annot_shape.set("bbox")
        self._admin_sync_annotation_mode()
        return "break"

    def _on_admin_annot_shortcut_polygon(self, _event=None) -> None:
        self._admin_annot_shape.set("polygon")
        self._admin_sync_annotation_mode()
        return "break"

    def _on_admin_annot_shortcut_delete(self, _event=None) -> None:
        self._admin_delete_selected_annotation()
        return "break"

    # ==================================================================
    # Augment Estimator
    # ==================================================================

    def _admin_update_augment_estimator(self) -> None:
        display = self._admin_aug_dataset_var.get().strip()
        ds_id = self._admin_train_display_to_id.get(display, display)
        if not ds_id:
            self._admin_aug_estimator_var.set("Select a dataset to see output estimate.")
            return
        ds = next((item for item in self._datasets_cache if str(item.get("id") or "") == ds_id), None)
        if not ds:
            self._admin_aug_estimator_var.set("Dataset info not loaded yet.")
            return
        source = int(ds.get("image_count") or 0)
        transforms = [name for name, var in self._admin_aug_transform_vars.items() if var.get()]
        multiplier = int(self._admin_aug_multiplier_var.get() or "2")
        if source == 0:
            self._admin_aug_estimator_var.set("Dataset has 0 images. Upload images first.")
            return
        if not transforms:
            self._admin_aug_estimator_var.set("Select at least one transform.")
            return
        per_image = len(transforms) * multiplier
        total_aug = source * per_image
        grand = source + total_aug
        t_names = ", ".join(transforms)
        self._admin_aug_estimator_var.set(
            f"Source: {source} imgs | Transforms: {t_names} ({len(transforms)}), "
            f"multiplier={multiplier} | Per image: {per_image} aug | "
            f"Total aug: {total_aug} | Grand total: {grand} imgs"
        )
