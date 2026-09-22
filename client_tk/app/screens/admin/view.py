from __future__ import annotations

import base64
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import customtkinter as ctk
import cv2
import numpy as np

from client_tk.app.components.async_bridge import run_async
from client_tk.app.components.scrollable_frame import AutoHideScrollbar, ScrollableFrame
from client_tk.app.screens.admin.tabs.templates_tab import TemplatesTab
from client_tk.app.screens.admin.tabs.operators_tab import OperatorsTab
from client_tk.app.screens.admin.tabs.models_tab import ModelsTab
from client_tk.app.screens.admin.tabs.results_tab import ResultsTab
from client_tk.app.screens.admin.tabs.machine_settings_tab import MachineSettingsTab
from client_tk.app.theme import (
    ACCENT,
    ACCENT_HOVER,
    APP_BG,
    BORDER,
    PANEL_ALT_BG,
    PANEL_BG,
    SHELL_BG,
    TEXT_ON_ACCENT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)




RESPONSIVE_BREAKPOINT = 1180


def _sibling_class_names(model_path: Path) -> list[str]:
    """Class names for a non-.pt model from `<stem>.meta.json`, `<stem>.json` or
    an Ultralytics `metadata.yaml` in the same folder (the backends only read the
    `.meta.json`, so the upload must carry the names explicitly)."""
    import json

    for candidate in (model_path.with_suffix(".meta.json"), model_path.with_suffix(".json")):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        names = data.get("class_names") or data.get("names") if isinstance(data, dict) else None
        if isinstance(names, dict):
            return [str(v) for _, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
        if isinstance(names, list):
            return [str(v) for v in names]
    for candidate in (model_path.parent / "metadata.yaml", model_path.parent / "metadata.yml"):
        try:
            import yaml

            data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        names = data.get("names") if isinstance(data, dict) else None
        if isinstance(names, dict):
            return [str(v) for _, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
        if isinstance(names, list):
            return [str(v) for v in names]
    return []


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
        self._users_cache: list[dict] = []
        self._results_cache: list[dict] = []

        self.current_template_id: int | None = None
        self.current_template_version_id: int | None = None
        self._editing_deployment_id: int | None = None

        self.status_var = tk.StringVar(value="Admin ready.")
        self.refresh_time_var = tk.StringVar(value="")
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
        self.preset_gap_threshold_var = tk.StringVar(value="0.85")
        self.preset_camera_index_var = tk.StringVar(value="0")
        self.preset_roi_choice_var = tk.StringVar(value="Sticker ROI")
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
        for tab_name in ("Templates", "Models", "Operators", "Monitor", "Machine Settings"):
            notebook.add(tab_name)

        original_tab = notebook.tab

        def _tabs() -> list[str]:
            return ["Templates", "Models", "Operators", "Monitor", "Machine Settings"]

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
        self.models_tab = notebook.tab("Models")
        self.operators_tab = notebook.tab("Operators")
        self.monitor_tab = notebook.tab("Monitor")
        self.plc_settings_tab = notebook.tab("Machine Settings")

        self._build_presets_tab()
        self._build_models_tab()
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
        self.refresh_operators()
        self.refresh_monitor()

    def request_backend_restart(self) -> None:
        """Restart the app/backend so Connection / Inference settings take effect."""
        app = self.winfo_toplevel()
        can_restart = getattr(app, "can_restart_backend", None)
        if can_restart is None or not can_restart():
            messagebox.showinfo(
                "Restart Backend",
                "Backend berjalan di proses/server lain. Restart backend di sana secara manual "
                "(scripts/run_backend.py), lalu login lagi.",
            )
            return
        if not messagebox.askyesno(
            "Restart Backend",
            "Aplikasi akan ditutup dan dijalankan ulang otomatis.\n"
            "Session login akan hilang. Lanjutkan?",
        ):
            return
        app.restart_app()

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
            self.preset_table.insert(
                "",
                "end",
                iid=f"dep:{item.get('id')}",
                values=(
                    f"D{item.get('id')}",
                    _safe_text(latest_name),
                    _safe_text(item.get("template_version_id")),
                    "ACTIVE",
                ),
            )
        for item in self._templates_cache:
            template_id = int(item.get("id") or 0)
            if template_id in active_template_ids:
                continue
            self.preset_table.insert(
                "",
                "end",
                iid=f"tpl:{template_id}",
                values=(
                    f"T{template_id}",
                    _safe_text(item.get("name")),
                    _safe_text(item.get("version_id") or item.get("current_version_id")),
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
            self.results_table.insert(
                "",
                "end",
                iid=str(item.get("id")),
                values=(
                    item.get("id"),
                    _format_timestamp(item.get("inspected_at")),
                    _safe_text(decision),
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
        "preset_model_choice_var": "",
        "preset_model_path_var": "",
        "preset_model_meta_path_var": "",
        "preset_model_classes_var": "",
        "preset_runtime_var": "auto",
        "preset_conf_threshold_var": "0.25",
        "preset_expected_class_var": "",
        "preset_gap_threshold_var": "0.85",
        "preset_canny_low_var": "",
        "preset_canny_high_var": "",
        "preset_gap_search_margin_var": "0.0",
        "preset_part_ready_method_var": "gap_template_match",
        "preset_mean_max_var": "105.0",
        "preset_std_max_var": "35.0",
        "preset_min_match_ratio_var": "0.5",
        "preset_camera_index_var": "0",
        "preset_roi_choice_var": "Part Ready ROI",
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
                self._sync_preset_roi_picker()
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
        # Clear gap ref status
        if hasattr(self, "gap_ref_status_label"):
            try:
                self.gap_ref_status_label.configure(
                    text="Referensi: belum dikonfigurasi", foreground="gray")
            except Exception:
                pass
        self._set_gap_ref_preview_image(None)

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
        self.preset_expected_class_var.set(str(sticker.get("expected_class") or ""))
        self.preset_gap_threshold_var.set(str(part_ready.get("gap_match_threshold", 0.85)))
        self.preset_canny_low_var.set("" if part_ready.get("canny_low") is None else str(part_ready.get("canny_low")))
        self.preset_canny_high_var.set("" if part_ready.get("canny_high") is None else str(part_ready.get("canny_high")))
        self.preset_gap_search_margin_var.set(str(part_ready.get("gap_search_margin", 0.0)))
        # Part ready method and mean-std thresholds
        _method = str(part_ready.get("method") or "").strip() or "gap_template_match"
        self.preset_part_ready_method_var.set(_method)
        self.preset_mean_max_var.set(str(part_ready.get("mean_max", 105.0)))
        self.preset_std_max_var.set(str(part_ready.get("std_max", 35.0)))
        self.preset_min_match_ratio_var.set(str(part_ready.get("min_match_ratio", 0.5)))
        # Update gap ref status label
        _gap_ref_path = detail.get("gap_ref_path") or part_ready.get("gap_ref_path")
        if _gap_ref_path:
            from pathlib import Path
            if not Path(_gap_ref_path).is_file():
                self.gap_ref_status_label.configure(
                    text="Referensi: file tidak ditemukan", foreground="orange")
            else:
                self.gap_ref_status_label.configure(
                    text="Referensi: edge map ✓", foreground="green")
        else:
            self.gap_ref_status_label.configure(
                text="Referensi: belum dikonfigurasi", foreground="gray")
        self._refresh_gap_ref_preview()
        camera_cfg = detail.get("camera") or {}
        self.preset_camera_index_var.set(str(camera_cfg.get("camera_index", 0)))
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
        self._sync_preset_roi_picker()

        vision = detail.get("vision") or {}
        _model_for_selector = str(vision.get("model_path") or "")
        self.preset_model_path_var.set(_model_for_selector)
        self.preset_model_meta_path_var.set(str(vision.get("model_meta_path") or ""))
        self.preset_conf_threshold_var.set(str(vision.get("conf_threshold", 0.25)))
        self.preset_runtime_var.set(str(vision.get("runtime") or "auto"))
        self._select_model_label_for_path(_model_for_selector)

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

    def _sync_preset_roi_picker(self) -> None:
        if not hasattr(self, "preset_roi_picker"):
            return
        self.preset_roi_picker.set_rois(
            part_ready_roi=self._roi_payload_from_vars(
                self.part_ready_roi_x_var,
                self.part_ready_roi_y_var,
                self.part_ready_roi_w_var,
                self.part_ready_roi_h_var,
                defaults={"x": 0.2, "y": 0.2, "w": 0.25, "h": 0.25},
            ),
            sticker_roi=self._roi_payload_from_vars(
                self.sticker_roi_x_var,
                self.sticker_roi_y_var,
                self.sticker_roi_w_var,
                self.sticker_roi_h_var,
                defaults={"x": 0.2, "y": 0.2, "w": 0.6, "h": 0.6},
            ),
        )
        self.preset_roi_picker.set_active_roi(self._preset_roi_kind())

    def _on_preset_roi_selected(self, _event=None) -> None:
        self._sync_preset_roi_picker()

    def _on_preset_roi_changed(self, kind: str, roi: dict) -> None:
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
            try:
                if version_id:
                    detail = self.api.get_template_version(version_id)
                else:
                    detail = self.api.get_template(template_id)
                self._apply_preset_detail(detail, deployment=None)
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
        # Re-apply saved data to wizard so UI reflects the stored version
        try:
            reloaded = self.api.get_template(template_id) if template_id else None
            if reloaded:
                self._apply_preset_detail(reloaded, deployment=deployment)
                self._set_status("Preset deployed.")
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

    def _current_canny_bounds(self) -> tuple[int | None, int | None]:
        """Parse the wizard's Canny lower/upper fields; empty = auto (None)."""
        def _parse(var_name: str) -> int | None:
            var = getattr(self, var_name, None)
            text = str(var.get()).strip() if var is not None else ""
            if not text:
                return None
            try:
                return int(float(text))
            except (TypeError, ValueError):
                return None
        return _parse("preset_canny_low_var"), _parse("preset_canny_high_var")

    def _set_gap_ref_preview_image(self, preview_b64: str | None) -> None:
        """Render the saved master edge-map as a persistent thumbnail below the capture button."""
        label = getattr(self, "gap_ref_preview_label", None)
        if label is None:
            return
        if not preview_b64:
            try:
                label.configure(image="", text="(belum ada foto master)")
            except Exception:
                pass
            self._gap_ref_preview_photo = None
            return
        try:
            import base64 as _b64
            import numpy as np
            import cv2
            from PIL import Image, ImageTk

            raw = _b64.b64decode(preview_b64)
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                label.configure(image="", text="(preview tidak tersedia)")
                self._gap_ref_preview_photo = None
                return
            max_w, max_h = 260, 160
            h, w = img.shape[:2]
            scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            photo = ImageTk.PhotoImage(Image.fromarray(img))
            label.configure(image=photo, text="")
            self._gap_ref_preview_photo = photo  # keep a reference, tkinter needs it
        except Exception:
            label.configure(image="", text="(preview tidak tersedia)")
            self._gap_ref_preview_photo = None

    def _refresh_gap_ref_preview(self) -> None:
        """Fetch the currently saved master reference (if any) and show it inline."""
        if not self.current_template_id:
            self._set_gap_ref_preview_image(None)
            return
        try:
            result = self.api.get_part_ready_ref_preview(self.current_template_id)
            self._set_gap_ref_preview_image(result.get("preview_b64") if result.get("exists") else None)
        except Exception:
            self._set_gap_ref_preview_image(None)

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
            roi = {
                "x": int(float(self.part_ready_roi_x_var.get() or 0.2) * frame.shape[1]),
                "y": int(float(self.part_ready_roi_y_var.get() or 0.2) * frame.shape[0]),
                "w": int(float(self.part_ready_roi_w_var.get() or 0.25) * frame.shape[1]),
                "h": int(float(self.part_ready_roi_h_var.get() or 0.25) * frame.shape[0]),
            }
            _, buf = cv2.imencode(".png", frame)
            frame_b64 = base64.b64encode(buf).decode("ascii")
            canny_low, canny_high = self._current_canny_bounds()
            result = self.api.capture_part_ready_ref(
                self.current_template_id, frame_b64, roi, canny_low=canny_low, canny_high=canny_high)
            if result.get("saved"):
                warning = result.get("warning")
                if warning:
                    self.gap_ref_status_label.configure(
                        text="Referensi: edge map kosong ⚠", foreground="orange")
                    messagebox.showwarning("Reference", warning)
                else:
                    self.gap_ref_status_label.configure(
                        text="Referensi: edge map ✓", foreground="green")
                self._set_gap_ref_preview_image(result.get("preview_b64"))
            else:
                messagebox.showerror("Reference", result.get("error", "Gagal menyimpan referensi."))
        except Exception as exc:
            messagebox.showerror("Reference", f"Capture failed: {exc}")

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
            canny_low, canny_high = self._current_canny_bounds()
            result = self.api.upload_part_ready_ref(
                self.current_template_id, file_path, canny_low=canny_low, canny_high=canny_high)
            if result.get("saved"):
                warning = result.get("warning")
                if warning:
                    self.gap_ref_status_label.configure(
                        text="Referensi: edge map kosong ⚠", foreground="orange")
                    messagebox.showwarning("Reference", warning)
                else:
                    self.gap_ref_status_label.configure(
                        text="Referensi: edge map ✓", foreground="green")
                    messagebox.showinfo("Reference", "Referensi edge map berhasil diupload.")
                self._set_gap_ref_preview_image(result.get("preview_b64"))
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
        if not model_path:
            raise ValueError("Model is required.")
        if not expected_class:
            raise ValueError("Expected class is required.")

        return {
            "id": self.current_template_id,
            "version_id": self.current_template_version_id,
            "version_number": 1,
            "name": name,
            "description": self.preset_description_var.get().strip(),
            "is_active": True,
            "camera": {
                "camera_index": int(_float_or_default(self.preset_camera_index_var.get(), 0)),
                "width": None, "height": None, "fps": None,
            },
            "part_ready_roi": {
                "x": _float_or_default(self.part_ready_roi_x_var.get(), 0.2),
                "y": _float_or_default(self.part_ready_roi_y_var.get(), 0.2),
                "w": _float_or_default(self.part_ready_roi_w_var.get(), 0.25),
                "h": _float_or_default(self.part_ready_roi_h_var.get(), 0.25),
            },
            "sticker_roi": {
                "x": _float_or_default(self.sticker_roi_x_var.get(), 0.2),
                "y": _float_or_default(self.sticker_roi_y_var.get(), 0.2),
                "w": _float_or_default(self.sticker_roi_w_var.get(), 0.6),
                "h": _float_or_default(self.sticker_roi_h_var.get(), 0.6),
            },
            "vision": {
                "model_path": model_path,
                "model_meta_path": self.preset_model_meta_path_var.get().strip() or None,
                "runtime": self.preset_runtime_var.get().strip() or "auto",
                "conf_threshold": _float_or_default(self.preset_conf_threshold_var.get(), 0.15),
                "inference_fps": 4.0,
                "imgsz": 640,
                "classes": [c.strip() for c in self.preset_model_classes_var.get().split(",") if c.strip()] or [expected_class],
            },
            "part_ready": {
                "enabled": True,
                "method": self.preset_part_ready_method_var.get(),
                "gap_match_threshold": _float_or_default(self.preset_gap_threshold_var.get(), 0.85),
                "gap_ref_path": self._get_existing_gap_ref_path(),
                "canny_low": self._current_canny_bounds()[0],
                "canny_high": self._current_canny_bounds()[1],
                "gap_search_margin": _float_or_default(self.preset_gap_search_margin_var.get(), 0.0),
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
                "min_roi_confidence": 0.0,
                "min_class_confidence": None,
                "max_offset_x": 80,
                "max_offset_y": 80,
                "expected_center_x": None,
                "expected_center_y": None,
                "part_ready_settle_ms": None,
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

    def import_model_archive(self) -> None:
        """Import a model package (.zip: OpenVINO folder export, Ultralytics export,
        or a QC Suite export). The backend unpacks it into data/models/<name>/."""
        path = filedialog.askopenfilename(
            filetypes=[("Model Archive", "*.zip"), ("All Files", "*.*")],
            title="Import Model Archive",
        )
        if not path:
            return
        name = self._admin_archive_name_var.get().strip() if hasattr(self, "_admin_archive_name_var") else ""
        self._set_status("Importing model archive...")

        def _load():
            return self.api.import_model_archive(path, name=name or None)

        def _done(result, error):
            if error:
                messagebox.showerror("Import Model Archive", str(error))
                self._set_status("Model archive import failed.")
                return
            data = result or {}
            if hasattr(self, "_admin_archive_name_var"):
                self._admin_archive_name_var.set("")
            self.refresh_model_options()
            lines = [
                f"Name: {data.get('imported_name')}",
                f"Runtime: {data.get('runtime')}",
                f"Folder: {data.get('model_dir')}",
                f"Classes: {', '.join(data.get('class_names') or []) or '-'}",
            ]
            for warning in data.get("warnings") or []:
                lines.append(f"Warning: {warning}")
            messagebox.showinfo("Import Model Archive", "\n".join(lines))
            self._set_status(f"Model archive imported: {data.get('imported_name')}.")

        run_async(self, _load, callback=_done)

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
        class_names = _sibling_class_names(Path(self._admin_import_path)) if runtime != "ultralytics" else []
        try:
            payload = {
                "name": name,
                "file_name": Path(self._admin_import_path).name,
                "content_b64": base64.b64encode(Path(self._admin_import_path).read_bytes()).decode("ascii"),
                "runtime": runtime,
                "class_names": class_names,
            }
            if companion_b64:
                payload["companion_file_name"] = companion_file_name
                payload["companion_b64"] = companion_b64
            result = self.api.upload_model_file(payload)
        except Exception as exc:
            messagebox.showerror("Upload", str(exc))
            return
        self._admin_import_path = ""
        self._admin_import_path_var.set("No file selected")
        self._admin_import_name_var.set("")
        self.refresh_model_options()
        msg = f"Model '{name}' uploaded ({runtime})."
        for warning in (result or {}).get("warnings") or []:
            msg += f"\nWarning: {warning}"
        messagebox.showinfo("Upload", msg)

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
        default_name = f"{str(model.get('name') or 'model').strip().replace(' ', '_')}_export.zip"
        target = filedialog.asksaveasfilename(
            title="Export Model Archive",
            defaultextension=".zip",
            initialfile=default_name,
            filetypes=[("Model Archive", "*.zip")],
        )
        if not target:
            return
        try:
            data = self.api.export_model_archive(int(model_id))
            Path(target).write_bytes(data)
            messagebox.showinfo("Export", f"Model exported to:\n{target}")
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

