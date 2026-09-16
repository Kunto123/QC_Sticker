from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import Text, filedialog, messagebox, ttk

import customtkinter as ctk

import cv2
import numpy as np

from client_tk.app.components.roi_picker_canvas import RoiPickerCanvas
from client_tk.app.components.scrollable_frame import AutoHideScrollbar, ScrollableFrame
from client_tk.app.theme import APP_BG, ACCENT, BORDER, INPUT_BG, PANEL_ALT_BG, PANEL_BG, TEXT_ON_ACCENT, TEXT_PRIMARY, TEXT_SECONDARY

ACCENT_HOVER = "#1d4ed8"


_MODEL_FILE_EXTENSIONS = {".pt", ".onnx", ".engine", ".bin"}


def _float_or_none(value: str) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    return float(raw)

def _int_or_none(value: str) -> int | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    return int(float(raw))


class JsonEditor(ctk.CTkFrame):
    def __init__(
        self,
        master,
        title: str,
        initial_payload: dict | None = None,
        *,
        text_height: int = 18,
        text_width: int = 60,
    ):
        super().__init__(master, fg_color=PANEL_BG, corner_radius=14, border_width=1, border_color=BORDER)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        ctk.CTkLabel(self, text=title, font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=(10, 6),
        )

        shell = ctk.CTkFrame(self, fg_color="transparent")
        shell.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)

        self.text = Text(
            shell,
            height=max(4, int(text_height)),
            width=max(20, int(text_width)),
            wrap="none",
            undo=True,
            font=("Consolas", 10),
            padx=8,
            pady=8,
            borderwidth=0,
            background=INPUT_BG,
            foreground=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            selectbackground=ACCENT,
            selectforeground=TEXT_ON_ACCENT,
        )
        y_scroll = AutoHideScrollbar(shell, orient="vertical", command=self.text.yview)
        x_scroll = AutoHideScrollbar(shell, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        self.text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        if initial_payload is not None:
            self.set_payload(initial_payload)

    def set_payload(self, payload: dict) -> None:
        self.text.delete("1.0", "end")
        self.text.insert("1.0", json.dumps(payload, ensure_ascii=True, indent=2))

    def get_payload(self) -> dict:
        raw = self.text.get("1.0", "end").strip()
        if not raw:
            return {}
        return json.loads(raw)


class LabeledValuePanel(ctk.CTkFrame):
    def __init__(self, master, title: str, fields: list[tuple[str, str]], *, columns: int = 1):
        super().__init__(master, fg_color=PANEL_BG, corner_radius=14, border_width=1, border_color=BORDER)
        self._labels: dict[str, ctk.CTkLabel] = {}
        self._n_columns = max(1, columns)
        columns = self._n_columns
        ctk.CTkLabel(self, text=title, font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).pack(anchor="w", padx=10, pady=(10, 6))

        self._fields_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._fields_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        for col in range(columns * 2):
            self._fields_frame.columnconfigure(col, weight=1 if col % 2 else 0)
        for index, (key, label) in enumerate(fields):
            row = index // columns
            col = (index % columns) * 2
            ctk.CTkLabel(self._fields_frame, text=f"{label}:", font=("Segoe UI", 9, "bold"), text_color=TEXT_PRIMARY).grid(
                row=row,
                column=col,
                sticky="w",
                padx=(0, 8),
                pady=3,
            )
            value = ctk.CTkLabel(self._fields_frame, text="-", wraplength=300, justify="left", text_color=TEXT_SECONDARY)
            value.grid(row=row, column=col + 1, sticky="ew", pady=3)
            self._labels[key] = value

        # Dynamically recompute wraplength when panel is resized
        self.bind("<Configure>", self._on_resize, add="+")

    def _on_resize(self, event) -> None:
        total = event.width - 20  # rough inner padding
        if total < 80:
            return
        # Estimate: label columns are ~85 px each; remaining split across value cols
        per_value = max(80, (total - 85 * self._n_columns) // self._n_columns - 8)
        for widget in self._labels.values():
            widget.configure(wraplength=per_value)

    def set_values(self, mapping: dict[str, object]) -> None:
        for key, widget in self._labels.items():
            widget.configure(text=str(mapping.get(key, "-")))

    def reset(self) -> None:
        for widget in self._labels.values():
            widget.configure(text="-")


class StatCard(ctk.CTkFrame):
    def __init__(self, master, title: str, *, background: str, foreground: str):
        super().__init__(master, fg_color=background, corner_radius=14, border_width=1, border_color=BORDER)
        shell = ctk.CTkFrame(self, fg_color="transparent")
        shell.pack(fill="both", expand=True, padx=14, pady=12)
        ctk.CTkLabel(shell, text=title, text_color=foreground, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.value_label = ctk.CTkLabel(shell, text="0", text_color=foreground, font=("Segoe UI", 22, "bold"))
        self.value_label.pack(anchor="w", pady=(6, 0))
        self.note_label = ctk.CTkLabel(shell, text="", text_color=foreground, font=("Segoe UI", 8))
        self.note_label.pack(anchor="w")

    def set_value(self, value: object, note: str = "") -> None:
        self.value_label.configure(text=str(value))
        self.note_label.configure(text=note)


class TemplateEditorForm(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color=APP_BG, corner_radius=0)
        self.columnconfigure(0, weight=1)

        self._model_lookup: dict[str, dict] = {}
        self._model_path_lookup: dict[str, dict] = {}
        self._profile_lookup: dict[str, dict] = {}

        self.name_var = tk.StringVar()
        self.description_var = tk.StringVar()
        self.is_active_var = tk.BooleanVar(value=True)

        self.camera_index_var = tk.StringVar(value="0")
        self.camera_width_var = tk.StringVar()
        self.camera_height_var = tk.StringVar()
        self.camera_fps_var = tk.StringVar()

        self.part_ready_roi_x_var = tk.StringVar(value="0.2")
        self.part_ready_roi_y_var = tk.StringVar(value="0.2")
        self.part_ready_roi_w_var = tk.StringVar(value="0.25")
        self.part_ready_roi_h_var = tk.StringVar(value="0.25")

        self.sticker_roi_x_var = tk.StringVar(value="0.2")
        self.sticker_roi_y_var = tk.StringVar(value="0.2")
        self.sticker_roi_w_var = tk.StringVar(value="0.6")
        self.sticker_roi_h_var = tk.StringVar(value="0.6")

        self.part_ready_enabled_var = tk.BooleanVar(value=True)
        self.part_ready_profile_choice = tk.StringVar()
        self.part_ready_profile_id_var = tk.StringVar()
        self.part_ready_colorspace_var = tk.StringVar(value="LAB")
        self.part_ready_distance_var = tk.StringVar()
        self.part_ready_ratio_var = tk.StringVar(value="0.75")

        self.model_choice_var = tk.StringVar()
        self.model_path_var = tk.StringVar()
        self.model_meta_path_var = tk.StringVar()
        self.model_runtime_var = tk.StringVar(value="ultralytics")
        self.model_conf_threshold_var = tk.StringVar(value="0.25")
        self.model_stream_fps_var = tk.StringVar(value="10")
        self.model_inference_fps_var = tk.StringVar(value="4")
        self.model_imgsz_var = tk.StringVar(value="640")
        self.model_classes_var = tk.StringVar()
        self.text_anchor_class_var = tk.StringVar(value="text_anchor")
        self.center_dot_class_var = tk.StringVar(value="center_dot")
        self.anchor_crop_padding_var = tk.StringVar(value="0.08")
        self.anchor_crop_scale_var = tk.StringVar(value="2.0")

        self.sticker_enabled_var = tk.BooleanVar(value=True)
        self.sticker_part_name_var = tk.StringVar()
        self.sticker_expected_class_var = tk.StringVar()
        self.sticker_validator_mode_var = tk.StringVar(value="ml_detection")
        self.sticker_min_roi_conf_var = tk.StringVar(value="0.0")
        self.sticker_min_class_conf_var = tk.StringVar()
        self.sticker_max_offset_x_var = tk.StringVar(value="80")
        self.sticker_max_offset_y_var = tk.StringVar(value="80")
        self.sticker_expected_center_x_var = tk.StringVar(value="")
        self.sticker_expected_center_y_var = tk.StringVar(value="")
        self.sticker_commit_stable_frames_var = tk.StringVar(value="5")
        self.sticker_settle_ms_var = tk.StringVar(value="")
        self.sticker_tilt_gate_enabled_var = tk.BooleanVar(value=False)
        self.sticker_expected_tilt_var = tk.StringVar(value="0.0")
        self.sticker_max_tilt_var = tk.StringVar(value="")
        self._api_client_ref = None  # set from outside for "Load from Session"

        self.write_to_db_var = tk.BooleanVar(value=True)

        header = ctk.CTkFrame(self, fg_color=PANEL_BG, corner_radius=14, border_width=1, border_color=BORDER)
        header.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 0))
        header.columnconfigure(1, weight=1)
        header.columnconfigure(3, weight=1)
        ctk.CTkLabel(header, text="Template Identity", font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).grid(
            row=0,
            column=0,
            columnspan=4,
            sticky="w",
            padx=10,
            pady=(10, 6),
        )
        self._entry(header, 1, 0, "Name", self.name_var)
        self._entry(header, 1, 2, "Description", self.description_var)
        ctk.CTkCheckBox(header, text="Active", variable=self.is_active_var, text_color=TEXT_PRIMARY).grid(row=2, column=0, sticky="w", pady=(8, 10), padx=10)

        notebook = ctk.CTkTabview(self)
        notebook.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.rowconfigure(1, weight=1)

        for tab_name in ("Camera", "Part Ready", "Sticker", "Vision", "Persistence", "Metadata"):
            notebook.add(tab_name)

        camera_tab = notebook.tab("Camera")
        part_ready_tab = notebook.tab("Part Ready")
        _sticker_tab_outer = notebook.tab("Sticker")
        _sticker_scroller = ScrollableFrame(_sticker_tab_outer)
        _sticker_scroller.pack(fill="both", expand=True)
        sticker_tab = ctk.CTkFrame(_sticker_scroller.body, fg_color=APP_BG)
        sticker_tab.pack(fill="both", expand=True)
        sticker_tab.columnconfigure(0, weight=1)
        vision_tab = notebook.tab("Vision")
        persistence_tab = notebook.tab("Persistence")
        metadata_tab = notebook.tab("Metadata")

        camera_tab.columnconfigure(1, weight=1)
        camera_tab.columnconfigure(3, weight=1)
        self._entry(camera_tab, 0, 0, "Camera Index", self.camera_index_var)
        self._entry(camera_tab, 0, 2, "FPS", self.camera_fps_var)
        self._entry(camera_tab, 1, 0, "Width", self.camera_width_var)
        self._entry(camera_tab, 1, 2, "Height", self.camera_height_var)

        self._build_roi_section(part_ready_tab, "Part Ready ROI", 0, self.part_ready_roi_x_var, self.part_ready_roi_y_var, self.part_ready_roi_w_var, self.part_ready_roi_h_var)
        self._build_roi_section(sticker_tab, "Sticker ROI", 0, self.sticker_roi_x_var, self.sticker_roi_y_var, self.sticker_roi_w_var, self.sticker_roi_h_var)

        part_ready_config = ctk.CTkFrame(part_ready_tab, fg_color=PANEL_BG, corner_radius=14, border_width=1, border_color=BORDER)
        part_ready_config.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        part_ready_config.columnconfigure(1, weight=1)
        part_ready_config.columnconfigure(3, weight=1)
        ctk.CTkLabel(part_ready_config, text="Gate Config", font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).grid(
            row=0,
            column=0,
            columnspan=4,
            sticky="w",
            padx=10,
            pady=(10, 6),
        )
        ctk.CTkCheckBox(part_ready_config, text="Enable Part Ready Gate", variable=self.part_ready_enabled_var, text_color=TEXT_PRIMARY).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(0, 8),
            padx=10,
        )
        ctk.CTkLabel(part_ready_config, text="Color Profile", text_color=TEXT_PRIMARY).grid(row=2, column=0, sticky="w", padx=(10, 8), pady=4)
        self.profile_selector = ttk.Combobox(part_ready_config, textvariable=self.part_ready_profile_choice, state="readonly")
        self.profile_selector.grid(row=2, column=1, sticky="ew", pady=4)
        self.profile_selector.bind("<<ComboboxSelected>>", self._on_profile_selected)
        self._entry(part_ready_config, 2, 2, "Profile ID", self.part_ready_profile_id_var)
        self._entry(part_ready_config, 3, 0, "Colorspace", self.part_ready_colorspace_var)
        self._entry(part_ready_config, 3, 2, "Distance Threshold", self.part_ready_distance_var)
        self._entry(part_ready_config, 4, 0, "Min Match Ratio", self.part_ready_ratio_var)

        sticker_config = ctk.CTkFrame(sticker_tab, fg_color=PANEL_BG, corner_radius=14, border_width=1, border_color=BORDER)
        sticker_config.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        sticker_config.columnconfigure(1, weight=1)
        sticker_config.columnconfigure(3, weight=1)
        ctk.CTkLabel(sticker_config, text="Sticker Rule", font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).grid(
            row=0,
            column=0,
            columnspan=4,
            sticky="w",
            padx=10,
            pady=(10, 6),
        )
        ctk.CTkCheckBox(sticker_config, text="Enable Sticker Validation", variable=self.sticker_enabled_var, text_color=TEXT_PRIMARY).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(0, 8),
            padx=10,
        )
        self._entry(sticker_config, 2, 0, "Part Name", self.sticker_part_name_var)
        self._entry(sticker_config, 2, 2, "Expected Class", self.sticker_expected_class_var)
        self._entry(sticker_config, 3, 0, "Validator Mode", self.sticker_validator_mode_var)
        self._entry(sticker_config, 3, 2, "Min ROI Conf", self.sticker_min_roi_conf_var)
        self._entry(sticker_config, 5, 0, "Min Class Conf", self.sticker_min_class_conf_var)
        self._entry(sticker_config, 5, 2, "Max Offset X", self.sticker_max_offset_x_var)
        self._entry(sticker_config, 6, 0, "Max Offset Y", self.sticker_max_offset_y_var)
        self._entry(sticker_config, 6, 2, "Expected Center X (0-1)", self.sticker_expected_center_x_var)
        self._entry(sticker_config, 7, 0, "Expected Center Y (0-1)", self.sticker_expected_center_y_var)
        self._entry(sticker_config, 7, 2, "Stable Frames (legacy)", self.sticker_commit_stable_frames_var)
        self._entry(sticker_config, 8, 0, "Part Ready Settle (ms)", self.sticker_settle_ms_var)
        ctk.CTkLabel(sticker_config, text="Settle (ms): nilai utama — mengontrol hold inferensi dan commit. 0 = nonaktif. Kosong = ikuti env.", text_color=TEXT_SECONDARY, font=("Segoe UI", 8)).grid(
            row=9, column=0, columnspan=2, sticky="w", padx=(10, 8))
        ctk.CTkLabel(sticker_config, text="Stable Frames: legacy, tidak memengaruhi runtime. Gunakan Part Ready Settle (ms) sebagai gantinya.", text_color=TEXT_SECONDARY, font=("Segoe UI", 8)).grid(
            row=9, column=2, columnspan=2, sticky="w", padx=(0, 8))
        ctk.CTkLabel(sticker_config, text="Kosong = auto center (0.5). Gunakan Visual Picker di bawah.", text_color=TEXT_SECONDARY, font=("Segoe UI", 8)).grid(
            row=10, column=0, columnspan=4, sticky="w", pady=(0, 4), padx=10
        )

        # Tilt gate section
        tilt_separator = ctk.CTkFrame(sticker_config, fg_color=BORDER, height=1)
        tilt_separator.grid(row=11, column=0, columnspan=4, sticky="ew", padx=10, pady=(6, 4))
        ctk.CTkLabel(sticker_config, text="Tilt Gate", font=("Segoe UI", 9, "bold"), text_color=TEXT_PRIMARY).grid(
            row=12, column=0, columnspan=4, sticky="w", padx=10, pady=(0, 4))
        self.tilt_gate_checkbox = ctk.CTkCheckBox(
            sticker_config,
            text="Aktifkan Gate Kemiringan (OUT_OF_ANGLE)",
            variable=self.sticker_tilt_gate_enabled_var,
            text_color=TEXT_PRIMARY,
            command=self._on_tilt_gate_toggled,
        )
        self.tilt_gate_checkbox.grid(row=13, column=0, columnspan=4, sticky="w", padx=10, pady=(0, 6))
        self._entry(sticker_config, 14, 0, "Expected Tilt (deg)", self.sticker_expected_tilt_var)
        self._entry(sticker_config, 14, 2, "Max Tilt Deviation (deg)", self.sticker_max_tilt_var)
        self.tilt_note_label = ctk.CTkLabel(
            sticker_config,
            text="Gate nonaktif — nilai tersimpan sebagai telemetry, tidak memengaruhi accept/reject.",
            text_color=TEXT_SECONDARY,
            font=("Segoe UI", 8),
        )
        self.tilt_note_label.grid(row=15, column=0, columnspan=4, sticky="w", padx=10, pady=(0, 8))
        self._tilt_entries: list[ctk.CTkEntry] = []
        self.sticker_tilt_gate_enabled_var.trace_add("write", lambda *_: self._on_tilt_gate_toggled())

        # Visual ROI Picker
        self.roi_picker = RoiPickerCanvas(sticker_tab, "Visual ROI & Expected Center Picker", size=(640, 300))
        self.roi_picker.grid(row=2, column=0, sticky="nsew", pady=(10, 0))

        picker_actions = ctk.CTkFrame(sticker_tab, fg_color="transparent")
        picker_actions.grid(row=3, column=0, sticky="w", pady=(4, 0))
        ctk.CTkButton(picker_actions, text="Load Image", command=self._picker_load_image, fg_color=ACCENT, hover_color="#1d4ed8", text_color=TEXT_ON_ACCENT).pack(
            side="left",
            padx=(0, 6),
        )
        ctk.CTkButton(picker_actions, text="Load from Session", command=self._picker_load_session, fg_color=ACCENT, hover_color="#1d4ed8", text_color=TEXT_ON_ACCENT).pack(
            side="left",
            padx=(0, 6),
        )
        ctk.CTkButton(picker_actions, text="Clear", command=self.roi_picker.clear, fg_color=PANEL_ALT_BG, hover_color="#1f3b57", text_color=TEXT_PRIMARY).pack(side="left")

        self.roi_picker.on_center_changed = self._on_picker_center_changed

        for var in (self.sticker_roi_x_var, self.sticker_roi_y_var, self.sticker_roi_w_var, self.sticker_roi_h_var,
                    self.part_ready_roi_x_var, self.part_ready_roi_y_var, self.part_ready_roi_w_var, self.part_ready_roi_h_var,
                    self.sticker_expected_center_x_var, self.sticker_expected_center_y_var):
            var.trace_add("write", lambda *_: self._sync_picker())

        vision_tab.columnconfigure(1, weight=1)
        vision_tab.columnconfigure(3, weight=1)
        ctk.CTkLabel(vision_tab, text="Registered Model", text_color=TEXT_PRIMARY).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.model_selector = ttk.Combobox(vision_tab, textvariable=self.model_choice_var, state="readonly")
        self.model_selector.grid(row=0, column=1, columnspan=3, sticky="ew", pady=4)
        self.model_selector.bind("<<ComboboxSelected>>", self._on_model_selected)
        ctk.CTkLabel(vision_tab, text="Model Path", text_color=TEXT_PRIMARY).grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        self.model_path_selector = ttk.Combobox(vision_tab, textvariable=self.model_path_var, state="readonly")
        self.model_path_selector.grid(row=1, column=1, sticky="ew", padx=(0, 12), pady=4)
        self.model_path_selector.bind("<<ComboboxSelected>>", self._on_model_path_selected)
        self._entry(vision_tab, 1, 2, "Meta Path", self.model_meta_path_var)
        self._entry(vision_tab, 2, 0, "Runtime", self.model_runtime_var)
        self._entry(vision_tab, 2, 2, "Conf Threshold", self.model_conf_threshold_var)
        self._entry(vision_tab, 3, 0, "Stream FPS", self.model_stream_fps_var)
        self._entry(vision_tab, 3, 2, "Inference FPS", self.model_inference_fps_var)
        self._entry(vision_tab, 4, 0, "Image Size", self.model_imgsz_var)
        self._entry(vision_tab, 4, 2, "Classes CSV", self.model_classes_var)

        persistence_tab.columnconfigure(0, weight=1)
        ctk.CTkCheckBox(persistence_tab, text="Write committed result to DB", variable=self.write_to_db_var, text_color=TEXT_PRIMARY).grid(
            row=0,
            column=0,
            sticky="w",
        )
        ctk.CTkLabel(
            persistence_tab,
            text="Jika dimatikan, event tetap dihitung di session counter tetapi tidak dipersist ke inspection results repository.",
            wraplength=520,
            justify="left",
            text_color=TEXT_SECONDARY,
        ).grid(row=1, column=0, sticky="w", pady=(10, 0))

        self.metadata_editor = Text(metadata_tab, height=10, width=40)
        self.metadata_editor.pack(fill="both", expand=True)
        self.metadata_editor.insert("1.0", "{}")

    # ------------------------------------------------------------------
    # Picker helpers
    # ------------------------------------------------------------------

    def _sync_picker(self) -> None:
        try:
            pr = {
                "x": float(self.part_ready_roi_x_var.get() or 0),
                "y": float(self.part_ready_roi_y_var.get() or 0),
                "w": float(self.part_ready_roi_w_var.get() or 1),
                "h": float(self.part_ready_roi_h_var.get() or 1),
            }
            sr = {
                "x": float(self.sticker_roi_x_var.get() or 0),
                "y": float(self.sticker_roi_y_var.get() or 0),
                "w": float(self.sticker_roi_w_var.get() or 1),
                "h": float(self.sticker_roi_h_var.get() or 1),
            }
        except ValueError:
            return
        cx_raw = self.sticker_expected_center_x_var.get().strip()
        cy_raw = self.sticker_expected_center_y_var.get().strip()
        try:
            cx = float(cx_raw) if cx_raw else None
            cy = float(cy_raw) if cy_raw else None
        except ValueError:
            cx = cy = None
        self.roi_picker.set_rois(part_ready_roi=pr, sticker_roi=sr)
        self.roi_picker.set_expected_center(cx if cx is not None else 0.5, cy if cy is not None else 0.5)

    def _on_picker_center_changed(self, cx: float, cy: float) -> None:
        self.sticker_expected_center_x_var.set(str(round(cx, 4)))
        self.sticker_expected_center_y_var.set(str(round(cy, 4)))

    def _picker_load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Pilih Gambar Referensi",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Load Image", f"Gagal membaca gambar: {path}")
            return
        self.roi_picker.load_image(frame)
        self._sync_picker()

    def _picker_load_session(self) -> None:
        if self._api_client_ref is None:
            messagebox.showinfo("Load from Session", "API client belum terhubung.")
            return
        try:
            result = self._api_client_ref.get_latest_preview()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Load from Session", f"Gagal mengambil preview: {exc}")
            return
        b64 = result.get("overlay_image_b64") if isinstance(result, dict) else None
        if not b64:
            messagebox.showinfo("Load from Session", "Tidak ada session aktif atau belum ada frame.")
            return
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            messagebox.showerror("Load from Session", "Gagal mendekode gambar dari session.")
            return
        self.roi_picker.load_image(frame)
        self._sync_picker()

    # ------------------------------------------------------------------

    def _on_tilt_gate_toggled(self) -> None:
        enabled = self.sticker_tilt_gate_enabled_var.get()
        note = (
            "Gate aktif — maks deviasi dipakai untuk reject OUT_OF_ANGLE."
            if enabled
            else "Gate nonaktif — nilai tersimpan sebagai telemetry, tidak memengaruhi accept/reject."
        )
        if hasattr(self, "tilt_note_label"):
            self.tilt_note_label.configure(text=note)

    def _entry(self, master, row: int, column: int, label: str, variable: tk.Variable) -> None:
        ctk.CTkLabel(master, text=label, text_color=TEXT_PRIMARY).grid(row=row, column=column, sticky="w", padx=(0, 8), pady=4)
        ctk.CTkEntry(master, textvariable=variable, fg_color=INPUT_BG, border_color=BORDER, text_color=TEXT_PRIMARY).grid(
            row=row,
            column=column + 1,
            sticky="ew",
            padx=(0, 12),
            pady=4,
        )

    def _build_roi_section(
        self,
        master,
        title: str,
        row: int,
        x_var: tk.StringVar,
        y_var: tk.StringVar,
        w_var: tk.StringVar,
        h_var: tk.StringVar,
    ) -> None:
        frame = ctk.CTkFrame(master, fg_color=PANEL_BG, corner_radius=14, border_width=1, border_color=BORDER)
        frame.grid(row=row, column=0, sticky="ew")
        for index in range(8):
            frame.columnconfigure(index, weight=1)
        ctk.CTkLabel(frame, text=title, font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).grid(row=0, column=0, columnspan=8, sticky="w", padx=10, pady=(10, 6))
        self._roi_entry(frame, 0, "x", x_var)
        self._roi_entry(frame, 2, "y", y_var)
        self._roi_entry(frame, 4, "w", w_var)
        self._roi_entry(frame, 6, "h", h_var)

    def _roi_entry(self, master, column: int, label: str, variable: tk.StringVar) -> None:
        ctk.CTkLabel(master, text=label, text_color=TEXT_PRIMARY).grid(row=1, column=column, sticky="w", padx=4, pady=4)
        ctk.CTkEntry(master, textvariable=variable, width=10, fg_color=INPUT_BG, border_color=BORDER, text_color=TEXT_PRIMARY).grid(
            row=1,
            column=column + 1,
            sticky="ew",
            padx=4,
            pady=4,
        )

    def _detect_model_file_options(self) -> list[dict]:
        data_root = Path(os.getenv("QC_SUITE_DATA_ROOT", Path(__file__).resolve().parents[3] / "data")).resolve()
        models_dir = data_root / "models"
        if not models_dir.exists():
            return []

        detected: list[dict] = []
        for model_file in sorted(models_dir.rglob("*")):
            if not model_file.is_file() or model_file.suffix.lower() not in _MODEL_FILE_EXTENSIONS:
                continue
            relative_name = str(model_file.relative_to(models_dir)).replace("/", "\\")
            meta_candidate = model_file.with_suffix(".meta.json")
            meta_path = str(meta_candidate) if meta_candidate.exists() else ""
            detected.append(
                {
                    "id": f"detected:{relative_name}",
                    "name": f"Detected {model_file.name}",
                    "path": str(model_file),
                    "meta_path": meta_path,
                    "runtime": "ultralytics",
                    "class_names": [],
                    "source": "detected",
                }
            )
        return detected

    def set_model_options(self, models: list[dict]) -> None:
        self._model_lookup = {}
        self._model_path_lookup = {}
        values: list[str] = []
        combined_models: list[dict] = []
        seen_model_keys: set[tuple[str, str]] = set()
        for source_item in list(models):
            item = dict(source_item)
            model_id = str(item.get("id") or "").strip().lower()
            model_path = str(item.get("path") or "").strip().lower()
            dedupe_key = (model_id, model_path)
            if dedupe_key in seen_model_keys:
                continue
            seen_model_keys.add(dedupe_key)
            combined_models.append(item)

        for item in combined_models:
            label = f"{item.get('id')} | {item.get('name')} | {item.get('runtime') or '-'}"
            self._model_lookup[label] = item
            values.append(label)
            path_key = str(item.get("path") or "").strip()
            if path_key:
                if path_key not in self._model_path_lookup:
                    self._model_path_lookup[path_key] = dict(item)
                    self._model_path_lookup[path_key]["_selector_label"] = label
                elif not str(self._model_path_lookup[path_key].get("meta_path") or "").strip():
                    meta = str(item.get("meta_path") or "").strip()
                    if meta:
                        self._model_path_lookup[path_key]["meta_path"] = meta
        self.model_selector.configure(values=values)

        path_values = sorted(self._model_path_lookup.keys(), key=lambda item: item.lower())
        self.model_path_selector.configure(values=path_values)

        current_path = self.model_path_var.get().strip().lower()
        for label, item in self._model_lookup.items():
            if str(item.get("path") or "").strip().lower() == current_path:
                self.model_choice_var.set(label)
                break

    def _sync_model_selection(self) -> None:
        """Re-select the correct dropdown entry based on the current model_path_var.

        Called from set_payload so that loading a template re-highlights the right
        registered model without rebuilding the entire dropdown from a stale cache.
        """
        current_path = self.model_path_var.get().strip().lower()
        if not current_path:
            return
        for label, item in self._model_lookup.items():
            if str(item.get("path") or "").strip().lower() == current_path:
                self.model_choice_var.set(label)
                return

    def set_profile_options(self, profiles: list[dict]) -> None:
        self._profile_lookup = {}
        values: list[str] = []
        for item in profiles:
            profile = item.get("profile") or {}
            colorspace = profile.get("colorspace") or "LAB"
            label = f"{item.get('id')} | {item.get('name')} | {colorspace}"
            self._profile_lookup[label] = item
            values.append(label)
        self.profile_selector.configure(values=values)
        current_id = self.part_ready_profile_id_var.get().strip()
        for label, item in self._profile_lookup.items():
            if str(item.get("id")) == current_id:
                self.part_ready_profile_choice.set(label)
                break

    def _sync_profile_selection(self) -> None:
        """Re-select the correct profile dropdown entry based on part_ready_profile_id_var.

        Called from set_payload to restore the selection without repopulating options
        from a stale cache.
        """
        current_id = self.part_ready_profile_id_var.get().strip()
        if not current_id:
            return
        for label, item in self._profile_lookup.items():
            if str(item.get("id")) == current_id:
                self.part_ready_profile_choice.set(label)
                return

    def _on_model_selected(self, _event=None) -> None:
        item = self._model_lookup.get(self.model_choice_var.get().strip())
        if not item:
            return
        self.model_path_var.set(str(item.get("path") or ""))
        self.model_meta_path_var.set(str(item.get("meta_path") or ""))
        self.model_runtime_var.set(str(item.get("runtime") or "ultralytics"))
        class_names = item.get("class_names") or []
        if class_names:
            self.model_classes_var.set(",".join(str(name) for name in class_names))

    def _on_model_path_selected(self, _event=None) -> None:
        item = self._model_path_lookup.get(self.model_path_var.get().strip())
        if not item:
            return
        meta_path = str(item.get("meta_path") or "").strip()
        if meta_path:
            self.model_meta_path_var.set(meta_path)
        self.model_runtime_var.set(str(item.get("runtime") or "ultralytics"))
        class_names = item.get("class_names") or []
        if class_names:
            self.model_classes_var.set(",".join(str(name) for name in class_names))
        selector_label = str(item.get("_selector_label") or "").strip()
        if selector_label:
            self.model_choice_var.set(selector_label)

    def _on_profile_selected(self, _event=None) -> None:
        item = self._profile_lookup.get(self.part_ready_profile_choice.get().strip())
        if not item:
            return
        self.part_ready_profile_id_var.set(str(item.get("id") or ""))
        profile = item.get("profile") or {}
        self.part_ready_colorspace_var.set(str(profile.get("colorspace") or "LAB"))
        tolerance = profile.get("tolerance") or {}
        distance = tolerance.get("distance_threshold")
        if distance is not None:
            self.part_ready_distance_var.set(str(distance))
        match_ratio = profile.get("min_match_ratio")
        if match_ratio is not None:
            self.part_ready_ratio_var.set(str(match_ratio))

    def set_payload(self, payload: dict) -> None:
        self.name_var.set(str(payload.get("name") or ""))
        self.description_var.set(str(payload.get("description") or ""))
        self.is_active_var.set(bool(payload.get("is_active", True)))

        camera = payload.get("camera") or {}
        self.camera_index_var.set(str(camera.get("camera_index") if camera.get("camera_index") is not None else 0))
        self.camera_width_var.set("" if camera.get("width") is None else str(camera.get("width")))
        self.camera_height_var.set("" if camera.get("height") is None else str(camera.get("height")))
        self.camera_fps_var.set("" if camera.get("fps") is None else str(camera.get("fps")))

        part_ready_roi = payload.get("part_ready_roi") or {}
        self.part_ready_roi_x_var.set(str(part_ready_roi.get("x", 0.2)))
        self.part_ready_roi_y_var.set(str(part_ready_roi.get("y", 0.2)))
        self.part_ready_roi_w_var.set(str(part_ready_roi.get("w", 0.25)))
        self.part_ready_roi_h_var.set(str(part_ready_roi.get("h", 0.25)))

        sticker_roi = payload.get("sticker_roi") or payload.get("roi") or {}
        self.sticker_roi_x_var.set(str(sticker_roi.get("x", 0.2)))
        self.sticker_roi_y_var.set(str(sticker_roi.get("y", 0.2)))
        self.sticker_roi_w_var.set(str(sticker_roi.get("w", 0.6)))
        self.sticker_roi_h_var.set(str(sticker_roi.get("h", 0.6)))

        part_ready = payload.get("part_ready") or {}
        self.part_ready_enabled_var.set(bool(part_ready.get("enabled", True)))
        self.part_ready_profile_id_var.set("" if part_ready.get("color_profile_id") is None else str(part_ready.get("color_profile_id")))
        self.part_ready_colorspace_var.set(str(part_ready.get("colorspace") or "LAB"))
        self.part_ready_distance_var.set("" if part_ready.get("distance_threshold") is None else str(part_ready.get("distance_threshold")))
        self.part_ready_ratio_var.set("" if part_ready.get("min_match_ratio") is None else str(part_ready.get("min_match_ratio")))

        vision = payload.get("vision") or {}
        self.model_path_var.set(str(vision.get("model_path") or ""))
        self.model_meta_path_var.set(str(vision.get("model_meta_path") or ""))
        self.model_runtime_var.set(str(vision.get("runtime") or "ultralytics"))
        self.model_conf_threshold_var.set(str(vision.get("conf_threshold", 0.25)))
        self.model_stream_fps_var.set(str(vision.get("stream_fps", 10)))
        self.model_inference_fps_var.set(str(vision.get("inference_fps", 4)))
        self.model_imgsz_var.set(str(vision.get("imgsz", 640)))
        self.model_classes_var.set(",".join(str(item) for item in (vision.get("classes") or [])))
        self.text_anchor_class_var.set(str(vision.get("text_anchor_class") or "text_anchor"))
        self.center_dot_class_var.set(str(vision.get("center_dot_class") or "center_dot"))
        self.anchor_crop_padding_var.set(str(vision.get("anchor_crop_padding_ratio", 0.08)))
        self.anchor_crop_scale_var.set(str(vision.get("anchor_crop_scale", 2.0)))

        sticker = payload.get("sticker") or {}
        self.sticker_enabled_var.set(bool(sticker.get("enabled", True)))
        self.sticker_part_name_var.set(str(sticker.get("part_name") or ""))
        self.sticker_expected_class_var.set(str(sticker.get("expected_class") or ""))
        self.sticker_validator_mode_var.set(str(sticker.get("validator_mode") or "ml_detection"))
        self.sticker_min_roi_conf_var.set(str(sticker.get("min_roi_confidence", 0.0)))
        self.sticker_min_class_conf_var.set("" if sticker.get("min_class_confidence") is None else str(sticker.get("min_class_confidence")))
        self.sticker_max_offset_x_var.set("" if sticker.get("max_offset_x") is None else str(sticker.get("max_offset_x")))
        self.sticker_max_offset_y_var.set("" if sticker.get("max_offset_y") is None else str(sticker.get("max_offset_y")))
        self.sticker_expected_center_x_var.set("" if sticker.get("expected_center_x") is None else str(sticker.get("expected_center_x")))
        self.sticker_expected_center_y_var.set("" if sticker.get("expected_center_y") is None else str(sticker.get("expected_center_y")))
        self.sticker_commit_stable_frames_var.set(str(sticker.get("commit_stable_frames") or "5"))
        _settle = sticker.get("part_ready_settle_ms")
        self.sticker_settle_ms_var.set("" if _settle is None else str(_settle))
        self.sticker_tilt_gate_enabled_var.set(bool(sticker.get("tilt_gate_enabled", False)))
        self.sticker_expected_tilt_var.set(str(sticker.get("expected_tilt_degrees", 0.0)))
        _max_tilt = sticker.get("max_tilt_degrees")
        self.sticker_max_tilt_var.set("" if _max_tilt is None else str(_max_tilt))
        self.after_idle(self._sync_picker)
        self.after_idle(self._on_tilt_gate_toggled)

        persistence = payload.get("persistence") or {}
        self.write_to_db_var.set(bool(persistence.get("write_to_db", True)))

        metadata = payload.get("metadata") or {}
        self.metadata_editor.delete("1.0", "end")
        self.metadata_editor.insert("1.0", json.dumps(metadata, ensure_ascii=True, indent=2))

        self._sync_model_selection()
        self._sync_profile_selection()

    def get_payload(self) -> dict:
        metadata_raw = self.metadata_editor.get("1.0", "end").strip() or "{}"
        metadata = json.loads(metadata_raw)
        if not isinstance(metadata, dict):
            raise ValueError("Metadata harus berupa object JSON.")
        classes = [
            item.strip()
            for item in self.model_classes_var.get().split(",")
            if item.strip()
        ]
        return {
            "name": self.name_var.get().strip() or "Untitled Template",
            "description": self.description_var.get().strip(),
            "is_active": bool(self.is_active_var.get()),
            "camera": {
                "camera_index": _int_or_none(self.camera_index_var.get()) or 0,
                "width": _int_or_none(self.camera_width_var.get()),
                "height": _int_or_none(self.camera_height_var.get()),
                "fps": _float_or_none(self.camera_fps_var.get()),
            },
            "part_ready_roi": {
                "x": _float_or_none(self.part_ready_roi_x_var.get()) or 0.0,
                "y": _float_or_none(self.part_ready_roi_y_var.get()) or 0.0,
                "w": _float_or_none(self.part_ready_roi_w_var.get()) or 1.0,
                "h": _float_or_none(self.part_ready_roi_h_var.get()) or 1.0,
            },
            "sticker_roi": {
                "x": _float_or_none(self.sticker_roi_x_var.get()) or 0.0,
                "y": _float_or_none(self.sticker_roi_y_var.get()) or 0.0,
                "w": _float_or_none(self.sticker_roi_w_var.get()) or 1.0,
                "h": _float_or_none(self.sticker_roi_h_var.get()) or 1.0,
            },
            "vision": {
                "model_path": self.model_path_var.get().strip() or "models/dummy.pt",
                "model_meta_path": self.model_meta_path_var.get().strip() or None,
                "runtime": self.model_runtime_var.get().strip() or "ultralytics",
                "conf_threshold": _float_or_none(self.model_conf_threshold_var.get()) or 0.25,
                "stream_fps": _float_or_none(self.model_stream_fps_var.get()) or 10.0,
                "inference_fps": _float_or_none(self.model_inference_fps_var.get()) or 4.0,
                "imgsz": _int_or_none(self.model_imgsz_var.get()) or 640,
                "classes": classes,
                "text_anchor_class": self.text_anchor_class_var.get().strip() or "text_anchor",
                "center_dot_class": self.center_dot_class_var.get().strip() or "center_dot",
                "anchor_crop_padding_ratio": _float_or_none(self.anchor_crop_padding_var.get()) if _float_or_none(self.anchor_crop_padding_var.get()) is not None else 0.08,
                "anchor_crop_scale": _float_or_none(self.anchor_crop_scale_var.get()) or 2.0,
            },
            "part_ready": {
                "enabled": bool(self.part_ready_enabled_var.get()),
                "color_profile_id": _int_or_none(self.part_ready_profile_id_var.get()),
                "colorspace": self.part_ready_colorspace_var.get().strip() or "LAB",
                "distance_threshold": _float_or_none(self.part_ready_distance_var.get()),
                "min_match_ratio": _float_or_none(self.part_ready_ratio_var.get()),
            },
            "sticker": {
                "part_name": self.sticker_part_name_var.get().strip() or "Sample Part",
                "expected_class": self.sticker_expected_class_var.get().strip() or (classes[0] if classes else "sample-sticker"),
                "enabled": bool(self.sticker_enabled_var.get()),
                "validator_mode": self.sticker_validator_mode_var.get().strip() or "ml_detection",
                "min_roi_confidence": _float_or_none(self.sticker_min_roi_conf_var.get()) or 0.0,
                "min_class_confidence": _float_or_none(self.sticker_min_class_conf_var.get()),
                "max_offset_x": _float_or_none(self.sticker_max_offset_x_var.get()),
                "max_offset_y": _float_or_none(self.sticker_max_offset_y_var.get()),
                "expected_center_x": _float_or_none(self.sticker_expected_center_x_var.get()),
                "expected_center_y": _float_or_none(self.sticker_expected_center_y_var.get()),
                "part_ready_settle_ms": _int_or_none(self.sticker_settle_ms_var.get()),
                "dot_min_confidence": _float_or_none(self.sticker_dot_min_conf_var.get()),
                "expected_dot_x": _float_or_none(self.sticker_expected_dot_x_var.get()),
                "expected_dot_y": _float_or_none(self.sticker_expected_dot_y_var.get()),
                "max_anchor_offset_x": _float_or_none(self.sticker_max_anchor_offset_x_var.get()),
                "max_anchor_offset_y": _float_or_none(self.sticker_max_anchor_offset_y_var.get()),
                "commit_stable_frames": _int_or_none(self.sticker_commit_stable_frames_var.get()) or 5,
                "part_ready_settle_ms": _int_or_none(self.sticker_settle_ms_var.get()),
                "tilt_gate_enabled": bool(self.sticker_tilt_gate_enabled_var.get()),
                "expected_tilt_degrees": _float_or_none(self.sticker_expected_tilt_var.get()) or 0.0,
                "max_tilt_degrees": _float_or_none(self.sticker_max_tilt_var.get()),
            },
            "persistence": {
                "write_to_db": bool(self.write_to_db_var.get()),
            },
            "metadata": metadata,
        }

    def reset(self) -> None:
        self.set_payload(
            {
                "name": "",
                "description": "",
                "is_active": True,
                "camera": {"camera_index": 0, "width": 640, "height": 480, "fps": 15},
                "part_ready_roi": {"x": 0.2, "y": 0.2, "w": 0.25, "h": 0.25},
                "sticker_roi": {"x": 0.2, "y": 0.2, "w": 0.6, "h": 0.6},
                "vision": {
                    "model_path": "",
                    "model_meta_path": "",
                    "runtime": "ultralytics",
                    "conf_threshold": 0.25,
                    "stream_fps": 10,
                    "inference_fps": 4,
                    "imgsz": 640,
                    "classes": [],
                    "text_anchor_class": "text_anchor",
                    "center_dot_class": "center_dot",
                    "anchor_crop_padding_ratio": 0.08,
                    "anchor_crop_scale": 2.0,
                },
                "part_ready": {
                    "enabled": True,
                    "color_profile_id": None,
                    "colorspace": "LAB",
                    "distance_threshold": None,
                    "min_match_ratio": 0.75,
                },
                "sticker": {
                    "part_name": "",
                "expected_class": "",
                    "enabled": True,
                    "validator_mode": "ml_detection",
                    "min_roi_confidence": 0.0,
                    "min_class_confidence": None,
                    "max_offset_x": 80,
                    "max_offset_y": 80,
                    "part_ready_settle_ms": None,
                    "tilt_gate_enabled": False,
                    "expected_tilt_degrees": 0.0,
                    "max_tilt_degrees": None,
                },
                "persistence": {"write_to_db": True},
                "metadata": {},
            }
        )
