"""Tab Templates -- preset library, preset wizard, ROI picker, gap reference."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import customtkinter as ctk

from client_tk.app.components.roi_picker_canvas import RoiPickerCanvas
from client_tk.app.components.scrollable_frame import ScrollableFrame


def _float_or_default(value, default):
    """Parse nilai string ke float, return default kalau kosong atau tidak valid."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


from client_tk.app.theme import (
    ACCENT,
    ACCENT_HOVER,
    BORDER,
    PANEL_ALT_BG,
    PANEL_BG,
    TEXT_ON_ACCENT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class TemplatesTab:
    """Preset library + preset wizard yang diekstrak dari AdminScreen."""

    def __init__(self, admin, tab_frame):
        self.admin = admin
        self.frame = tab_frame
        self._build()

    # ------------------------------------------------------------------
    # Bangun
    def _build(self) -> None:
        a = self.admin
        self.frame.columnconfigure(0, weight=3)
        self.frame.columnconfigure(1, weight=2)
        self.frame.rowconfigure(0, weight=1)

        body = a._make_scrollable_body(self.frame, "Templates")

        a.presets_left = ttk.Frame(body, padding=8)
        a.presets_right = ttk.Frame(body, padding=8)
        a.presets_left.grid(row=0, column=0, sticky="nsew")
        a.presets_right.grid(row=0, column=1, sticky="nsew")

        self._build_library(a, a.presets_left)
        self._build_wizard(a, a.presets_right)

    def _build_library(self, a, parent) -> None:
        listing = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        listing.pack(fill="both", expand=True)

        listing.columnconfigure(0, weight=1)
        listing.rowconfigure(2, weight=1)

        ctk.CTkLabel(listing, text="Preset Library", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).grid(
            row=0, column=0, sticky="w", padx=12, pady=(10, 0),
        )
        ctk.CTkLabel(
            listing,
            text="Menampilkan template dan deployment aktif bersamaan; ACTIVE menandai record yang di-deploy.",
            text_color=TEXT_SECONDARY,
        ).grid(row=1, column=0, sticky="w", padx=12, pady=(2, 8))

        a.preset_table = a._build_table(
            listing,
            [
                ("id", "ID", 55, "center"),
                ("preset", "Preset", 220, "w"),
                ("version", "Version", 80, "center"),
                ("status", "Status", 90, "center"),
            ],
            row=2,
            height=18,
        )
        a.preset_table.bind("<<TreeviewSelect>>", a._on_preset_selected)

        footer = a._build_action_row(
            listing,
            [
                ("Refresh", a.refresh_presets, "neutral", "left"),
                ("Preset Baru", a.reset_preset_wizard, "neutral", "right"),
                ("Hapus/Nonaktifkan Terpilih", a.deactivate_selected_preset, "neutral", "right"),
            ],
        )
        footer.grid(row=3, column=0, sticky="ew", padx=12, pady=(8, 10))

    def _build_wizard(self, a, parent) -> None:
        wizard = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        wizard.pack(fill="both", expand=True)
        wizard.columnconfigure(1, weight=1)
        wizard.columnconfigure(3, weight=1)

        ctk.CTkLabel(wizard, text="Preset Wizard", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=12, pady=(10, 0),
        )
        ctk.CTkLabel(
            wizard,
            text="Isi cuma nilai yang kritis untuk produksi. Default teknis diterapkan otomatis.",
            text_color=TEXT_SECONDARY, wraplength=520, justify="left",
        ).grid(row=1, column=0, columnspan=4, sticky="w", padx=12, pady=(2, 10))

        a._entry(wizard, 3, 0, "Nama Preset", a.preset_name_var, columnspan=3)
        a._entry(wizard, 4, 0, "Deskripsi", a.preset_description_var, columnspan=3)

        a._entry(wizard, 5, 0, "Index Kamera", a.preset_camera_index_var, columnspan=1)

        ttk.Label(wizard, text="Model").grid(row=6, column=0, sticky="w", padx=(12, 8), pady=5)
        a.preset_model_selector = ttk.Combobox(wizard, textvariable=a.preset_model_choice_var, state="readonly")
        a.preset_model_selector.grid(row=6, column=1, columnspan=3, sticky="ew", padx=(0, 12), pady=5)
        a.preset_model_selector.bind("<<ComboboxSelected>>", a._on_preset_model_selected)

        ttk.Label(wizard, text="Runtime").grid(row=7, column=0, sticky="w", padx=(12, 8), pady=5)
        runtime_combo = ttk.Combobox(
            wizard,
            textvariable=a.preset_runtime_var,
            values=["auto", "ultralytics", "tflite", "onnx", "openvino"],
            width=14,
            state="readonly",
        )
        runtime_combo.grid(row=7, column=1, columnspan=3, sticky="w", padx=(0, 12), pady=5)
        a._entry(wizard, 8, 0, "Threshold Confidence", a.preset_conf_threshold_var, columnspan=3)

        # Variabel threshold Mean-Std (di-init lazy saat method pertama kali diubah)
        a.preset_mean_max_var = tk.StringVar(value="105.0")
        a.preset_std_max_var = tk.StringVar(value="35.0")
        a.preset_min_match_ratio_var = tk.StringVar(value="0.5")

        a._entry(wizard, 9, 0, "Expected Class", a.preset_expected_class_var, columnspan=3)

        a._entry(wizard, 11, 0, "Gap Threshold (0-1)", a.preset_gap_threshold_var, columnspan=2)
        # Lacak widget gap threshold untuk show/hide berdasarkan method (row 13 ditambah di bawah)
        a._gap_threshold_widgets = []
        for _r in (11,):
            try:
                for w in wizard.grid_slaves(row=_r):
                    a._gap_threshold_widgets.append(w)
            except Exception:
                pass

        # Selector method part-ready
        _pr_label = ttk.Label(wizard, text="Part Ready Method")
        _pr_label.grid(row=12, column=0, sticky="w", padx=(12, 8), pady=5)
        a.preset_part_ready_method_var = tk.StringVar(value="gap_template_match")
        method_combo = ttk.Combobox(
            wizard,
            textvariable=a.preset_part_ready_method_var,
            values=["gap_template_match", "mean_std_threshold"],
            width=20,
            state="readonly",
        )
        method_combo.grid(row=12, column=1, columnspan=2, sticky="w", padx=(0, 12), pady=5)
        a.preset_part_ready_method_var.trace_add("write", lambda *_: self._on_part_ready_method_changed(a))

        # Threshold edge Canny (khusus gap_template_match) — antara dropdown
        # method dan tombol capture reference. Kosong = auto-tuned (legacy).
        a.preset_canny_low_var = tk.StringVar(value="")
        a.preset_canny_high_var = tk.StringVar(value="")
        canny_row = ttk.Frame(wizard)
        canny_row.grid(row=13, column=0, columnspan=4, sticky="ew", padx=(12, 12), pady=5)
        ttk.Label(canny_row, text="Canny Lower").pack(side="left", padx=(0, 4))
        ttk.Entry(canny_row, textvariable=a.preset_canny_low_var, width=6).pack(side="left", padx=(0, 12))
        ttk.Label(canny_row, text="Canny Upper").pack(side="left", padx=(0, 4))
        ttk.Entry(canny_row, textvariable=a.preset_canny_high_var, width=6).pack(side="left", padx=(0, 12))
        ttk.Label(canny_row, text="(kosongkan = otomatis)", foreground="gray").pack(side="left")
        a._gap_threshold_widgets.append(canny_row)

        # Margin pencarian: seberapa jauh (fraksi dari w/h ROI itu sendiri) area
        # pencarian runtime membesar di sekitar part_ready_roi supaya part yang
        # geser/miring sedikit tetap match, bukan harus pixel-perfect. 0 = legacy.
        a.preset_gap_search_margin_var = tk.StringVar(value="0.0")
        margin_row = ttk.Frame(wizard)
        margin_row.grid(row=17, column=0, columnspan=4, sticky="ew", padx=(12, 12), pady=5)
        ttk.Label(margin_row, text="Margin Pencarian (0-1)").pack(side="left", padx=(0, 4))
        ttk.Entry(margin_row, textvariable=a.preset_gap_search_margin_var, width=6).pack(side="left", padx=(0, 12))
        ttk.Label(margin_row, text="(toleransi part geser/miring sedikit, 0 = harus pas persis)", foreground="gray").pack(side="left")
        a._gap_threshold_widgets.append(margin_row)

        # Field threshold Mean-Std (tampil cuma kalau method=mean_std_threshold)
        a._entry(wizard, 14, 0, "MEAN_MAX", a.preset_mean_max_var, columnspan=2)
        a._entry(wizard, 15, 0, "STD_MAX", a.preset_std_max_var, columnspan=2)
        a._entry(wizard, 16, 0, "Min Confidence (0-1)", a.preset_min_match_ratio_var, columnspan=2)

        # Tombol reference patch — row 19 (setelah field mean_std + spacing)
        ref_btn_row = ttk.Frame(wizard)
        ref_btn_row.grid(row=19, column=0, columnspan=4, sticky="ew", padx=12, pady=(0, 4))
        ttk.Button(ref_btn_row, text="Capture Referensi", command=a._capture_part_ready_ref).pack(side="left", padx=(0, 6))
        ttk.Button(ref_btn_row, text="Upload Referensi", command=a._upload_part_ready_ref).pack(side="left")
        a.gap_ref_status_label = ttk.Label(wizard, text="Referensi: belum dikonfigurasi", foreground="gray")
        a.gap_ref_status_label.grid(row=20, column=0, columnspan=4, sticky="w", padx=12, pady=(0, 6))

        # Preview persisten dari gambar master (edge-map) yang tersimpan, di bawah
        # tombol capture/upload.
        a.gap_ref_preview_label = tk.Label(wizard, text="(belum ada foto master)", fg="gray")
        a.gap_ref_preview_label.grid(row=21, column=0, columnspan=4, sticky="w", padx=12, pady=(0, 8))
        a._gap_ref_preview_photo = None

        # Visual ROI picker
        self._build_roi_picker(a, wizard)

        # Tombol aksi
        btn_row = ttk.Frame(wizard)
        btn_row.grid(row=26, column=0, columnspan=4, sticky="ew", padx=12, pady=(16, 6))
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(0, weight=1)

        a._preset_action_btn = ctk.CTkButton(
            btn_row,
            text="Simpan & Deploy Preset",
            command=a.save_and_deploy_preset,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=TEXT_ON_ACCENT,
            height=34,
            corner_radius=6,
        )
        a._preset_action_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        ctk.CTkButton(
            btn_row,
            text="Export template.json",
            command=a.export_runtime_template,
            fg_color=PANEL_ALT_BG,
            hover_color=BORDER,
            text_color=TEXT_PRIMARY,
            height=34,
            corner_radius=6,
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # Widget referensi part ready (capture/upload ref, label status, preview) — rows 19-21
        a._part_ready_ref_widgets = []
        for _r in (19, 20, 21):
            try:
                for w in wizard.grid_slaves(row=_r):
                    a._part_ready_ref_widgets.append(w)
            except Exception:
                pass

        # Simpan referensi ke field threshold mean-std (tampil cuma kalau method=mean_std_threshold)
        a._mean_std_field_widgets = []
        for _r in (14, 15, 16):
            try:
                for w in wizard.grid_slaves(row=_r):
                    a._mean_std_field_widgets.append(w)
            except Exception:
                pass

        # Sinkronisasi visibility awal
        self._on_part_ready_method_changed(a)

    def _on_part_ready_method_changed(self, a) -> None:
        """Tampilkan/sembunyikan field gap vs mean-std berdasarkan method part-ready."""
        method = a.preset_part_ready_method_var.get()
        show_mean_std = (method == "mean_std_threshold")
        for w in getattr(a, "_mean_std_field_widgets", []):
            w.grid() if show_mean_std else w.grid_remove()
        if hasattr(a, "_calib_mean_std_frame"):
            a._calib_mean_std_frame.grid() if show_mean_std else a._calib_mean_std_frame.grid_remove()
        show_ref = (method == "gap_template_match")
        for w in getattr(a, "_part_ready_ref_widgets", []) + getattr(a, "_gap_threshold_widgets", []):
            w.grid() if show_ref else w.grid_remove()

    # ------------------------------------------------------------------
    # ROI Picker
    def _build_roi_picker(self, a, wizard) -> None:
        roi_panel = ctk.CTkFrame(wizard, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        roi_panel.grid(row=22, column=0, columnspan=4, sticky="ew", padx=12, pady=(12, 2))
        roi_panel.columnconfigure(0, weight=1)
        a._roi_picker_panel = roi_panel

        ctk.CTkLabel(roi_panel, text="Visual ROI Picker", font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).grid(
            row=0, column=0, sticky="w", padx=10, pady=(10, 4),
        )

        roi_toolbar = ctk.CTkFrame(roi_panel, fg_color="transparent")
        roi_toolbar.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        roi_toolbar.columnconfigure(1, weight=1)

        ttk.Label(roi_toolbar, text="ROI").grid(row=0, column=0, sticky="w", padx=(0, 6))
        a.preset_roi_selector = ttk.Combobox(
            roi_toolbar,
            textvariable=a.preset_roi_choice_var,
            values=["Part Ready ROI", "Sticker ROI"],
            state="readonly",
            width=18,
        )
        a.preset_roi_selector.grid(row=0, column=1, sticky="w", padx=(0, 8))
        a.preset_roi_selector.bind("<<ComboboxSelected>>", a._on_preset_roi_selected)

        ttk.Button(roi_toolbar, text="Pilih Gambar", command=a._pick_preset_roi_image).grid(row=0, column=3, padx=(0, 4))
        a._live_cam_btn = ttk.Button(roi_toolbar, text="Mulai Live Camera", command=lambda: a._toggle_live_camera())
        a._live_cam_btn.grid(row=0, column=4, padx=(0, 4))
        ttk.Button(roi_toolbar, text="Reset", command=a._reset_preset_roi).grid(row=0, column=5)

        a.preset_roi_picker = RoiPickerCanvas(roi_panel, "Drag/resize ROI terpilih di gambar", size=(520, 292))
        a.preset_roi_picker.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        a.preset_roi_picker.on_roi_changed = lambda kind, roi: a._on_preset_roi_changed(kind, roi)
        a._on_preset_roi_selected()

        # Bagian Kalibrasi Mean Std (bisa di-collapse)
        a._calib_mean_std_frame = ctk.CTkFrame(roi_panel, fg_color=PANEL_ALT_BG, corner_radius=6, border_width=1, border_color=BORDER)
        a._calib_mean_std_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 8))
        a._calib_mean_std_frame.columnconfigure(1, weight=1)

        calib_header = ctk.CTkFrame(a._calib_mean_std_frame, fg_color="transparent")
        calib_header.grid(row=0, column=0, columnspan=4, sticky="ew", padx=8, pady=(4, 2))
        ctk.CTkLabel(calib_header, text="Kalibrasi Mean-Std", font=("Segoe UI", 9, "bold"), text_color=TEXT_PRIMARY).grid(row=0, column=0, sticky="w")

        # Step 1: Empty
        step1_frame = ctk.CTkFrame(a._calib_mean_std_frame, fg_color="transparent")
        step1_frame.grid(row=1, column=0, columnspan=4, sticky="ew", padx=8, pady=1)
        ctk.CTkButton(step1_frame, text="1. Capture Empty (tanpa part)", width=160, height=26,
                      command=lambda: self._calib_capture(a, "empty")).grid(row=0, column=0, padx=(0, 4))
        a._calib_empty_result = ctk.CTkLabel(step1_frame, text="—", text_color=TEXT_SECONDARY, font=("Segoe UI", 9))
        a._calib_empty_result.grid(row=0, column=1, sticky="w")

        # Step 2: Part
        step2_frame = ctk.CTkFrame(a._calib_mean_std_frame, fg_color="transparent")
        step2_frame.grid(row=2, column=0, columnspan=4, sticky="ew", padx=8, pady=1)
        ctk.CTkButton(step2_frame, text="2. Capture Part (hitam)", width=160, height=26,
                      command=lambda: self._calib_capture(a, "part")).grid(row=0, column=0, padx=(0, 4))
        a._calib_part_result = ctk.CTkLabel(step2_frame, text="—", text_color=TEXT_SECONDARY, font=("Segoe UI", 9))
        a._calib_part_result.grid(row=0, column=1, sticky="w")

        # Step 3: Sticker
        step3_frame = ctk.CTkFrame(a._calib_mean_std_frame, fg_color="transparent")
        step3_frame.grid(row=3, column=0, columnspan=4, sticky="ew", padx=8, pady=1)
        ctk.CTkButton(step3_frame, text="3. Capture Sticker", width=160, height=26,
                      command=lambda: self._calib_capture(a, "sticker")).grid(row=0, column=0, padx=(0, 4))
        a._calib_sticker_result = ctk.CTkLabel(step3_frame, text="—", text_color=TEXT_SECONDARY, font=("Segoe UI", 9))
        a._calib_sticker_result.grid(row=0, column=1, sticky="w")

        # Hasil
        result_frame = ctk.CTkFrame(a._calib_mean_std_frame, fg_color="transparent")
        result_frame.grid(row=4, column=0, columnspan=4, sticky="ew", padx=8, pady=(4, 2))
        a._calib_computed = ctk.CTkLabel(result_frame, text="Capture ketiganya untuk hitung threshold", text_color=TEXT_SECONDARY, font=("Segoe UI", 9))
        a._calib_computed.grid(row=0, column=0, sticky="w")
        ctk.CTkButton(result_frame, text="Terapkan", width=60, height=24, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                      text_color=TEXT_ON_ACCENT, command=lambda: self._calib_apply(a)).grid(row=0, column=3, sticky="e", padx=(8, 0))

        for var in (
            a.part_ready_roi_x_var,
            a.part_ready_roi_y_var,
            a.part_ready_roi_w_var,
            a.part_ready_roi_h_var,
            a.sticker_roi_x_var,
            a.sticker_roi_y_var,
            a.sticker_roi_w_var,
            a.sticker_roi_h_var,
        ):
            var.trace_add("write", lambda *_: a._sync_preset_roi_picker())

    # ------------------------------------------------------------------
    # Editor ROI Defect
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Capture Logo

    # ------------------------------------------------------------------
    # Kalibrasi Mean-Std
    # ------------------------------------------------------------------

    def _calib_capture(self, a, step: str) -> None:
        """Capture satu frame dari kamera dan hitung mean/std untuk ROI saat ini."""
        cam_idx = int(_float_or_default(a.preset_camera_index_var.get(), 0))
        try:
            from client_tk.app.services.camera_capture import CameraCaptureService
            cam = CameraCaptureService()
            cam.start(cam_idx)
            import time
            time.sleep(0.5)
            frame = cam.get_latest_frame()
            cam.stop()
            if frame is None:
                messagebox.showwarning("Kalibrasi", "Kamera tidak mengembalikan frame.")
                return
        except Exception as exc:
            messagebox.showerror("Kalibrasi", f"Gagal capture: {exc}")
            return

        # Crop ke ROI part_ready
        roi_f = a._roi_payload_from_vars(
            a.part_ready_roi_x_var, a.part_ready_roi_y_var,
            a.part_ready_roi_w_var, a.part_ready_roi_h_var,
            defaults={"x": 0.2, "y": 0.2, "w": 0.25, "h": 0.25},
        )
        fh, fw = frame.shape[:2]
        x = max(0, int(roi_f["x"] * fw))
        y = max(0, int(roi_f["y"] * fh))
        w = max(1, int(roi_f["w"] * fw))
        h = max(1, int(roi_f["h"] * fh))
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)
        crop = frame[y:y2, x:x2]

        if crop.size == 0:
            messagebox.showwarning("Kalibrasi", "Crop ROI kosong. Periksa posisi ROI.")
            return

        import cv2
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        mean_val = float(gray.mean())
        std_val = float(gray.std())

        if step == "empty":
            a._calib_empty_mean = mean_val
            a._calib_empty_result.configure(text=f"mean={mean_val:.1f}")
        elif step == "part":
            a._calib_part_mean = mean_val
            a._calib_part_std = std_val
            a._calib_part_result.configure(text=f"mean={mean_val:.1f}, std={std_val:.1f}")
        elif step == "sticker":
            a._calib_sticker_std = std_val
            a._calib_sticker_result.configure(text=f"std={std_val:.1f}")

        # Auto-hitung kalau ketiganya sudah di-capture
        if a._calib_empty_mean > 0 and a._calib_part_mean > 0 and a._calib_sticker_std > 0:
            from backend.app.services.part_ready_detector import compute_mean_std_thresholds
            result = compute_mean_std_thresholds(
                a._calib_empty_mean, a._calib_part_mean,
                a._calib_part_std, a._calib_sticker_std,
            )
            a._calib_computed.configure(
                text=f"MEAN_MAX={result['mean_max']:.1f}, STD_MAX={result['std_max']:.1f} (gaps: mean={a._calib_empty_mean - a._calib_part_mean:.1f}, std={a._calib_sticker_std - a._calib_part_std:.1f})"
            )

    def _calib_apply(self, a) -> None:
        """Terapkan threshold hasil hitung ke template."""
        if a._calib_empty_mean == 0 or a._calib_part_mean == 0 or a._calib_sticker_std == 0:
            messagebox.showwarning("Kalibrasi", "Capture ketiga kondisi dulu.")
            return
        from backend.app.services.part_ready_detector import compute_mean_std_thresholds
        result = compute_mean_std_thresholds(
            a._calib_empty_mean, a._calib_part_mean,
            a._calib_part_std, a._calib_sticker_std,
        )
        a.preset_mean_max_var.set(str(result["mean_max"]))
        a.preset_std_max_var.set(str(result["std_max"]))
        a._set_status(f"Threshold mean_std diterapkan: MEAN_MAX={result['mean_max']}, STD_MAX={result['std_max']}")
