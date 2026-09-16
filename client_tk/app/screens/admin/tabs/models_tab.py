"""Models tab -- model registry, import, export, delete."""
from __future__ import annotations

import base64
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import customtkinter as ctk

from client_tk.app.theme import (
    ACCENT,
    BORDER,
    PANEL_BG,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class ModelsTab:
    """Model registry and import interface."""

    def __init__(self, admin: object, tab_frame: tk.Frame) -> None:
        self.admin = admin
        self.frame = tab_frame
        self._build()

    def _build(self) -> None:
        a = self.admin
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        body = a._make_scrollable_body(self.frame, "Models")

        shell = ttk.Frame(body)
        shell.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        shell.columnconfigure(0, weight=2)
        shell.columnconfigure(1, weight=3)
        shell.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(shell, text="Model Registry", padding=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        a._admin_model_list = tk.Listbox(left, height=16)
        a._admin_model_list.grid(row=0, column=0, sticky="nsew")
        a._admin_model_list.bind("<<ListboxSelect>>", a._on_admin_model_selected)

        btn_bar = ttk.Frame(left)
        btn_bar.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(btn_bar, text="Refresh", command=a.refresh_model_options).pack(side="left")
        ttk.Button(btn_bar, text="Export", command=a._admin_export_model).pack(side="left", padx=4)
        ttk.Button(btn_bar, text="Delete", command=a._admin_delete_model).pack(side="left")

        right = ttk.LabelFrame(shell, text="Import / Detail", padding=8)
        right.grid(row=0, column=1, sticky="nsew")

        import_frame = ttk.LabelFrame(right, text="Upload Model (.pt / .tflite / .onnx / .xml)", padding=6)
        import_frame.pack(fill="x")
        a._admin_import_path_var = tk.StringVar(value="No file selected")
        a._admin_import_name_var = tk.StringVar()
        a._admin_import_format_var = tk.StringVar(value="auto")
        ttk.Label(import_frame, text="Model Name:").pack(anchor="w")
        ttk.Entry(import_frame, textvariable=a._admin_import_name_var).pack(fill="x", pady=(2, 4))
        fmt_row = ttk.Frame(import_frame)
        fmt_row.pack(fill="x", pady=(0, 4))
        ttk.Label(fmt_row, text="Format:").pack(side="left")
        ttk.Radiobutton(fmt_row, text="Auto-detect", variable=a._admin_import_format_var, value="auto").pack(side="left", padx=(4, 0))
        ttk.Radiobutton(fmt_row, text="PyTorch (.pt)", variable=a._admin_import_format_var, value="pt").pack(side="left", padx=(4, 0))
        ttk.Radiobutton(fmt_row, text="TFLite (.tflite)", variable=a._admin_import_format_var, value="tflite").pack(side="left", padx=(4, 0))
        ttk.Radiobutton(fmt_row, text="ONNX (.onnx)", variable=a._admin_import_format_var, value="onnx").pack(side="left", padx=(4, 0))
        ttk.Radiobutton(fmt_row, text="OpenVINO (.xml)", variable=a._admin_import_format_var, value="openvino").pack(side="left", padx=(4, 0))
        ttk.Button(import_frame, text="Choose File", command=a._admin_choose_import_file).pack(anchor="w")
        ttk.Label(import_frame, textvariable=a._admin_import_path_var, wraplength=300).pack(anchor="w", pady=(2, 0))
        ttk.Button(import_frame, text="Upload", command=a._admin_import_model).pack(anchor="w", pady=(4, 0))
        a._admin_import_path: str = ""

        a._admin_model_detail_var = tk.StringVar(value="Select a model to view details.")
        ttk.Label(right, textvariable=a._admin_model_detail_var, foreground="#475569", wraplength=400, justify="left").pack(anchor="w", pady=(10, 0))
