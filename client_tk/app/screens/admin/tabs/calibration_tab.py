"""Calibration tab -- color calibration profiles."""
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


class CalibrationTab:
    """Calibration profile management interface."""

    def __init__(self, admin: object, tab_frame: tk.Frame) -> None:
        self.admin = admin
        self.frame = tab_frame
        self._build()

    def _build(self) -> None:
        a = self.admin
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        body = a._make_scrollable_body(self.frame, "Calibration")

        shell = ttk.Frame(body)
        shell.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(shell, text="Calibration Profiles", padding=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        a._admin_cal_list = tk.Listbox(left, height=12)
        a._admin_cal_list.grid(row=0, column=0, sticky="nsew")

        btn_bar = ttk.Frame(left)
        btn_bar.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(btn_bar, text="Refresh", command=a._admin_refresh_calibration).pack(side="left")
        ttk.Button(btn_bar, text="Delete", command=a._admin_delete_calibration).pack(side="left", padx=4)

        right = ttk.LabelFrame(shell, text="Create Profile", padding=8)
        right.grid(row=0, column=1, sticky="nsew")

        form = ttk.Frame(right)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        a._admin_cal_name_var = tk.StringVar()
        a._admin_cal_desc_var = tk.StringVar()
        ttk.Label(form, text="Name").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=a._admin_cal_name_var).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Label(form, text="Description").grid(row=1, column=0, sticky="w")
        ttk.Entry(form, textvariable=a._admin_cal_desc_var).grid(row=1, column=1, sticky="ew", padx=(4, 0))

        ttk.Label(right, text="Upload a calibration image:", foreground="#475569").pack(anchor="w", pady=(10, 2))
        a._admin_cal_image_var = tk.StringVar(value="No image selected")
        ttk.Button(right, text="Choose Image", command=a._admin_choose_cal_image).pack(anchor="w")
        ttk.Label(right, textvariable=a._admin_cal_image_var, wraplength=300).pack(anchor="w", pady=(2, 0))
        a._admin_cal_image_path: str = ""

        ttk.Button(right, text="Create Profile", command=a._admin_create_calibration).pack(anchor="w", pady=(10, 0))
