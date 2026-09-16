from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import customtkinter as ctk


APP_BG = "#07111d"
SHELL_BG = "#0b1624"
PANEL_BG = "#102033"
PANEL_ALT_BG = "#13263a"
BORDER = "#1f3750"
INPUT_BG = "#0f1c2b"
TEXT_PRIMARY = "#e2e8f0"
TEXT_SECONDARY = "#94a3b8"
TEXT_ON_ACCENT = "#f8fafc"
ACCENT = "#2563eb"
ACCENT_HOVER = "#1d4ed8"
ACCENT_SOFT = "#1e40af"
SUCCESS = "#15803d"
SUCCESS_HOVER = "#166534"
DANGER = "#b91c1c"
DANGER_HOVER = "#991b1b"
WARNING = "#d97706"
WARNING_HOVER = "#b45309"


def configure_customtkinter() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")


def configure_ttk_navy_theme(style: ttk.Style) -> None:
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", background=APP_BG, foreground=TEXT_PRIMARY)
    style.configure("TFrame", background=APP_BG)
    style.configure("TLabel", background=APP_BG, foreground=TEXT_PRIMARY)
    style.configure("TLabelframe", background=PANEL_BG, foreground=TEXT_PRIMARY, borderwidth=1)
    style.configure("TLabelframe.Label", background=PANEL_BG, foreground=TEXT_PRIMARY)
    style.configure("TButton", background=ACCENT, foreground=TEXT_ON_ACCENT, padding=(12, 8), borderwidth=0)
    style.map(
        "TButton",
        background=[("active", ACCENT_HOVER), ("pressed", ACCENT_SOFT), ("disabled", PANEL_ALT_BG)],
        foreground=[("disabled", TEXT_SECONDARY)],
    )
    style.configure(
        "TEntry",
        fieldbackground=INPUT_BG,
        foreground=TEXT_PRIMARY,
        insertcolor=TEXT_PRIMARY,
        borderwidth=1,
        relief="flat",
    )
    style.map(
        "TEntry",
        fieldbackground=[("disabled", PANEL_ALT_BG), ("readonly", PANEL_BG)],
        foreground=[("disabled", TEXT_SECONDARY)],
    )
    style.configure(
        "TSpinbox",
        fieldbackground=INPUT_BG,
        background=INPUT_BG,
        foreground=TEXT_PRIMARY,
        insertcolor=TEXT_PRIMARY,
        arrowcolor=TEXT_PRIMARY,
        borderwidth=1,
        relief="flat",
    )
    style.map(
        "TSpinbox",
        fieldbackground=[("disabled", PANEL_ALT_BG), ("readonly", PANEL_BG)],
        foreground=[("disabled", TEXT_SECONDARY)],
        arrowcolor=[("disabled", TEXT_SECONDARY)],
    )
    style.configure(
        "TCombobox",
        fieldbackground=INPUT_BG,
        background=INPUT_BG,
        foreground=TEXT_PRIMARY,
        arrowcolor=TEXT_PRIMARY,
        borderwidth=1,
        relief="flat",
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", PANEL_BG), ("disabled", PANEL_ALT_BG)],
        foreground=[("disabled", TEXT_SECONDARY)],
        arrowcolor=[("disabled", TEXT_SECONDARY)],
    )
    style.configure("TNotebook", background=APP_BG, borderwidth=0, tabmargins=(4, 4, 4, 0))
    style.configure(
        "TNotebook.Tab",
        background=PANEL_BG,
        foreground=TEXT_SECONDARY,
        padding=(14, 8),
        borderwidth=0,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", SHELL_BG), ("active", PANEL_ALT_BG)],
        foreground=[("selected", TEXT_PRIMARY), ("active", TEXT_PRIMARY)],
    )
    style.configure("Treeview", background=PANEL_BG, fieldbackground=PANEL_BG, foreground=TEXT_PRIMARY, borderwidth=0, rowheight=26)
    style.configure("Treeview.Heading", background=SHELL_BG, foreground=TEXT_PRIMARY, relief="flat")
    style.map("Treeview", background=[("selected", ACCENT_SOFT)], foreground=[("selected", TEXT_ON_ACCENT)])
    style.configure("TScrollbar", background=PANEL_BG, troughcolor=APP_BG, borderwidth=0, arrowsize=14)
