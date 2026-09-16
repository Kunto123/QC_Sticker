"""Monitor / Production Results tab - extracted from AdminScreen."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import customtkinter as ctk

from client_tk.app.components.template_forms import LabeledValuePanel
from client_tk.app.screens.admin.tabs._widgets import CompactStatCard
from client_tk.app.theme import (
    ACCENT,
    ACCENT_HOVER,
    BORDER,
    PANEL_ALT_BG,
    PANEL_BG,
    SUCCESS,
    SUCCESS_HOVER,
    TEXT_ON_ACCENT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class ResultsTab:
    """Monitor / Production Results tab extracted from AdminScreen."""

    def __init__(self, admin, tab_frame):
        self.admin = admin
        self.frame = tab_frame
        self._build()

    def _build(self) -> None:
        a = self.admin
        f = self.frame

        f.columnconfigure(0, weight=1)
        f.rowconfigure(2, weight=1)

        body = self.admin._make_scrollable_body(f, "Monitor")

        filters = ctk.CTkFrame(body, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        filters.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))
        for index in range(6):
            filters.columnconfigure(index, weight=1 if index % 2 else 0)
        ctk.CTkLabel(filters, text="Production Monitor", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).grid(row=0, column=0, columnspan=6, sticky="w", padx=10, pady=(10, 0))
        ctk.CTkButton(filters, text="Refresh", command=self.admin.refresh_monitor, fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=TEXT_ON_ACCENT, height=28, corner_radius=6).grid(row=1, column=4, sticky="e", padx=(8, 4), pady=(4, 10))
        ctk.CTkButton(filters, text="Export CSV", command=self.admin.export_monitor_csv, fg_color=SUCCESS_HOVER, hover_color=SUCCESS, text_color=TEXT_ON_ACCENT, height=28, corner_radius=6).grid(row=1, column=5, sticky="e", padx=(4, 10), pady=(4, 10))

        a.monitor_cards_frame = ttk.Frame(body, padding=(8, 0, 8, 6))
        a.monitor_cards_frame.grid(row=1, column=0, sticky="ew")
        for index in range(6):
            a.monitor_cards_frame.columnconfigure(index, weight=1)
        self.admin.monitor_cards = {
            "total": CompactStatCard(a.monitor_cards_frame, "Total", background="#0f172a", foreground="#f8fafc"),
            "accept": CompactStatCard(a.monitor_cards_frame, "Accept", background="#166534", foreground="#f0fdf4"),
            "reject": CompactStatCard(a.monitor_cards_frame, "Reject", background="#991b1b", foreground="#fef2f2"),
            "reject_rate": CompactStatCard(a.monitor_cards_frame, "Reject Rate", background="#7c2d12", foreground="#fff7ed"),
            "pending": CompactStatCard(a.monitor_cards_frame, "Push Pending", background="#a16207", foreground="#fffbeb"),
            "failed": CompactStatCard(a.monitor_cards_frame, "Push Failed", background="#334155", foreground="#f8fafc"),
        }
        self.admin._layout_monitor_cards(compact=False)

        content = ttk.Frame(body, padding=8)
        content.grid(row=2, column=0, sticky="nsew")
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)
        self.admin.monitor_left = ttk.Frame(content)
        self.admin.monitor_right = ttk.Frame(content)
        self.admin.monitor_left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.admin.monitor_right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        recent = ctk.CTkFrame(self.admin.monitor_left, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        recent.pack(fill="both", expand=True)
        recent.columnconfigure(0, weight=1)
        recent.rowconfigure(2, weight=1)
        ctk.CTkLabel(recent, text="Recent Inspections", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 0))
        ctk.CTkLabel(recent, textvariable=self.admin.monitor_context_var, text_color=TEXT_SECONDARY).grid(row=1, column=0, sticky="w", padx=12, pady=(2, 8))
        self.admin.results_table = self.admin._build_table(
            recent,
            [
                ("id", "ID", 60, "center"),
                ("time", "Time", 145, "w"),
                ("decision", "Decision", 90, "center"),
                ("mode", "Mode", 100, "center"),
                ("part", "Part", 130, "w"),
                ("push", "Push", 90, "center"),
                ("reason", "Reason", 130, "w"),
            ],
            row=2,
            height=16,
        )
        actions = self.admin._build_action_row(recent, [("Retry Failed Visible", self.admin.retry_visible_failed_pushes, "neutral", "left")])
        actions.grid(row=3, column=0, sticky="ew", padx=12, pady=(8, 10))

        side = ctk.CTkFrame(self.admin.monitor_right, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        side.pack(fill="both", expand=True)
        side.columnconfigure(0, weight=1)
        side.rowconfigure(1, weight=1)
        ctk.CTkLabel(side, text="Recent Rejects", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 8))
        self.admin.reject_table = self.admin._build_table(
            side,
            [
                ("id", "ID", 60, "center"),
                ("part", "Part", 130, "w"),
                ("reason", "Reason", 150, "w"),
                ("time", "Time", 140, "w"),
            ],
            row=1,
            height=12,
        )
        self.admin.monitor_summary = LabeledValuePanel(
            side,
            "Selected Summary",
            [
                ("decision", "Decision"),
                ("reason", "Reason"),
                ("push_status", "Push"),
            ],
            columns=1,
        )
        self.admin.monitor_summary.grid(row=2, column=0, sticky="ew", padx=12, pady=(8, 10))
        self.admin.results_table.bind("<<TreeviewSelect>>", self.admin.open_monitor_result)

    # ------------------------------------------------------------------
    # Refresh and render
