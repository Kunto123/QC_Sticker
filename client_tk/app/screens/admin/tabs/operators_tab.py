"""Operators tab -- user management and RFID bind."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import customtkinter as ctk

from client_tk.app.theme import (
    ACCENT,
    ACCENT_HOVER,
    BORDER,
    PANEL_BG,
    SUCCESS,
    SUCCESS_HOVER,
    TEXT_ON_ACCENT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class OperatorsTab:
    """User management and RFID bind interface."""

    def __init__(self, admin: object, tab_frame: tk.Frame) -> None:
        self.admin = admin
        self.frame = tab_frame
        self._build()

    def _build(self) -> None:
        a = self.admin
        a.operators_tab = self.frame
        a.operators_tab.columnconfigure(0, weight=3)
        a.operators_tab.columnconfigure(1, weight=2)
        a.operators_tab.rowconfigure(0, weight=1)

        body = a._make_scrollable_body(a.operators_tab, "Operators")

        a.operators_left = ttk.Frame(body, padding=8)
        a.operators_right = ttk.Frame(body, padding=8)
        a.operators_left.grid(row=0, column=0, sticky="nsew")
        a.operators_right.grid(row=0, column=1, sticky="nsew")

        self._build_user_list(a, a.operators_left)
        self._build_user_form(a, a.operators_right)
        self._build_bind_section(a, a.operators_right)

    def _build_user_list(self, a, parent) -> None:
        listing = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        listing.pack(fill="both", expand=True)
        listing.columnconfigure(0, weight=1)
        listing.rowconfigure(2, weight=1)
        ctk.CTkLabel(listing, text="Users", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY).grid(
            row=0, column=0, sticky="w", padx=12, pady=(10, 0)
        )
        ctk.CTkLabel(
            listing,
            text="Manage operators and admins. Edit role or delete users.",
            text_color=TEXT_SECONDARY,
        ).grid(row=1, column=0, sticky="w", padx=12, pady=(2, 8))
        a.users_table = a._build_table(
            listing,
            [
                ("id", "ID", 50, "center"),
                ("username", "Username", 140, "w"),
                ("role", "Role", 80, "center"),
                ("status", "Status", 80, "center"),
                ("rfid", "RFID", 80, "center"),
            ],
            row=2,
            height=18,
        )
        a.users_table.bind("<<TreeviewSelect>>", a._on_user_selected)
        a.users_table.bind("<Double-1>", a._on_user_double_click)
        footer = a._build_action_row(
            listing,
            [
                ("Refresh", a.refresh_operators, "neutral", "left"),
            ],
        )
        footer.grid(row=3, column=0, sticky="ew", padx=12, pady=(8, 10))
        a.users_table.bind("<<TreeviewSelect>>", a._on_user_selected)

    def _build_user_form(self, a, parent) -> None:
        form = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        a.operator_form_title = ctk.CTkLabel(
            form, text="Add User", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY
        )
        a.operator_form_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 0))
        a.operator_form_hint = ctk.CTkLabel(
            form,
            text="Create a new user, then bind RFID below.",
            text_color=TEXT_SECONDARY,
            wraplength=420,
            justify="left",
        )
        a.operator_form_hint.grid(row=1, column=0, columnspan=2, sticky="w", padx=12, pady=(2, 10))
        a._entry(form, 2, 0, "Username", a.operator_username_var, columns=2)
        ttk.Label(form, text="Password").grid(row=3, column=0, sticky="w", padx=(12, 8), pady=5)
        a.operator_password_entry = ttk.Entry(form, textvariable=a.operator_password_var, show="*")
        a.operator_password_entry.grid(row=3, column=1, sticky="ew", padx=(0, 12), pady=5)
        a.operator_password_hint = ctk.CTkLabel(
            form,
            text="Required for new users (min 6 chars). Leave blank when editing to keep current password.",
            text_color=TEXT_SECONDARY,
            font=("Segoe UI", 9),
            wraplength=420,
            justify="left",
        )
        a.operator_password_hint.grid(row=4, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 6))
        ttk.Label(form, text="Role").grid(row=5, column=0, sticky="w", padx=(12, 8), pady=5)
        role_combo = ttk.Combobox(
            form,
            textvariable=a.operator_role_var,
            values=("operator", "admin"),
            state="readonly",
            width=18,
        )
        role_combo.grid(row=5, column=1, sticky="ew", padx=(0, 12), pady=5)
        btn_row = ctk.CTkFrame(form, fg_color="transparent")
        btn_row.grid(row=6, column=0, columnspan=2, sticky="ew", padx=12, pady=(12, 10))
        a.operator_save_btn = ctk.CTkButton(
            btn_row,
            text="Create User",
            command=a._on_save_user,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=TEXT_ON_ACCENT,
            height=32,
            corner_radius=6,
        )
        a.operator_save_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))
        a.operator_cancel_btn = ctk.CTkButton(
            btn_row,
            text="Cancel Edit",
            command=a._on_cancel_edit,
            fg_color="#475569",
            hover_color="#64748b",
            text_color="#f8fafc",
            height=32,
            corner_radius=6,
        )
        a.operator_cancel_btn.pack(side="left", fill="x", expand=True, padx=(6, 0))
        a.operator_delete_btn = ctk.CTkButton(
            form,
            text="Delete User",
            command=a._on_delete_user,
            fg_color="#991b1b",
            hover_color="#7f1d1d",
            text_color="#fef2f2",
            height=32,
            corner_radius=6,
        )
        a.operator_delete_btn.grid(row=7, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 10))

    def _build_bind_section(self, a, parent) -> None:
        bind_frame = ctk.CTkFrame(parent, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER)
        bind_frame.pack(fill="x", pady=(8, 0))
        bind_frame.columnconfigure(1, weight=1)
        ctk.CTkLabel(
            bind_frame, text="Bind RFID", font=("Segoe UI", 12, "bold"), text_color=TEXT_PRIMARY
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 0))
        ctk.CTkLabel(
            bind_frame,
            text="Select a user from the list, scan RFID, then click Bind.",
            text_color=TEXT_SECONDARY,
            wraplength=420,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=12, pady=(2, 8))
        a.bind_target_label = ctk.CTkLabel(
            bind_frame,
            text="Select a user from the list",
            text_color=TEXT_SECONDARY,
            font=("Segoe UI", 10),
        )
        a.bind_target_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 6))
        bind_entry_row = ttk.Frame(bind_frame)
        bind_entry_row.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 6))
        bind_entry_row.columnconfigure(0, weight=1)
        a.unified_rfid_entry = ttk.Entry(bind_entry_row, textvariable=a.unified_rfid_var)
        a.unified_rfid_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        a.unified_rfid_entry.bind("<Return>", lambda e: "break")
        bind_btn_row = ctk.CTkFrame(bind_frame, fg_color="transparent")
        bind_btn_row.grid(row=4, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 6))
        a.bind_rfid_btn = ctk.CTkButton(
            bind_btn_row,
            text="Bind",
            command=a._on_bind_rfid,
            fg_color=SUCCESS,
            hover_color=SUCCESS_HOVER,
            text_color=TEXT_ON_ACCENT,
            height=30,
            corner_radius=6,
        )
        a.bind_rfid_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))
        a.clear_rfid_btn = ctk.CTkButton(
            bind_btn_row,
            text="Clear RFID",
            command=a._on_clear_rfid,
            fg_color="#475569",
            hover_color="#64748b",
            text_color="#f8fafc",
            height=30,
            corner_radius=6,
        )
        a.clear_rfid_btn.pack(side="left", fill="x", expand=True, padx=(6, 0))
        a.bind_rfid_status = ctk.CTkLabel(bind_frame, text="", text_color=TEXT_SECONDARY, font=("Segoe UI", 9))
        a.bind_rfid_status.grid(row=5, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 10))
