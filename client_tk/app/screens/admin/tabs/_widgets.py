"""Shared UI widgets for admin tabs."""
from __future__ import annotations

import customtkinter as ctk

from client_tk.app.theme import BORDER


class CompactStatCard(ctk.CTkFrame):
    def __init__(self, master, title: str, *, background: str, foreground: str):
        super().__init__(master, fg_color=background, corner_radius=8, border_width=1, border_color=BORDER)
        self.columnconfigure(0, weight=1)
        ctk.CTkLabel(self, text=title, text_color=foreground, font=("Segoe UI", 9, "bold")).grid(
            row=0, column=0, sticky="w", padx=10, pady=(8, 0),
        )
        self.value_label = ctk.CTkLabel(self, text="0", text_color=foreground, font=("Segoe UI", 16, "bold"))
        self.value_label.grid(row=1, column=0, sticky="w", padx=10)
        self.note_label = ctk.CTkLabel(self, text="", text_color=foreground, font=("Segoe UI", 8))
        self.note_label.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 8))

    def set_value(self, value: object, note: str = "") -> None:
        self.value_label.configure(text=str(value))
        self.note_label.configure(text=str(note or ""))
