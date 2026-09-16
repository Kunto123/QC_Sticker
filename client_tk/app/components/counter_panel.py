from __future__ import annotations

import tkinter as tk

import customtkinter as ctk

from client_tk.app.theme import BORDER, PANEL_BG, TEXT_PRIMARY, TEXT_SECONDARY, SUCCESS, DANGER

BREAKDOWN_ORDER = (
    "OUT_OF_ANGLE",
    "WRONG_TYPE",
    "COMMIT_TIMEOUT",
)


class CounterPanel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color=PANEL_BG, corner_radius=16, border_width=1, border_color=BORDER)
        self.grid_columnconfigure(0, weight=1)

        # Match Ratio - prominent display
        self.match_ratio_var = ctk.CTkLabel(
            self,
            text="Match Ratio: --%",
            font=("Segoe UI", 18, "bold"),
            text_color=TEXT_PRIMARY,
        )
        self.match_ratio_var.pack(anchor="w", padx=16, pady=(16, 8))

        # Session counters (secondary)
        cards = ctk.CTkFrame(self, fg_color="transparent")
        cards.pack(fill="x", padx=12, pady=(0, 8))

        self.total_value = self._build_card(cards, "Total", "#0f172a", "#f8fafc")
        self.accept_value = self._build_card(cards, "Accept", "#166534", "#f0fdf4")
        self.reject_value = self._build_card(cards, "Reject", "#991b1b", "#fef2f2")

        self.meta_var = ctk.CTkLabel(self, text="Scope: session", text_color=TEXT_SECONDARY)
        self.meta_var.pack(anchor="w", padx=16, pady=(8, 16))

    def _build_card(self, master, title: str, bg: str, fg: str) -> tk.Label:
        frame = ctk.CTkFrame(master, fg_color=bg, corner_radius=14)
        frame.pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkLabel(frame, text=title, text_color=fg, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 0))
        value = ctk.CTkLabel(frame, text="0", text_color=fg, font=("Segoe UI", 24, "bold"))
        value.pack(anchor="w", padx=12, pady=(0, 10))
        return value

    def update_payload(self, payload: dict | None) -> None:
        counters = (payload or {}).get("counters") or {}
        self.total_value.configure(text=str(counters.get("session_total", 0)))
        self.accept_value.configure(text=str(counters.get("session_accept", 0)))
        self.reject_value.configure(text=str(counters.get("session_reject", 0)))
        self.meta_var.configure(text=f"Scope: {counters.get('scope') or 'session'}")

        # Update match ratio from validation_details
        if payload:
            validation_details = (payload.get("validation") or {}).get("validation_details") or {}
            if validation_details.get("mode") == "counter":
                rois = validation_details.get("rois", [])
                if rois:
                    total = len(rois)
                    ok_count = sum(1 for r in rois if r.get("ok", False))
                    ratio = (ok_count / total * 100) if total > 0 else 0.0
                    self.update_match_ratio(ratio)
                else:
                    self.update_match_ratio(0.0)
            # For sticker mode: show part_ready match_ratio instead
            elif validation_details.get("mode") in (None, "", "sticker"):
                pr = payload.get("part_ready") or {}
                match_r = pr.get("match_ratio")
                if match_r is not None:
                    self.update_match_ratio(float(match_r) * 100.0)
                else:
                    self.match_ratio_var.configure(text="Match Ratio: --%", text_color=TEXT_PRIMARY)

    def update_match_ratio(self, ratio: float) -> None:
        """Update the match ratio display with color coding."""
        self.match_ratio_var.configure(text=f"Match Ratio: {ratio:.1f}%")
        # Color: green if 100%, yellow if partial, red if 0%
        if ratio >= 100.0:
            self.match_ratio_var.configure(text_color=SUCCESS)
        elif ratio > 0.0:
            self.match_ratio_var.configure(text_color="#f59e0b")  # amber
        else:
            self.match_ratio_var.configure(text_color=DANGER)

    def reset(self) -> None:
        self.total_value.configure(text="0")
        self.accept_value.configure(text="0")
        self.reject_value.configure(text="0")
        self.meta_var.configure(text="Scope: session")
        self.match_ratio_var.configure(text="Match Ratio: --%", text_color=TEXT_PRIMARY)