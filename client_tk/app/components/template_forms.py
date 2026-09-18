from __future__ import annotations

import customtkinter as ctk

from client_tk.app.theme import BORDER, PANEL_BG, TEXT_PRIMARY, TEXT_SECONDARY


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


