"""Modal warning shown when the backend's datapart guard locks further
judgements (every 5 accepted judgements pushed to SQL, until a downstream
MES process fills DatapartID for all 5). Offers a "Bypass" flow gated on
scanning a LEADERPI's RFID card — see backend/app/api/datapart_guard_routes.py.

The popup never closes itself except via the owning screen calling
``destroy()`` (or a successful bypass) — the owning screen polls
GET /datapart-guard/status in the background and dismisses it once the
downstream MES confirms, or once a LEADERPI bypasses it here.
"""
from __future__ import annotations

import customtkinter as ctk

from client_tk.app.theme import (
    ACCENT,
    ACCENT_HOVER,
    DANGER,
    PANEL_BG,
    TEXT_ON_ACCENT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WARNING,
    WARNING_HOVER,
)


class DatapartGuardPopup(ctk.CTkToplevel):
    def __init__(self, parent, *, on_bypass_rfid) -> None:
        super().__init__(parent)
        self._on_bypass_rfid = on_bypass_rfid
        self._rfid_var = ctk.StringVar()

        self.title("DATAPART Diperlukan")
        self.geometry("480x280")
        self.resizable(False, False)
        self.configure(fg_color=PANEL_BG)
        # No manual close — must resolve via bypass (RFID) or the guard
        # clearing on its own once DatapartID is confirmed.
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        self.transient(parent)
        self.grab_set()

        self._build_warning_view()
        self.after(50, self._center_on_parent)

    def _center_on_parent(self) -> None:
        try:
            self.update_idletasks()
            master = self.master
            px, py = master.winfo_rootx(), master.winfo_rooty()
            pw, ph = master.winfo_width(), master.winfo_height()
            w, h = self.winfo_width(), self.winfo_height()
            x = px + max(0, (pw - w) // 2)
            y = py + max(0, (ph - h) // 2)
            self.geometry(f"+{x}+{y}")
        except Exception:  # noqa: BLE001
            pass

    def _clear(self) -> None:
        for child in self.winfo_children():
            child.destroy()

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def _build_warning_view(self) -> None:
        self._clear()
        ctk.CTkLabel(
            self,
            text="⚠ Konfirmasi DATAPART Diperlukan",
            font=("", 18, "bold"),
            text_color=WARNING,
        ).pack(padx=24, pady=(28, 8))
        ctk.CTkLabel(
            self,
            text=(
                "5 judgement terakhir menunggu konfirmasi DatapartID dari sistem.\n"
                "Silakan scan DATAPART untuk part-part tersebut.\n"
                "Kamera dijeda sampai konfirmasi selesai."
            ),
            text_color=TEXT_PRIMARY,
            wraplength=420,
            justify="center",
        ).pack(padx=24, pady=(0, 8))
        self._status_label = ctk.CTkLabel(
            self,
            text="Memeriksa otomatis setiap beberapa detik...",
            text_color=TEXT_SECONDARY,
            font=("", 11),
        )
        self._status_label.pack(padx=24, pady=(0, 20))
        ctk.CTkButton(
            self,
            text="Bypass (RFID LEADERPI)",
            fg_color=WARNING,
            hover_color=WARNING_HOVER,
            text_color=TEXT_ON_ACCENT,
            command=self._build_bypass_view,
        ).pack(padx=24, pady=(0, 20))

    def _build_bypass_view(self) -> None:
        self._clear()
        ctk.CTkLabel(
            self,
            text="Scan RFID LEADERPI",
            font=("", 18, "bold"),
            text_color=TEXT_PRIMARY,
        ).pack(padx=24, pady=(28, 8))
        ctk.CTkLabel(
            self,
            text="Tempelkan/scan kartu RFID LEADERPI untuk melanjutkan.",
            text_color=TEXT_SECONDARY,
            wraplength=420,
            justify="center",
        ).pack(padx=24, pady=(0, 12))

        self._rfid_var.set("")
        entry = ctk.CTkEntry(self, textvariable=self._rfid_var, show="*", width=320)
        entry.pack(padx=24, pady=(0, 8))
        entry.bind("<Return>", lambda _e: self._submit_rfid())
        self._rfid_entry = entry

        self._error_label = ctk.CTkLabel(self, text="", text_color=DANGER, wraplength=420)
        self._error_label.pack(padx=24, pady=(0, 8))

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(padx=24, pady=(0, 20))
        ctk.CTkButton(
            btn_row,
            text="Batal",
            fg_color="transparent",
            border_width=1,
            text_color=TEXT_PRIMARY,
            command=self._build_warning_view,
            width=120,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            btn_row,
            text="Konfirmasi",
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=TEXT_ON_ACCENT,
            command=self._submit_rfid,
            width=120,
        ).pack(side="left")

        entry.focus_set()

    def _submit_rfid(self) -> None:
        uid = self._rfid_var.get().strip()
        self._rfid_var.set("")
        if not uid:
            return
        self.set_bypass_error("Memeriksa kartu...")
        self._on_bypass_rfid(uid)

    # ------------------------------------------------------------------
    # Called by the owning screen
    # ------------------------------------------------------------------

    def set_status_text(self, text: str) -> None:
        try:
            self._status_label.configure(text=text)
        except Exception:  # noqa: BLE001
            pass

    def set_bypass_error(self, text: str) -> None:
        try:
            self._error_label.configure(text=text)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._rfid_entry.focus_set()
        except Exception:  # noqa: BLE001
            pass
