from __future__ import annotations

import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import customtkinter as ctk
from dotenv import load_dotenv
from PIL import Image, ImageTk
import io

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS_DIR / "aski_logo.svg"
_logo_image_ref = None


LOGO_HEADER_WIDTH = 140  # Lebar logo di header shell, proporsional dengan tombol Start

def _get_logo_image(width: int = LOGO_HEADER_WIDTH):
    """Load and cache logo image (PNG preferred, SVG fallback via cairosvg)."""
    global _logo_image_ref
    try:
        # Try PNG first (most reliable in production)
        logo_png = ASSETS_DIR / "aski_logo.png"
        if logo_png.exists():
            img = Image.open(str(logo_png))
        else:
            # Fallback: SVG via cairosvg
            import cairosvg
            png_data = cairosvg.svg2png(url=str(LOGO_PATH), output_width=width)
            img = Image.open(io.BytesIO(png_data))
        ratio = width / img.width
        new_h = int(img.height * ratio)
        img = img.resize((width, new_h), Image.LANCZOS)
        _logo_image_ref = ImageTk.PhotoImage(img)
        return _logo_image_ref
    except Exception:
        return None
load_dotenv(PROJECT_ROOT / ".env")

from client_tk.app.api_client import ApiClient
from client_tk.app.config import DEFAULT_LOCAL_ONLY, DEFAULT_SERVER_URL
from client_tk.app.theme import (
    ACCENT,
    ACCENT_HOVER,
    APP_BG,
    PANEL_BG,
    SHELL_BG,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_ON_ACCENT,
    configure_customtkinter,
    configure_ttk_navy_theme,
)
from client_tk.app.screens.admin.view import AdminScreen
from client_tk.app.screens.operator.view import OperatorScreen
from client_tk.app.services.session_state import SessionState
from shared.contracts.enums import UserRole


ROLE_SCREEN_MAP = {
    UserRole.ADMIN.value: AdminScreen,
    UserRole.OPERATOR.value: OperatorScreen,
}


class LoginFrame(ctk.CTkFrame):
    def __init__(self, master, on_login, *, local_only: bool = False) -> None:
        super().__init__(master, fg_color=APP_BG, corner_radius=0)
        self._on_login = on_login
        self._local_only = bool(local_only)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        card = ctk.CTkFrame(self, fg_color=PANEL_BG, corner_radius=20, border_width=1, border_color="#26445f")
        card.grid(row=0, column=0, sticky="n", padx=32, pady=32)
        card.grid_columnconfigure(0, weight=0)
        card.grid_columnconfigure(1, weight=1)

        # Logo only — no text title
        logo_img = _get_logo_image(width=64)
        title_frame = ctk.CTkFrame(card, fg_color="transparent")
        title_frame.grid(row=0, column=0, columnspan=2, sticky="w", padx=24, pady=(24, 4))
        if logo_img is not None:
            logo_label = ctk.CTkLabel(title_frame, image=logo_img, text="")
            logo_label.pack(side="left")
        ctk.CTkLabel(
            card,
            text="Client desktop interface with a dark navy CustomTkinter shell.",
            font=("Segoe UI", 11),
            text_color=TEXT_SECONDARY,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=24, pady=(0, 20))

        ctk.CTkLabel(
            card,
            text="Login dengan username/password atau scan RFID.",
            font=("Segoe UI", 11, "bold"),
            text_color=TEXT_SECONDARY,
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=24, pady=(0, 8))

        ctk.CTkLabel(card, text="Runtime" if self._local_only else "Server URL", text_color=TEXT_PRIMARY).grid(row=3, column=0, sticky="w", padx=24, pady=8)
        ctk.CTkLabel(card, text="Username", text_color=TEXT_PRIMARY).grid(row=4, column=0, sticky="w", padx=24, pady=8)
        ctk.CTkLabel(card, text="Password", text_color=TEXT_PRIMARY).grid(row=5, column=0, sticky="w", padx=24, pady=8)
        ctk.CTkLabel(card, text="RFID Card", text_color=TEXT_PRIMARY).grid(row=6, column=0, sticky="w", padx=24, pady=8)

        self.base_url_var = tk.StringVar(value=DEFAULT_SERVER_URL)
        self.username_var = tk.StringVar()
        self.password_var = tk.StringVar()
        self.rfid_uid_var = tk.StringVar()

        self.base_url_entry = ctk.CTkEntry(card, textvariable=self.base_url_var, width=420, fg_color="#0f1c2b", border_color="#26445f", text_color=TEXT_PRIMARY)
        self.username_entry = ctk.CTkEntry(card, textvariable=self.username_var, width=420, fg_color="#0f1c2b", border_color="#26445f", text_color=TEXT_PRIMARY)
        self.password_entry = ctk.CTkEntry(card, textvariable=self.password_var, show="*", width=420, fg_color="#0f1c2b", border_color="#26445f", text_color=TEXT_PRIMARY)
        self.rfid_entry = ctk.CTkEntry(card, textvariable=self.rfid_uid_var, width=420, fg_color="#0f1c2b", border_color="#26445f", text_color=TEXT_PRIMARY)
        self.base_url_entry.grid(row=3, column=1, sticky="ew", padx=(0, 24), pady=8)
        self.username_entry.grid(row=4, column=1, sticky="ew", padx=(0, 24), pady=8)
        self.password_entry.grid(row=5, column=1, sticky="ew", padx=(0, 24), pady=8)
        self.rfid_entry.grid(row=6, column=1, sticky="ew", padx=(0, 24), pady=8)
        if self._local_only:
            self.base_url_entry.configure(state="disabled")
        ctk.CTkLabel(
            card,
            text="Scan kartu RFID di field ini untuk login langsung.",
            font=("Segoe UI", 11),
            text_color=TEXT_SECONDARY,
        ).grid(row=7, column=1, sticky="w", padx=(0, 24), pady=(0, 8))

        ctk.CTkButton(
            card,
            text="Login Username/Password",
            command=self._submit_password,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=TEXT_ON_ACCENT,
            width=200,
        ).grid(row=8, column=1, sticky="w", padx=(0, 24), pady=(18, 24))

        ctk.CTkButton(
            card,
            text="Login RFID",
            command=self._submit_rfid,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=TEXT_ON_ACCENT,
            width=140,
        ).grid(row=8, column=1, sticky="e", padx=(0, 24), pady=(18, 24))

        self.username_entry.bind("<Return>", lambda _event: self._submit_password())
        self.password_entry.bind("<Return>", lambda _event: self._submit_password())
        self.rfid_entry.bind("<Return>", lambda _event: self._submit())

    def _submit_password(self) -> None:
        self._on_login(
            self.base_url_var.get().strip(),
            "password",
            self.username_var.get().strip(),
            self.password_var.get().strip(),
            "",
        )

    def _submit_rfid(self) -> None:
        rfid_uid = self.rfid_uid_var.get().strip()
        self.rfid_uid_var.set("")
        self._on_login(
            self.base_url_var.get().strip(),
            "rfid",
            "",
            "",
            rfid_uid,
        )

    def _submit(self) -> None:
        self._submit_rfid()

    def focus_credentials(self) -> None:
        self.rfid_entry.focus_set()
        self.rfid_entry.select_range(0, "end")

    def set_base_url(self, value: str) -> None:
        self.base_url_var.set(value)


class QcSuiteDesktopApp(ctk.CTk):
    def __init__(self) -> None:
        configure_customtkinter()
        super().__init__()
        self.local_only = DEFAULT_LOCAL_ONLY
        self.title("QC Suite")
        self.geometry("1440x900")
        self.minsize(1160, 720)
        self.resizable(True, True)
        self.configure(fg_color=APP_BG)

        style = ttk.Style(self)
        configure_ttk_navy_theme(style)

        self.api = ApiClient(DEFAULT_SERVER_URL)
        self.session_state = SessionState(base_url=DEFAULT_SERVER_URL)
        self.active_screen: ttk.Frame | None = None

        self.login_frame = LoginFrame(self, self._handle_login, local_only=self.local_only)
        self.shell = ctk.CTkFrame(self, fg_color=APP_BG, corner_radius=0)
        self.shell.pack_forget()

        header = ctk.CTkFrame(self.shell, fg_color=SHELL_BG, corner_radius=0, border_width=0)
        header.pack(fill="x")
        # Logo in header
        header_logo = _get_logo_image(width=LOGO_HEADER_WIDTH)
        if header_logo is not None:
            ctk.CTkLabel(header, image=header_logo, text="").pack(side="left", padx=(16, 8), pady=10)
        self.user_label = ctk.CTkLabel(header, text="Not authenticated", text_color=TEXT_SECONDARY)
        self.user_label.pack(side="left", padx=16)
        self.endpoint_label = ctk.CTkLabel(
            header,
            text="Embedded local runtime" if self.local_only else DEFAULT_SERVER_URL,
            text_color=TEXT_SECONDARY,
        )
        self.endpoint_label.pack(side="left", padx=16)
        ctk.CTkButton(
            header,
            text="Logout",
            command=self._logout,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=TEXT_ON_ACCENT,
            width=120,
        ).pack(side="right", padx=16, pady=12)

        self.screen_host = ctk.CTkFrame(self.shell, fg_color=APP_BG, corner_radius=0)
        self.screen_host.pack(fill="both", expand=True)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind_all("<Control-Shift-Q>", lambda _event: self._on_close())
        self.bind_all("<Control-Shift-q>", lambda _event: self._on_close())
        self._show_login()

    def _focus_login_rfid(self) -> None:
        try:
            if self.login_frame.winfo_ismapped():
                self.focus_force()
                self.login_frame.focus_credentials()
        except tk.TclError:
            return

    def _show_login(self) -> None:
        self._teardown_screen()
        self.shell.pack_forget()
        self.login_frame.set_base_url(self.session_state.base_url if not self.local_only else DEFAULT_SERVER_URL)
        self.login_frame.pack(fill="both", expand=True)
        self.after_idle(self._focus_login_rfid)
        self.after(100, self._focus_login_rfid)

    def _show_shell(self) -> None:
        self.login_frame.pack_forget()
        self.shell.pack(fill="both", expand=True)

    def _handle_login(self, base_url: str, login_method: str, username: str, password: str, rfid_uid: str) -> None:
        if self.local_only:
            base_url = DEFAULT_SERVER_URL
        if (not self.local_only and not base_url):
            messagebox.showerror("Login", "Server URL wajib diisi.")
            return
        try:
            self.api = ApiClient(base_url)
            if login_method == "rfid":
                if not rfid_uid:
                    messagebox.showerror("Login", "Kartu RFID wajib discan.")
                    return
                auth_payload = self.api.login_rfid(rfid_uid)
            else:
                if not username or not password:
                    messagebox.showerror("Login", "Username dan password wajib diisi.")
                    return
                auth_payload = self.api.login(username, password)
            token = auth_payload["token"]
            self.api.set_token(token)
            user = self.api.me()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Login failed", str(exc))
            return

        self.session_state = SessionState(base_url=base_url, token=token, user=user)
        self._mount_screen(user.get("role"))

    def _mount_screen(self, role: str | None) -> None:
        screen_class = ROLE_SCREEN_MAP.get(str(role or "").strip())
        if screen_class is None:
            messagebox.showerror("Role", f"Unsupported role: {role}")
            self._show_login()
            return
        self._teardown_screen()
        self.user_label.configure(text=f"{self.session_state.user.get('username')} ({self.session_state.user.get('role')})")
        endpoint_display = "Embedded local runtime" if self.local_only else self.session_state.base_url
        self.endpoint_label.configure(text=endpoint_display)
        self.active_screen = screen_class(self.screen_host, self.api, self.session_state)
        self.active_screen.pack(fill="both", expand=True)
        self._show_shell()

    def _teardown_screen(self) -> None:
        if self.active_screen is None:
            return
        if hasattr(self.active_screen, "shutdown"):
            try:
                self.active_screen.shutdown()
            except Exception:  # noqa: BLE001
                pass
        self.active_screen.destroy()
        self.active_screen = None

    def _logout(self) -> None:
        base_url = self.session_state.base_url or DEFAULT_SERVER_URL
        if self.session_state.token:
            try:
                self.api.logout()
            except Exception:  # noqa: BLE001
                pass
        self._teardown_screen()
        self.api.set_token(None)
        self.session_state = SessionState(base_url=base_url)
        self.user_label.configure(text="Not authenticated")
        self.endpoint_label.configure(text=base_url)
        self._show_login()

    def _on_close(self) -> None:
        self._teardown_screen()
        self.destroy()

    # ── Restart (Machine Settings → Connection / Inference changes) ──

    def can_restart_backend(self) -> bool:
        """True when restarting this process also restarts the backend: embedded
        local-only mode, or split mode launched by scripts/run_desktop.py (which owns
        the backend subprocess). A remote backend cannot be restarted from here."""
        if self.local_only:
            return True
        return Path(sys.argv[0]).name == "run_desktop.py"

    def restart_app(self) -> None:
        """Shut this process down cleanly and relaunch the same command line.

        Order matters: release the camera and stop the PLC worker (all coils off,
        port closed) BEFORE the detached relauncher is spawned, then exit. The
        relauncher waits until this pid is gone, so in split mode the old backend
        subprocess is already terminated when the new one binds its port."""
        from client_tk.app.services.app_restart import schedule_relaunch

        self._teardown_screen()
        if self.local_only:
            try:
                from backend.app.core.shutdown import shutdown_workers
                shutdown_workers(reason="restart")
            except Exception:  # noqa: BLE001
                pass
        argv = [sys.executable, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
        data_root = Path(os.getenv("QC_SUITE_DATA_ROOT", str(PROJECT_ROOT / "data")))
        schedule_relaunch(argv, cwd=os.getcwd(), log_path=data_root / "restart.log")
        self.destroy()


def launch() -> None:
    app = QcSuiteDesktopApp()
    app.mainloop()


if __name__ == "__main__":
    launch()
