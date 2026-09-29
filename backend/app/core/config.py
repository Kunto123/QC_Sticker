"""Konfigurasi proses.

Cuma dua jenis nilai yang ada di sini:

* **Secret / bootstrap yang dibaca dari environment** (`.env`): secret key,
  backend database + kredensial, data root, topologi deployment (host / port /
  local-only). Ini harus sudah diketahui sebelum JSON store manapun dibuka.
* **Runtime settings milik `data/json_store/machine_settings.json`** — bagian
  `timing`, `inference` dan (untuk PLC) `connection` / `io` yang diedit dari
  Admin → Machine Settings. Default polos di bawah cuma placeholder;
  `backend/app/core/container.py` menimpanya dari file JSON saat boot lewat
  `AppConfig.apply_machine_settings()`. Tidak ada lagi di grup ini yang dibaca
  dari environment.

Selain itu, semua yang dulunya knob env sekarang jadi konstanta tetap.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


_SQL_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _sql_identifier(env_var: str, default: str) -> str:
    """Baca nama tabel/kolom dari environment, tolak apa pun yang bukan
    identifier SQL polos (boleh pakai prefix schema) — nilai ini langsung
    disisipkan ke string SQL, jadi env var yang salah harus gagal cepat saat
    import, bukan membuka celah injection."""
    value = os.getenv(env_var, default).strip()
    if not _SQL_IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{env_var}={value!r} bukan identifier SQL yang valid "
            "(huruf, angka, underscore, boleh pakai prefix schema)."
        )
    return value


def _sql_identifier_list(env_var: str, default: str) -> list[str]:
    """Sama seperti `_sql_identifier`, tapi env var boleh menyebut lebih dari
    satu kolom, dipisah koma — nilai logis yang sama akan ditulis ke semua
    kolom yang disebutkan. Untuk tabel tujuan yang menyimpan nilai sama di
    beberapa kolom duplikat/legacy (mis. `DateCheckMC` dan `DateSendDB`)."""
    raw = os.getenv(env_var, default)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{env_var}={raw!r} harus menyebut minimal satu kolom.")
    for part in parts:
        if not _SQL_IDENTIFIER_PATTERN.fullmatch(part):
            raise ValueError(
                f"{env_var}={raw!r} mengandung {part!r}, yang bukan identifier SQL yang valid "
                "(huruf, angka, underscore, boleh prefix schema; pisahkan beberapa "
                "nama kolom dengan koma)."
            )
    return parts


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.getenv("QC_SUITE_DATA_ROOT", PROJECT_ROOT / "data")).resolve()
JSON_STORE_DIR = DATA_ROOT / "json_store"
MODELS_DIR = DATA_ROOT / "models"

# ── Konstanta tetap (dulunya knob env) ─────────────────────────────
ACCESS_TOKEN_TTL_SECONDS = 86400
PUSH_WORKER_INTERVAL_SECONDS = 30
PUSH_WORKER_MAX_RETRY = 5
# Cuma reject reason ini (plus COMMIT_TIMEOUT) yang terminal; selain itu
# tetap pending supaya inference terus mencoba sampai ACCEPT.
INSPECT_HARD_REJECT_REASONS = "WRONG_TYPE"


@dataclass(slots=True)
class AppConfig:
    # ── Secret / bootstrap (env) ────────────────────────────────────
    host: str = os.getenv("QC_SUITE_HOST", "127.0.0.1")
    port: int = int(os.getenv("QC_SUITE_PORT", "8100"))
    debug: bool = os.getenv("QC_SUITE_DEBUG", "0").strip() == "1"
    secret_key: str = os.getenv("QC_SUITE_SECRET_KEY", "qc-suite-dev-secret")
    local_only: bool = os.getenv("QC_SUITE_LOCAL_ONLY", "1").strip() != "0"
    # Environment deployment: "dev" atau "prod". Mengontrol pengecekan keamanan.
    environment: str = os.getenv("QC_SUITE_ENV", "dev")
    relational_backend: str = os.getenv("QC_SUITE_DATABASE_BACKEND", "").strip().lower()
    sql_server: str = os.getenv("MSSQL_SERVER", "")
    sql_database: str = os.getenv("MSSQL_DATABASE", "")
    sql_username: str = os.getenv("MSSQL_USERNAME", "")
    sql_password: str = os.getenv("MSSQL_PASSWORD", "")
    sql_driver: str = os.getenv("MSSQL_DRIVER", "ODBC Driver 17 for SQL Server")
    postgresql_host: str = os.getenv("POSTGRESQL_HOST", "")
    postgresql_port: int = int(os.getenv("POSTGRESQL_PORT", "5432"))
    postgresql_database: str = os.getenv("POSTGRESQL_DATABASE", "")
    postgresql_username: str = os.getenv("POSTGRESQL_USERNAME", "")
    postgresql_password: str = os.getenv("POSTGRESQL_PASSWORD", "")
    postgresql_schema: str = os.getenv("POSTGRESQL_SCHEMA", "public")
    postgresql_sslmode: str = os.getenv("POSTGRESQL_SSLMODE", "prefer")
    # Tabel eksternal "operator" (milik MES pabrik, bukan app ini) yang jadi
    # basis autentikasi + manajemen user di backend SQL. Nama tabel dan
    # kolom spesifik per deployment — configurable supaya app ini bisa
    # menunjuk ke skema apa pun yang dipakai lokasi tersebut.
    operator_table: str = _sql_identifier("QC_SUITE_OPERATOR_TABLE", "operator")
    operator_col_no: str = _sql_identifier("QC_SUITE_OPERATOR_COL_NO", "No")
    operator_col_mc_id: str = _sql_identifier("QC_SUITE_OPERATOR_COL_MC_ID", "MC_ID")
    operator_col_rfid: str = _sql_identifier("QC_SUITE_OPERATOR_COL_RFID", "No_RFID")
    operator_col_member_id: str = _sql_identifier("QC_SUITE_OPERATOR_COL_MEMBER_ID", "Member_ID")
    operator_col_status: str = _sql_identifier("QC_SUITE_OPERATOR_COL_STATUS", "StatusMP")
    # Tabel push hasil-inspeksi eksternal (milik MES pabrik / sistem
    # reporting, bukan app ini) yang jadi tujuan mirror `HybridInspectionResultsRepository`
    # untuk hasil yang diterima. Alasannya sama seperti tabel operator di
    # atas: tidak pernah dibuat/diubah oleh app ini, nama tabel dan kolom
    # spesifik per deployment. Lima kolom data tetap pakai mapping logis
    # yang sudah ada (PartName/DateCheckMC/MPCheck/Data1/Data2/Line dari
    # `build_sql_payload`) — cuma nama kolom *fisik* di tabel tujuan yang
    # configurable di sini. Tiap satu dari lima itu boleh menyebut lebih
    # dari satu kolom fisik (dipisah koma) kalau tabel tujuan menyimpan
    # nilai yang sama di kolom-kolom duplikat — nilai yang sama itu lalu
    # ditulis ke semua kolom yang disebutkan saat insert.
    inspection_push_table: str = _sql_identifier("QC_SUITE_INSPECTION_TABLE", "qc_inspection_push")
    inspection_push_col_id: str = _sql_identifier("QC_SUITE_INSPECTION_COL_ID", "id")
    inspection_push_col_part_name: list[str] = field(
        default_factory=lambda: _sql_identifier_list("QC_SUITE_INSPECTION_COL_PART_NAME", "PartName")
    )
    inspection_push_col_date_check_mc: list[str] = field(
        default_factory=lambda: _sql_identifier_list("QC_SUITE_INSPECTION_COL_DATE_CHECK_MC", "DateCheckMC")
    )
    inspection_push_col_mp_check: list[str] = field(
        default_factory=lambda: _sql_identifier_list("QC_SUITE_INSPECTION_COL_MP_CHECK", "MPCheck")
    )
    inspection_push_col_data1: list[str] = field(
        default_factory=lambda: _sql_identifier_list("QC_SUITE_INSPECTION_COL_DATA1", "Data1")
    )
    inspection_push_col_data2: list[str] = field(
        default_factory=lambda: _sql_identifier_list("QC_SUITE_INSPECTION_COL_DATA2", "Data2")
    )
    inspection_push_col_line: list[str] = field(
        default_factory=lambda: _sql_identifier_list("QC_SUITE_INSPECTION_COL_LINE", "Line")
    )

    # ── Konstanta tetap yang diekspos di instance (dibaca service di sini) ──
    access_token_ttl_seconds: int = ACCESS_TOKEN_TTL_SECONDS
    push_worker_interval_seconds: int = PUSH_WORKER_INTERVAL_SECONDS
    push_worker_max_retry: int = PUSH_WORKER_MAX_RETRY
    inspect_hard_reject_reasons: str = INSPECT_HARD_REJECT_REASONS
    # Logging request/access ikut mode debug.
    access_logs_enabled: bool = os.getenv("QC_SUITE_DEBUG", "0").strip() == "1"
    werkzeug_logs_enabled: bool = os.getenv("QC_SUITE_DEBUG", "0").strip() == "1"

    # ── Milik machine_settings.json → `inference` (placeholder) ───
    sticker_inference_mode: str = "auto"
    device_mode: str = "auto"
    cuda_device_id: int = 0
    inference_num_threads: int = 4
    inference_timeout_s: float = 5.0
    default_sticker_model_path: str = ""
    default_sticker_model_meta_path: str = ""

    # ── Milik machine_settings.json → `timing` (placeholder) ──────
    phase_next_part_delay_ms: int = 2000
    phase_sticker_install_delay_ms: int = 0
    accept_stable_frames: int = 1
    accept_stable_ms: int = 200
    commit_grace_ms: int = 1500
    reject_timeout_ms: int = 15000
    part_ready_release_ms_default: int = 300
    part_ready_settle_ms_default: int = 0
    inference_cache_grace_ms: int = 300
    accept_holdover_ms: int = 2000
    inference_cache_ttl_ms: int = 10000
    session_idle_timeout_s: int = 300
    max_consecutive_rejects: int = 0

    # ── Milik machine_settings.json → `identity` (placeholder) ─────
    # Fallback untuk SessionState.line_id kalau sesi dimulai tanpa line_id —
    # client desktop tidak pernah mengirim line_id (slot line/station sudah
    # dihapus 2026-09-18), jadi ini satu-satunya cara "Line" di push SQL
    # tidak jadi null. Admin -> Machine Settings -> Identity.
    machine_line_id: str = ""

    def apply_machine_settings(self, settings) -> None:
        """Salin bagian `inference` dan `timing` dari MachineSettings ke object
        ini supaya tiap service tetap baca `app_config.<field>` tanpa berubah."""
        self.machine_line_id = str(getattr(settings.identity, "line", "") or "").strip()
        inf = settings.inference
        self.sticker_inference_mode = str(inf.mode or "auto").strip().lower() or "auto"
        self.device_mode = str(inf.device or "auto").strip().lower() or "auto"
        self.cuda_device_id = max(0, int(inf.cuda_device_id))
        self.inference_num_threads = max(1, int(inf.num_threads))
        self.inference_timeout_s = max(1.0, float(inf.timeout_s))
        self.default_sticker_model_path = str(inf.default_model_path or "").strip()
        self.default_sticker_model_meta_path = str(inf.default_model_meta_path or "").strip()
        t = settings.timing
        self.phase_next_part_delay_ms = max(0, int(t.phase_next_part_delay_ms))
        self.phase_sticker_install_delay_ms = max(0, int(t.phase_sticker_install_delay_ms))
        self.accept_stable_frames = max(1, int(t.accept_stable_frames))
        self.accept_stable_ms = max(0, int(t.accept_stable_ms))
        self.commit_grace_ms = max(0, int(t.commit_grace_ms))
        self.reject_timeout_ms = max(0, int(t.reject_timeout_ms))
        self.part_ready_release_ms_default = max(0, int(t.part_ready_release_ms))
        self.part_ready_settle_ms_default = max(0, int(t.part_ready_settle_ms_default))
        self.inference_cache_grace_ms = max(0, int(t.inference_cache_grace_ms))
        self.accept_holdover_ms = max(0, int(t.accept_holdover_ms))
        self.inference_cache_ttl_ms = max(100, int(t.inference_cache_ttl_ms))
        self.session_idle_timeout_s = max(0, int(t.session_idle_timeout_s))
        self.max_consecutive_rejects = max(0, int(t.max_consecutive_rejects))

    def _has_sqlserver_credentials(self) -> bool:
        return bool(
            self.sql_server
            and self.sql_database
            and self.sql_username
            and self.sql_password
        )

    def _has_postgresql_credentials(self) -> bool:
        return bool(
            self.postgresql_host
            and self.postgresql_database
            and self.postgresql_username
            and self.postgresql_password
        )

    @property
    def database_backend(self) -> str:
        """`QC_SUITE_DATABASE_BACKEND` harus di-set eksplisit; backend SQL
        yang dipilih tapi kredensialnya tidak lengkap diam-diam jatuh ke "local"."""
        backend = self.relational_backend
        if backend == "sqlserver":
            return backend if self._has_sqlserver_credentials() else "local"
        if backend == "postgresql":
            return backend if self._has_postgresql_credentials() else "local"
        return "local"

    @property
    def sql_enabled(self) -> bool:
        return self.database_backend in {"sqlserver", "postgresql"}

    @property
    def postgresql_enabled(self) -> bool:
        return self.database_backend == "postgresql"


def ensure_data_dirs() -> None:
    for path in (DATA_ROOT, JSON_STORE_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
