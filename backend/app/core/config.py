"""Process configuration.

Only two kinds of values live here:

* **Secrets / bootstrap read from the environment** (`.env`): secret key,
  database backend + credentials, data root, deployment topology (host / port /
  local-only). These must be known before any JSON store can be opened.
* **Runtime settings owned by `data/json_store/machine_settings.json`** — the
  `timing`, `inference` and (for the PLC) `connection` / `io` sections edited
  from Admin → Machine Settings. The plain defaults below are placeholders;
  `backend/app/core/container.py` overwrites them from the JSON file at boot
  via `AppConfig.apply_machine_settings()`. Nothing in this group is read from
  the environment any more.

Everything else that used to be an env knob is a fixed constant now.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.getenv("QC_SUITE_DATA_ROOT", PROJECT_ROOT / "data")).resolve()
JSON_STORE_DIR = DATA_ROOT / "json_store"
MODELS_DIR = DATA_ROOT / "models"

# ── Fixed constants (formerly env knobs) ─────────────────────────────
ACCESS_TOKEN_TTL_SECONDS = 86400
PUSH_WORKER_INTERVAL_SECONDS = 30
PUSH_WORKER_MAX_RETRY = 5
# Only these reject reasons (plus COMMIT_TIMEOUT) are terminal; everything else
# stays pending so inference keeps retrying until ACCEPT.
INSPECT_HARD_REJECT_REASONS = "WRONG_TYPE"


@dataclass(slots=True)
class AppConfig:
    # ── Secrets / bootstrap (env) ────────────────────────────────────
    host: str = os.getenv("QC_SUITE_HOST", "127.0.0.1")
    port: int = int(os.getenv("QC_SUITE_PORT", "8100"))
    debug: bool = os.getenv("QC_SUITE_DEBUG", "0").strip() == "1"
    secret_key: str = os.getenv("QC_SUITE_SECRET_KEY", "qc-suite-dev-secret")
    local_only: bool = os.getenv("QC_SUITE_LOCAL_ONLY", "1").strip() != "0"
    # Deployment environment: "dev" or "prod". Controls hardening checks.
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

    # ── Fixed constants exposed on the instance (services read them here) ──
    access_token_ttl_seconds: int = ACCESS_TOKEN_TTL_SECONDS
    push_worker_interval_seconds: int = PUSH_WORKER_INTERVAL_SECONDS
    push_worker_max_retry: int = PUSH_WORKER_MAX_RETRY
    inspect_hard_reject_reasons: str = INSPECT_HARD_REJECT_REASONS
    # Request/access logging follows debug mode.
    access_logs_enabled: bool = os.getenv("QC_SUITE_DEBUG", "0").strip() == "1"
    werkzeug_logs_enabled: bool = os.getenv("QC_SUITE_DEBUG", "0").strip() == "1"

    # ── Owned by machine_settings.json → `inference` (placeholders) ───
    sticker_inference_mode: str = "auto"
    device_mode: str = "auto"
    cuda_device_id: int = 0
    inference_num_threads: int = 4
    inference_timeout_s: float = 5.0
    default_sticker_model_path: str = ""
    default_sticker_model_meta_path: str = ""

    # ── Owned by machine_settings.json → `timing` (placeholders) ──────
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

    def apply_machine_settings(self, settings) -> None:
        """Copy the `inference` and `timing` sections of MachineSettings onto this
        object so every service keeps reading `app_config.<field>` unchanged."""
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
        """`QC_SUITE_DATABASE_BACKEND` must be set explicitly; a selected SQL
        backend with incomplete credentials silently falls back to "local"."""
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
