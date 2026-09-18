from __future__ import annotations

import logging

from backend.app.core.config import AppConfig
from backend.app.core.device_runtime import DeviceRuntimeResolver
from backend.app.repositories.auth_audit_repository import AuthAuditRepository
from backend.app.core.security import TokenStore
from backend.app.repositories.deployments_repository import DeploymentsRepository
from backend.app.repositories.hybrid_inspection_results_repository import HybridInspectionResultsRepository
from backend.app.repositories.inspection_results_repository import InspectionResultsRepository
from backend.app.repositories.models_repository import ModelsRepository
from backend.app.repositories.reject_log_repository import RejectLogRepository
from backend.app.repositories.postgres.inspection_mirror_repository import PostgresInspectionMirrorRepository
from backend.app.repositories.postgres.users_repository import PostgresUsersRepository
from backend.app.repositories.sqlserver.inspection_mirror_repository import SqlServerInspectionMirrorRepository
from backend.app.repositories.sqlserver.users_repository import SqlServerUsersRepository
from backend.app.repositories.templates_repository import TemplatesRepository
from backend.app.repositories.users_repository import UsersRepository
from backend.app.repositories.workstation_registry_repository import WorkstationRegistryRepository
from backend.app.repositories.machine_settings_repository import MachineSettingsRepository
from backend.app.services.model_export_service import ModelExportService
from backend.app.services.inspection_session import InspectionSessionService
from backend.app.services.sticker_inference import StickerInferenceService
from backend.app.services.template_runtime import TemplateRuntimeService
from backend.app.workers.push_worker import PushWorker
from backend.app.services.plc_adapter import build_plc_adapter
from backend.app.workers.plc_worker import PlcWorker

_logger = logging.getLogger("backend.container")

# ── Config: env (secrets/bootstrap) + machine_settings.json (everything else) ──
app_config = AppConfig()
machine_settings_repo = MachineSettingsRepository()
machine_settings = machine_settings_repo.load_settings()
app_config.apply_machine_settings(machine_settings)
# The connection section that was actually used to build the PLC adapter. A later
# PUT that changes `connection` cannot be applied live — the route compares
# against this and reports `restart_required`.
boot_connection = machine_settings.connection

device_runtime = DeviceRuntimeResolver(app_config)
database_backend = app_config.database_backend
users_repo = (
    PostgresUsersRepository(app_config)
    if database_backend == "postgresql"
    else SqlServerUsersRepository(app_config)
    if database_backend == "sqlserver"
    else UsersRepository()
)
audit_repo = AuthAuditRepository()
templates_repo = TemplatesRepository(
    default_model_path=app_config.default_sticker_model_path,
    default_meta_path=app_config.default_sticker_model_meta_path,
)
deployments_repo = DeploymentsRepository()
reject_log_repo = RejectLogRepository()
models_repo = ModelsRepository(
    default_model_path=app_config.default_sticker_model_path,
    default_meta_path=app_config.default_sticker_model_meta_path,
)
local_inspection_results_repo = InspectionResultsRepository()
inspection_sql_mirror_repo = (
    PostgresInspectionMirrorRepository(app_config)
    if database_backend == "postgresql"
    else SqlServerInspectionMirrorRepository(app_config)
    if database_backend == "sqlserver"
    else None
)
inspection_results_repo = HybridInspectionResultsRepository(
    local_inspection_results_repo,
    inspection_sql_mirror_repo,
)

token_store = TokenStore(ttl_seconds=app_config.access_token_ttl_seconds)

template_runtime_service = TemplateRuntimeService(templates_repo, deployments_repo)
sticker_inference_service = StickerInferenceService(app_config, models_repo, device_runtime)
model_export_service = ModelExportService(models_repo, templates_repo, deployments_repo)

# ── PLC: adapter + worker entirely from machine_settings.connection / .io ──
plc_worker: PlcWorker | None = None
if boot_connection.enabled:
    plc_worker = PlcWorker(
        build_plc_adapter(boot_connection),
        num_channels=4,
        dry_run=boot_connection.dry_run,
    )
    plc_worker.apply_machine_settings(machine_settings)

inspection_session_service = InspectionSessionService(
    template_runtime_service,
    inspection_results_repo,
    sticker_inference_service,
    app_config=app_config,
    plc_worker=plc_worker,
    reject_log_repo=reject_log_repo,
)
workstation_registry_repo = WorkstationRegistryRepository()

def _log_startup_config() -> None:
    """Log the effective config (env + machine_settings.json) in one block."""
    _cfg = app_config
    _ms = machine_settings
    _lines = [
        "=== QC Suite Config ===",
        f"host: {_cfg.host}",
        f"port: {_cfg.port}",
        f"debug: {_cfg.debug}",
        f"local_only: {_cfg.local_only}",
        f"database_backend: {_cfg.database_backend}",
        "--- machine_settings.json ---",
        f"inference.mode: {_cfg.sticker_inference_mode}",
        f"inference.device: {_cfg.device_mode} (cuda_device_id={_cfg.cuda_device_id})",
        f"inference.default_model_path: {_cfg.default_sticker_model_path or '(empty — template vision.model_path is used)'}",
        f"inference.num_threads: {_cfg.inference_num_threads}",
        f"inference.timeout_s: {_cfg.inference_timeout_s}",
        f"timing.commit_grace_ms: {_cfg.commit_grace_ms}",
        f"timing.accept_stable_frames/ms: {_cfg.accept_stable_frames}/{_cfg.accept_stable_ms}",
        f"timing.part_ready_settle_ms_default: {_cfg.part_ready_settle_ms_default}",
        f"timing.part_ready_release_ms: {_cfg.part_ready_release_ms_default}",
        f"timing.reject_timeout_ms: {_cfg.reject_timeout_ms}",
        f"timing.inference_cache_ttl_ms: {_cfg.inference_cache_ttl_ms}",
        f"timing.session_idle_timeout_s: {_cfg.session_idle_timeout_s}",
        f"connection.enabled: {_ms.connection.enabled}",
        f"connection.dry_run: {_ms.connection.dry_run}",
        f"connection.transport: {_ms.connection.transport}",
        f"io.relay clamp/ok/enji: {_ms.io.relay_clamp_address}/{_ms.io.relay_ok_light_buzzer_address}/{_ms.io.relay_enji_buzzer_address}",
        f"io.input release/template/clamp_engaged: {_ms.io.input_release_address}/{_ms.io.input_template_address}/{_ms.io.input_clamp_engaged_address}",
        f"io.accept_pulse_ms: {_ms.io.accept_pulse_ms}",
        f"io.min_reclamp_interval_ms: {_ms.io.min_reclamp_interval_ms}",
        "=======================",
    ]
    for _line in _lines:
        _logger.info("[config] %s", _line)

_log_startup_config()

push_worker = PushWorker(
    inspection_results_repo,
    interval_seconds=int(app_config.push_worker_interval_seconds),
    max_retry_count=int(app_config.push_worker_max_retry),
)
