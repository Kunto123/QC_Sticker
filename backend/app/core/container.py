from __future__ import annotations

import logging

from backend.app.core.config import AppConfig
from backend.app.core.device_runtime import DeviceRuntimeResolver
from backend.app.repositories.auth_audit_repository import AuthAuditRepository
from backend.app.repositories.filesystem.storage_repository import FilesystemStorageRepository
from backend.app.core.security import TokenStore
from backend.app.repositories.augment_repository import AugmentRepository
from backend.app.repositories.dataset_versions_repository import DatasetVersionRepository
from backend.app.repositories.datasets_repository import DatasetsRepository
from backend.app.repositories.deployments_repository import DeploymentsRepository
from backend.app.repositories.hybrid_inspection_results_repository import HybridInspectionResultsRepository
from backend.app.repositories.inspection_results_repository import InspectionResultsRepository
from backend.app.repositories.models_repository import ModelsRepository
from backend.app.repositories.profiles_repository import ProfilesRepository
from backend.app.repositories.reject_log_repository import RejectLogRepository
from backend.app.repositories.postgres.inspection_mirror_repository import PostgresInspectionMirrorRepository
from backend.app.repositories.postgres.users_repository import PostgresUsersRepository
from backend.app.repositories.sqlserver.inspection_mirror_repository import SqlServerInspectionMirrorRepository
from backend.app.repositories.sqlserver.users_repository import SqlServerUsersRepository
from backend.app.repositories.templates_repository import TemplatesRepository
from backend.app.repositories.training_repository import TrainingRepository
from backend.app.repositories.users_repository import UsersRepository
from backend.app.repositories.workstation_registry_repository import WorkstationRegistryRepository
from backend.app.repositories.machine_settings_repository import MachineSettingsRepository
from backend.app.services.model_export_service import ModelExportService
from backend.app.services.inspection_session import InspectionSessionService
from backend.app.services.sticker_inference import StickerInferenceService
from backend.app.services.template_runtime import TemplateRuntimeService
from backend.app.services.training import TrainingService
from backend.app.workers.push_worker import PushWorker
from backend.app.services.plc_adapter import build_plc_adapter_from_connection
from backend.app.workers.plc_worker import PlcWorker


app_config = AppConfig()
device_runtime = DeviceRuntimeResolver(app_config)
filesystem_storage_repo = FilesystemStorageRepository()
database_backend = app_config.database_backend
users_repo = (
    PostgresUsersRepository(app_config)
    if database_backend == "postgresql"
    else SqlServerUsersRepository(app_config)
    if database_backend == "sqlserver"
    else UsersRepository()
)
audit_repo = AuthAuditRepository()
templates_repo = TemplatesRepository()
deployments_repo = DeploymentsRepository()
profiles_repo = ProfilesRepository()
datasets_repo = DatasetsRepository()
reject_log_repo = RejectLogRepository()
dataset_versions_repo = DatasetVersionRepository(
    datasets_repo,
    geometric_augment_enabled=app_config.geometric_augment_enabled,
)
models_repo = ModelsRepository()
training_repo = TrainingRepository()
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

# ── Machine settings: env seeds the DB once (fresh PC), after that the DB wins ──
# Everything PLC/timing below is built from _machine_settings, not app_config.
# app_config.plc_* / timing fields are only read by seed_from_env.
machine_settings_repo = MachineSettingsRepository()
machine_settings_repo.seed_from_env(app_config, force=False)
_machine_settings = machine_settings_repo.load_settings()
# Snapshot of the connection the adapter was built with. PUT /machine-settings
# compares against this to tell the UI a restart is needed.
boot_connection = _machine_settings.connection

_plc_adapter = build_plc_adapter_from_connection(_machine_settings.connection)
plc_worker: PlcWorker | None = (
    PlcWorker(
        _plc_adapter,
        accept_pulse_ms=_machine_settings.sticker.accept_pulse_ms,
        num_channels=4,
        input_release_address=_machine_settings.sticker.input_release_address,
        input_template_address=_machine_settings.sticker.input_template_address,
        input_clamp_engaged_address=_machine_settings.sticker.input_clamp_engaged_address,
        clamp_feedback_enabled=_machine_settings.sticker.clamp_feedback_enabled,
        relay_clamp_address=_machine_settings.sticker.relay_clamp_address,
        relay_ok_light_buzzer_address=_machine_settings.sticker.relay_ok_light_buzzer_address,
        relay_enji_buzzer_address=_machine_settings.sticker.relay_enji_buzzer_address,
    )
    if _machine_settings.connection.enabled
    else None
)
if plc_worker is not None:
    # dry_run is set exactly once here — it describes the adapter above.
    plc_worker.configure_guards(
        min_reclamp_interval_ms=_machine_settings.sticker.min_reclamp_interval_ms,
        release_input_debounce_ms=_machine_settings.sticker.release_input_debounce_ms,
        dry_run=_machine_settings.connection.dry_run,
    )
    plc_worker.apply_machine_settings(_machine_settings)
    logging.getLogger("backend.container").info(
        "[container] PLC worker strategy: %s (from MachineSettings DB)",
        plc_worker.status().get("strategy", "none"),
    )

inspection_session_service = InspectionSessionService(
    template_runtime_service,
    profiles_repo,
    inspection_results_repo,
    sticker_inference_service,
    app_config=app_config,
    plc_worker=plc_worker,
    reject_log_repo=reject_log_repo,
)
inspection_session_service.apply_machine_settings(_machine_settings)
training_service = TrainingService(training_repo, models_repo, device_runtime, app_config=app_config)
workstation_registry_repo = WorkstationRegistryRepository()
augment_repo = AugmentRepository()

from backend.app.workers.augment_worker import AugmentWorker  # noqa: E402


def _log_startup_config() -> None:
    """Log semua config kritis satu blok saat service start.

    Nilai PLC dan timing diambil dari MachineSettings (yang benar-benar dipakai),
    bukan dari env — env hanya berperan saat seed pertama.
    """
    _cfg = app_config
    _ms = _machine_settings
    _logger = logging.getLogger("backend.startup")
    _lines = [
        "=== QC Suite Config ===",
        f"host: {_cfg.host}",
        f"port: {_cfg.port}",
        f"debug: {_cfg.debug}",
        f"database_backend: {_cfg.database_backend}",
        f"sticker_inference_mode: {_cfg.sticker_inference_mode}",
        f"default_sticker_model_path: {_cfg.default_sticker_model_path or '(empty — auto-discover from template)'}",
        f"inspect_hard_reject_reasons: {_cfg.inspect_hard_reject_reasons}",
        f"inference_num_threads: {_cfg.inference_num_threads}",
        f"inference_timeout_s: {_cfg.inference_timeout_s}",
        f"inference_interval_ms: {_cfg.inference_interval_ms}",
        f"device_mode: {_cfg.device_mode}",
        f"cuda_device_id: {_cfg.cuda_device_id}",
        f"local_only: {_cfg.local_only}",
        f"access_logs_enabled: {_cfg.access_logs_enabled}",
        f"--- machine_settings ({'seeded from env' if _ms.seeded_from_env else 'user-edited'} @ {_ms.seeded_at or '-'}) ---",
        f"plc_enabled: {_ms.connection.enabled}",
        f"plc_dry_run: {_ms.connection.dry_run}",
        f"plc_transport: {_ms.connection.transport}",
        f"plc_min_reclamp_interval_ms: {_ms.sticker.min_reclamp_interval_ms}",
        f"plc_release_input_debounce_ms: {_ms.sticker.release_input_debounce_ms}",
        f"plc_clamp_feedback_enabled: {_ms.sticker.clamp_feedback_enabled}",
        f"commit_grace_ms: {_ms.timing.commit_grace_ms}",
        f"accept_stable_ms: {_ms.timing.accept_stable_ms}",
        f"accept_stable_frames: {_ms.timing.accept_stable_frames}",
        f"hard_reject_stable_ms: {_ms.timing.hard_reject_stable_ms}",
        f"hard_reject_stable_frames: {_ms.timing.hard_reject_stable_frames}",
        f"part_ready_settle_ms_default: {_ms.timing.part_ready_settle_ms_default}",
        f"part_ready_release_ms: {_ms.timing.part_ready_release_ms}",
        f"reject_timeout_ms: {_ms.timing.reject_timeout_ms}",
        f"inference_cache_ttl_ms: {_ms.timing.inference_cache_ttl_ms}",
        f"session_idle_timeout_s: {_ms.timing.session_idle_timeout_s}",
        "=======================",
    ]
    for _line in _lines:
        _logger.info("[config] %s", _line)


_log_startup_config()

_augment_worker = AugmentWorker(augment_repo, datasets_repo)
_augment_worker.start()

push_worker = PushWorker(
    inspection_results_repo,
    interval_seconds=int(app_config.push_worker_interval_seconds),
    max_retry_count=int(app_config.push_worker_max_retry),
)
