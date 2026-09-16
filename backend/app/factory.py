from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from flask import Flask, jsonify

from backend.app.api.auth_routes import auth_blueprint
from backend.app.api.calibration_routes import calibration_blueprint
from backend.app.api.dashboard_routes import dashboard_blueprint
from backend.app.api.deployment_routes import deployment_blueprint
from backend.app.api.inspection_routes import inspection_blueprint
from backend.app.api.template_routes import template_blueprint
from backend.app.api.machine_settings_routes import machine_settings_blueprint
from backend.app.api.workstation_routes import workstation_blueprint
from backend.app.core.config import AppConfig, ensure_data_dirs
from backend.app.core.logging_config import configure_logging

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    ensure_data_dirs()
    config = AppConfig()
    # P3-8: Refuse to start in production with default secret key
    _default_secret = "qc-suite-dev-secret"
    if config.secret_key == _default_secret:
        if config.environment.lower() == "prod":
            raise RuntimeError(
                "Refusing to start in production with default secret_key. "
                "Set QC_SUITE_SECRET_KEY env var."
            )
        else:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Using default secret_key — OK for dev, but set QC_SUITE_SECRET_KEY for production."
            )
    app = Flask(__name__)
    app.config["QC_SUITE"] = config
    configure_logging(app)
    app.register_blueprint(auth_blueprint)
    app.register_blueprint(template_blueprint)
    app.register_blueprint(deployment_blueprint)
    app.register_blueprint(inspection_blueprint)
    app.register_blueprint(dashboard_blueprint)
    app.register_blueprint(calibration_blueprint)
    app.register_blueprint(workstation_blueprint)
    app.register_blueprint(machine_settings_blueprint)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "app": "qc-suite-python"})

    @app.get("/health/detailed")
    def health_detailed():
        from backend.app.core.container import app_config, inspection_results_repo, plc_worker, push_worker, token_store
        checks: dict[str, dict] = {}

        backend_name = app_config.database_backend

        # SQL mirror
        sql_mirror_active = inspection_results_repo._sql_mirror_repo is not None  # noqa: SLF001
        mirror_note = {
            "postgresql": "PostgreSQL mirror active",
            "sqlserver": "SQL Server mirror active",
        }.get(backend_name, "local-only mode")
        checks["sql_mirror"] = {"ok": sql_mirror_active, "note": mirror_note if sql_mirror_active else "local-only mode"}

        # Session store
        store_type = type(token_store).__name__
        checks["session_store"] = {"ok": True, "backend": store_type, "mode": backend_name}

        checks["database_backend"] = {"ok": True, "backend": backend_name}

        model_path = str(app_config.default_sticker_model_path or "").strip()
        model_exists = bool(model_path and Path(model_path).exists())
        checks["sticker_model"] = {
            "ok": model_exists or app_config.sticker_inference_mode in {"classic", "auto"},
            "path": model_path or None,
            "exists": model_exists,
            "mode": app_config.sticker_inference_mode,
        }

        checks["ocr_runtime"] = {
            "ok": True,
            "mode": "disabled",
            "engine": "disabled",
            "required": False,
            "note": "OCR removed",
        }

        # Push worker
        worker_alive = push_worker._thread is not None and push_worker._thread.is_alive()  # noqa: SLF001
        checks["push_worker"] = {"ok": worker_alive, "note": "running" if worker_alive else "not started"}

        # Pending pushes
        try:
            pending_count = len(
                inspection_results_repo._local_repo.list_results(push_status="pending", limit=1000, offset=0)  # noqa: SLF001
            )
            dead_count = len(
                inspection_results_repo._local_repo.list_results(push_status="dead_letter", limit=1000, offset=0)  # noqa: SLF001
            )
            checks["push_queue"] = {"pending": pending_count, "dead_letter": dead_count}
        except Exception as exc:  # noqa: BLE001
            checks["push_queue"] = {"error": str(exc)}

        # PLC worker
        if plc_worker is not None:
            plc_status = plc_worker.status()
            checks["plc_worker"] = {"ok": plc_status.get("running", False), **plc_status}
        else:
            checks["plc_worker"] = {"ok": True, "note": "disabled"}

        all_ok = all(v.get("ok", True) for v in checks.values() if isinstance(v, dict) and "ok" in v)
        return jsonify({
            "status": "ok" if all_ok else "degraded",
            "timestamp": datetime.now(UTC).isoformat(),
            "app": "qc-suite-python",
            "checks": checks,
        }), 200 if all_ok else 207

    _register_worker_lifecycle(app)
    return app


def _register_worker_lifecycle(app: Flask) -> None:
    """Start push worker and WebSocket streaming server on first request."""
    _started = [False]

    @app.before_request
    def _start_workers():
        if _started[0]:
            return
        _started[0] = True
        try:
            from backend.app.core.container import push_worker
            push_worker.start()
            logger.info("[factory] push worker started")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[factory] push worker failed to start: %s", exc)

        try:
            from backend.app.core.container import plc_worker
            if plc_worker is not None:
                plc_worker.start()
                logger.info("[factory] plc worker started")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[factory] plc worker failed to start: %s", exc)

        try:
            from backend.app.streaming.server import start as start_stream_server
            config = app.config["QC_SUITE"]
            if not getattr(config, "local_only", False):
                stream_host = config.stream_host or config.host
                start_stream_server(host=stream_host, port=config.stream_port)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[factory] streaming server failed to start: %s", exc)

    @app.teardown_appcontext
    def _stop_workers(_exc=None):
        pass  # process-level teardown is handled by backend.app.core.shutdown
