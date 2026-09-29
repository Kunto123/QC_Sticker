from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from flask import Flask, g, request


def configure_logging(app: Flask) -> None:
    """Siapkan structured logging dan correlation ID per-request.

    Perilakunya dikontrol dua flag dari env di AppConfig:

    * ``access_logs_enabled`` (QC_SUITE_ACCESS_LOGS_ENABLED, default 1):
        - ON  → tiap request dicatat di level INFO (perilaku lama).
        - OFF → cuma request dengan status >= 400 dicatat di WARNING; 2xx/3xx tidak dicatat.

    * ``werkzeug_logs_enabled`` (QC_SUITE_WERKZEUG_REQUEST_LOGS_ENABLED, default 1):
        - ON  → werkzeug tetap pakai logging INFO default-nya.
        - OFF → level logger werkzeug dinaikkan ke ERROR (tidak ada baris per-request).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Baca config lebih awal; AppConfig sudah di-set di app.config sebelum fungsi ini dipanggil.
    qc_config = app.config.get("QC_SUITE")
    access_logs_enabled: bool = getattr(qc_config, "access_logs_enabled", True)
    werkzeug_logs_enabled: bool = getattr(qc_config, "werkzeug_logs_enabled", True)

    if not werkzeug_logs_enabled:
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

    _access_logger = logging.getLogger("qc.access")

    @app.before_request
    def _assign_correlation_id():
        g.correlation_id = (
            request.headers.get("X-Correlation-Id")
            or request.headers.get("X-Request-Id")
            or uuid.uuid4().hex
        )
        g.request_start = datetime.now(UTC)

    @app.after_request
    def _log_request(response):
        cid = getattr(g, "correlation_id", "-")
        start = getattr(g, "request_start", None)
        duration_ms = (
            round((datetime.now(UTC) - start).total_seconds() * 1000, 1)
            if start else -1
        )
        status_code = response.status_code
        if access_logs_enabled:
            _access_logger.info(
                "method=%s path=%s status=%d duration_ms=%.1f cid=%s",
                request.method, request.path, status_code, duration_ms, cid,
            )
        elif status_code >= 400:
            _access_logger.warning(
                "method=%s path=%s status=%d duration_ms=%.1f cid=%s",
                request.method, request.path, status_code, duration_ms, cid,
            )
        # else: 2xx/3xx tidak dicatat di quiet mode
        response.headers["X-Correlation-Id"] = cid
        return response
