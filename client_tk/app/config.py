from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip() or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip() or default)
    except (TypeError, ValueError):
        return default


DEFAULT_LOCAL_ONLY = os.getenv("QC_SUITE_LOCAL_ONLY", "1").strip() != "0"
DEFAULT_SERVER_URL = os.getenv(
    "QC_SUITE_SERVER_URL",
    "local://embedded" if DEFAULT_LOCAL_ONLY else "http://127.0.0.1:8100",
)
DEFAULT_UPLOAD_INTERVAL_MS = int(os.getenv("QC_SUITE_UPLOAD_INTERVAL_MS", "150"))
DEFAULT_OPERATOR_PREVIEW_FPS = max(1.0, _env_float("QC_SUITE_OPERATOR_PREVIEW_FPS", 20.0))
DEFAULT_CAMERA_WIDTH = max(0, _env_int("QC_SUITE_CAMERA_WIDTH", 0))
DEFAULT_CAMERA_HEIGHT = max(0, _env_int("QC_SUITE_CAMERA_HEIGHT", 0))
DEFAULT_CAMERA_FPS = max(0.0, _env_float("QC_SUITE_CAMERA_FPS", 0.0))
# WebSocket streaming is disabled by default in local-only mode.
DEFAULT_STREAM_URL = os.getenv(
    "QC_SUITE_STREAM_URL",
    "" if DEFAULT_LOCAL_ONLY else "ws://127.0.0.1:8101",
)

