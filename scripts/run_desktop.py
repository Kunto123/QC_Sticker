from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from backend.app.core.config import AppConfig


def _backend_health_url(config: AppConfig) -> str:
    host = (config.host or "127.0.0.1").strip() or "127.0.0.1"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    return f"http://{host}:{config.port}/health"


def _wait_for_backend_ready(health_url: str, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(health_url, timeout=2):
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"Backend did not become ready at {health_url}: {last_error}")


def _start_backend_process() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_backend.py")],
        cwd=str(PROJECT_ROOT),
    )


def _stop_backend_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch QC Suite desktop with one command.")
    parser.add_argument(
        "--split",
        action="store_true",
        help="Force split deployment mode and start a backend subprocess before the UI.",
    )
    args = parser.parse_args()

    config = AppConfig()
    split_mode = bool(args.split or not config.local_only)

    if not split_mode:
        from backend.app.main import app as _backend_app  # noqa: F401
        from client_tk.app.main import launch

        launch()
        return 0

    os.environ["QC_SUITE_LOCAL_ONLY"] = "0"
    backend_url = _backend_health_url(config).removesuffix("/health")
    os.environ["QC_SUITE_SERVER_URL"] = backend_url

    backend_process = _start_backend_process()
    try:
        _wait_for_backend_ready(f"{backend_url}/health")

        from client_tk.app.main import launch

        launch()
        return 0
    finally:
        _stop_backend_process(backend_process)


if __name__ == "__main__":
    raise SystemExit(main())