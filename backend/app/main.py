from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from backend.app.core.shutdown import install_process_shutdown_handlers
from backend.app.factory import create_app


app = create_app()


if __name__ == "__main__":
    install_process_shutdown_handlers()
    config = app.config["QC_SUITE"]
    app.run(host=config.host, port=config.port, debug=config.debug)
