from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.shutdown import install_process_shutdown_handlers
from backend.app.main import app


if __name__ == "__main__":
    install_process_shutdown_handlers()
    config = app.config["QC_SUITE"]
    app.run(host=config.host, port=config.port, debug=config.debug)
