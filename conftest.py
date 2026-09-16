"""Root pytest configuration for the qc-suite-python backend test suite.

Goals:
- Make ``python -m pytest backend/tests`` work from the repo root without every
  test file having to hand-roll ``sys.path`` insertion (they still do, harmlessly).
- Seed environment defaults that the app factory reads at import time so the
  model registry is non-empty in a bare CI/dev checkout (see
  ``backend/app/repositories/models_repository.py`` — an empty
  ``QC_SUITE_DEFAULT_STICKER_MODEL_PATH`` seeds an EMPTY registry).
- Register custom markers used to quarantine tests that need real infrastructure
  (a production-trained sticker model, PLC hardware, a real DB) which is not
  present in the test environment.

IMPORTANT: os.environ mutations here run at conftest import, which pytest loads
BEFORE collecting/importing test modules. test_api_smoke.py sets a few env vars
at its own module top; we only set defaults with setdefault so we never clobber
an explicit value a developer/CI exported.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _seed_default_sticker_model_env() -> None:
    """Point the seeded model registry at a real .pt file if one ships in-repo.

    ``models_repository._default_models_payload`` returns an EMPTY registry when
    ``QC_SUITE_DEFAULT_STICKER_MODEL_PATH`` is blank. A generic YOLO weight file
    is enough to make the registry non-empty (test_00b) and to give the classic
    sticker pipeline a model_path — it is NOT enough to make the pipeline detect
    the synthetic test sticker (that needs the real 'AKH Sticker Detector'
    weights, which live outside the repo). Tests that need actual detection are
    marked ``requires_real_sticker_model`` and skipped; see HANDOFF.md.
    """
    if os.environ.get("QC_SUITE_DEFAULT_STICKER_MODEL_PATH", "").strip():
        return
    for candidate in ("yolov5su.pt", "yolo11n.pt", "yolo11n.pt"):
        weight = PROJECT_ROOT / candidate
        if weight.exists():
            os.environ["QC_SUITE_DEFAULT_STICKER_MODEL_PATH"] = str(weight)
            return


_seed_default_sticker_model_env()


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_real_sticker_model: needs the production-trained sticker "
        "model (outside repo) to detect the synthetic test image; skipped in a "
        "bare checkout. See HANDOFF.md.",
    )
    config.addinivalue_line(
        "markers",
        "requires_plc_hardware: needs real Modbus/serial PLC hardware; skipped.",
    )
    config.addinivalue_line(
        "markers",
        "suspected_regression: pinned failing test kept RED on purpose to flag a "
        "possible production regression. See HANDOFF.md.",
    )
