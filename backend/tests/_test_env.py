"""Isolated data root for the whole backend test run.

`backend.app.core.config` computes `DATA_ROOT` from `QC_SUITE_DATA_ROOT` **at import
time**, and `container.py` reads `machine_settings.json` from it. Whichever test
module imports `backend.app.*` first therefore decides where every test reads and
writes. Until 2026-09-18 that was `test_api_smoke.py` by alphabetical accident; a
new file sorting before it silently ran the suite against the developer's real
`data/` (`test_accept_stability.py` did exactly that).

`conftest.py` calls `ensure_test_data_root()` before any test module is imported.
Test modules that need the root call it too (idempotent), so they also work under
`python -m unittest`.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
from pathlib import Path

_MARKER = "QC_SUITE_TEST_DATA_ROOT"  # set by us, so we know the root is ours


def ensure_test_data_root() -> Path:
    """Point `QC_SUITE_DATA_ROOT` at a throwaway directory holding a test
    `machine_settings.json` (PLC off, inference `classic`, default model from
    `QC_SUITE_DEFAULT_STICKER_MODEL_PATH` when a developer exported one).

    Always overrides an inherited `QC_SUITE_DATA_ROOT`: tests must never touch a
    real data root.
    """
    existing = os.environ.get(_MARKER, "").strip()
    if existing and Path(existing).is_dir():
        os.environ["QC_SUITE_DATA_ROOT"] = existing
        return Path(existing)

    root = Path(tempfile.mkdtemp(prefix="qc-suite-tests-"))
    atexit.register(lambda: shutil.rmtree(root, ignore_errors=True))
    (root / "json_store").mkdir(parents=True, exist_ok=True)
    (root / "json_store" / "machine_settings.json").write_text(
        json.dumps({
            "version": 2,
            "connection": {"enabled": False, "dry_run": True},
            "inference": {
                "mode": "classic",
                "default_model_path": os.environ.get("QC_SUITE_DEFAULT_STICKER_MODEL_PATH", ""),
                "default_model_meta_path": os.environ.get("QC_SUITE_DEFAULT_STICKER_MODEL_META_PATH", ""),
            },
        }),
        encoding="utf-8",
    )
    os.environ["QC_SUITE_DATA_ROOT"] = str(root)
    os.environ[_MARKER] = str(root)
    return root
