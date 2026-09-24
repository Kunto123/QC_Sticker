"""line_id fallback for InspectionSessionService.start_session.

The desktop client never sends line_id when starting a session (line/station
slots were removed 2026-09-18), so every persisted/pushed inspection result
had a permanently null "Line" column. `machine_settings.json -> identity.line`
existed to back this but was never wired anywhere. Covers:
- start_session falls back to app_config.machine_line_id when no line_id given
- an explicit line_id from the caller still wins
- InspectionSessionService.apply_machine_settings updates the fallback live
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.tests._test_env import ensure_test_data_root  # noqa: E402

ensure_test_data_root()

from backend.app.core.config import AppConfig  # noqa: E402
from backend.app.models.machine_settings import MachineSettings  # noqa: E402
from backend.app.services.inspection_session import InspectionSessionService  # noqa: E402
from shared.contracts.templates import template_from_dict  # noqa: E402


def _minimal_sticker_payload() -> dict:
    return {
        "id": 1,
        "version_id": 1,
        "version_number": 1,
        "name": "Sticker Template",
        "description": "",
        "is_active": True,
        "camera": {"camera_index": 0},
        "part_ready_roi": {"x": 0.2, "y": 0.2, "w": 0.5, "h": 0.5},
        "sticker_roi": {"x": 0.1, "y": 0.1, "w": 0.6, "h": 0.6},
        "vision": {"model_path": "models/sticker.pt", "conf_threshold": 0.3},
        "part_ready": {"enabled": True, "method": "gap_template_match"},
        "sticker": {
            "part_name": "P1",
            "expected_class": "K0W-HB0",
            "enabled": True,
            "validator_mode": "ml_detection",
            "min_roi_confidence": 0.25,
        },
        "persistence": {"write_to_db": True},
    }


class _StubTemplateRuntime:
    def __init__(self, template) -> None:
        self._template = template

    def resolve_template_by_version(self, version_id: int):
        return self._template


def _build_service(app_config: AppConfig) -> InspectionSessionService:
    template = template_from_dict(_minimal_sticker_payload())
    return InspectionSessionService(
        _StubTemplateRuntime(template),
        results_repo=None,
        sticker_inference=None,
        app_config=app_config,
    )


class SessionLineIdFallbackTest(unittest.TestCase):
    def test_falls_back_to_configured_machine_line_id(self) -> None:
        cfg = AppConfig()
        cfg.machine_line_id = "LINE-7"
        service = _build_service(cfg)

        payload = service.start_session(
            client_id="client-1", camera_index=0, template_version_id=1
        )

        self.assertEqual(payload["line_id"], "LINE-7")

    def test_explicit_line_id_still_wins_over_configured_default(self) -> None:
        cfg = AppConfig()
        cfg.machine_line_id = "LINE-7"
        service = _build_service(cfg)

        payload = service.start_session(
            client_id="client-1", camera_index=0, template_version_id=1, line_id="LINE-EXPLICIT"
        )

        self.assertEqual(payload["line_id"], "LINE-EXPLICIT")

    def test_stays_null_when_machine_line_id_unset(self) -> None:
        cfg = AppConfig()
        cfg.machine_line_id = ""
        service = _build_service(cfg)

        payload = service.start_session(
            client_id="client-1", camera_index=0, template_version_id=1
        )

        self.assertIsNone(payload["line_id"])

    def test_apply_machine_settings_updates_fallback_live(self) -> None:
        cfg = AppConfig()
        cfg.machine_line_id = ""
        service = _build_service(cfg)

        settings = MachineSettings()
        settings.identity.line = "LINE-9"
        service.apply_machine_settings(settings)

        payload = service.start_session(
            client_id="client-1", camera_index=0, template_version_id=1
        )

        self.assertEqual(payload["line_id"], "LINE-9")


if __name__ == "__main__":
    unittest.main()
