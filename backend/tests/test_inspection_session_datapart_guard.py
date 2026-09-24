"""InspectionSessionService._maybe_persist -> DatapartGuardService.record_pushed
hook: every accepted judgement that gets a "sent" SQL push must report its
sql_mirror_id to the guard so it can count toward the 5-judgement batch;
a push that fails/stays pending must NOT be counted (there's no confirmed
mirror row to ever check DatapartID against).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.tests._test_env import ensure_test_data_root  # noqa: E402

ensure_test_data_root()

from backend.app.core.config import AppConfig  # noqa: E402
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


class _FakeResultsRepo:
    """Simulates HybridInspectionResultsRepository.create_result: each call
    returns push_status/sql_mirror_id from a scripted queue."""

    def __init__(self, scripted_records: list[dict[str, Any]]) -> None:
        self._queue = list(scripted_records)
        self._next_id = 1
        self.created: list[dict[str, Any]] = []

    def create_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        extra = self._queue.pop(0) if self._queue else {"push_status": "sent", "sql_mirror_id": self._next_id}
        record = {**payload, "id": self._next_id, **extra}
        self._next_id += 1
        self.created.append(record)
        return record


class _FakeGuard:
    def __init__(self) -> None:
        self.calls: list[int | None] = []

    def record_pushed(self, mirror_id: int | None) -> None:
        self.calls.append(mirror_id)


def _sample_validation() -> dict[str, Any]:
    return {
        "decision": "ACCEPT",
        "decision_code": "ACCEPT",
        "reject_reason_code": None,
        "part_name": "P1",
        "expected_class": "K0W-HB0",
        "detected_class": "K0W-HB0",
        "sticker_confidence": 0.9,
        "data1": 0.9,
        "data2": 0.9,
        "operator_user_id": 2,
        "mp_check": "operator",
        "line_id": None,
        "station_id": None,
        "validation_details": {},
        "sticker_bbox": None,
        "sticker_backend": "classic",
        "text_bbox": None,
        "dot_bbox": None,
        "dot_position": None,
        "anchor_offset": None,
        "pose_angle": None,
        "targets": [],
    }


class MaybePersistDatapartGuardHookTest(unittest.TestCase):
    def _build_service(self, results_repo, guard) -> tuple[InspectionSessionService, Any]:
        template = template_from_dict(_minimal_sticker_payload())
        service = InspectionSessionService(
            _StubTemplateRuntime(template),
            results_repo=results_repo,
            sticker_inference=None,
            app_config=AppConfig(),
            datapart_guard=guard,
        )
        return service

    def _common_kwargs(self) -> dict[str, Any]:
        return dict(
            part_ready={"status": "ready"},
            part_ready_roi_meta={"x": 0, "y": 0, "width": 1, "height": 1},
            sticker_roi_meta={"x": 0, "y": 0, "width": 1, "height": 1},
            sticker_detection={},
        )

    def test_reports_sql_mirror_id_when_push_succeeds(self) -> None:
        guard = _FakeGuard()
        results_repo = _FakeResultsRepo([{"push_status": "sent", "sql_mirror_id": 42}])
        service = self._build_service(results_repo, guard)

        service._maybe_persist(
            _sample_validation(), _fake_state(),
            event_id="evt-00001", **self._common_kwargs(),
        )

        self.assertEqual(guard.calls, [42])

    def test_does_not_report_when_push_pending(self) -> None:
        guard = _FakeGuard()
        results_repo = _FakeResultsRepo([{"push_status": "pending"}])
        service = self._build_service(results_repo, guard)

        service._maybe_persist(
            _sample_validation(), _fake_state(),
            event_id="evt-00001", **self._common_kwargs(),
        )

        self.assertEqual(guard.calls, [])

    def test_does_not_report_when_push_failed(self) -> None:
        guard = _FakeGuard()
        results_repo = _FakeResultsRepo([{"push_status": "failed"}])
        service = self._build_service(results_repo, guard)

        service._maybe_persist(
            _sample_validation(), _fake_state(),
            event_id="evt-00001", **self._common_kwargs(),
        )

        self.assertEqual(guard.calls, [])

    def test_no_guard_configured_does_not_raise(self) -> None:
        results_repo = _FakeResultsRepo([{"push_status": "sent", "sql_mirror_id": 1}])
        service = self._build_service(results_repo, guard=None)

        result = service._maybe_persist(
            _sample_validation(), _fake_state(),
            event_id="evt-00001", **self._common_kwargs(),
        )

        self.assertTrue(result["written"])


def _fake_state():
    from backend.app.models.session_state import SessionState

    template = template_from_dict(_minimal_sticker_payload())
    return SessionState(
        session_id="s1",
        client_id="c1",
        camera_index=0,
        template=template,
    )


if __name__ == "__main__":
    unittest.main()
