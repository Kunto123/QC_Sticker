"""Regression for InspectionSessionService._maybe_persist's dedup key.

Real-world symptom: within one continuous session, the first accepted
judgement was written to inspection_results.json (and pushed to SQL), but
every later accepted judgement for the same part/decision was silently
dropped — the in-session "Accepted" counter kept incrementing (it doesn't go
through this dedup), but no new row ever appeared.

Root cause: _maybe_persist built its dedup key from state.current_event_id,
but by the time it runs, the caller's "full cycle reset" (right after a
commit) has already set state.current_event_id back to None for the *next*
cycle. So persist_key collapsed to the same "event:ACCEPT:OK:<part_name>"
string for every commit sharing decision+part_name — the second commit's key
matched state.last_persisted_key from the first, tripping the
"duplicate_event" guard and returning written=False without writing
anything, even though it was a genuinely new physical part/event.

Fix: build persist_key from the event_id *parameter* (the id of the event
actually being committed, captured before the reset), not from the
already-reset state.current_event_id.
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
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self._next_id = 1

    def create_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = {**payload, "id": self._next_id}
        self._next_id += 1
        self.created.append(record)
        return record


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


class MaybePersistDedupKeyTest(unittest.TestCase):
    def _build_service_and_state(self):
        results_repo = _FakeResultsRepo()
        template = template_from_dict(_minimal_sticker_payload())
        service = InspectionSessionService(
            _StubTemplateRuntime(template),
            results_repo=results_repo,
            sticker_inference=None,
            app_config=AppConfig(),
        )
        session_payload = service.start_session(
            client_id="client-1", camera_index=0, template_version_id=1
        )
        state = service._sessions[session_payload["session_id"]]
        return service, state, results_repo

    def test_second_commit_with_same_decision_and_part_is_not_treated_as_duplicate(self) -> None:
        service, state, results_repo = self._build_service_and_state()
        common_kwargs = dict(
            part_ready={"status": "ready"},
            part_ready_roi_meta={"x": 0, "y": 0, "width": 1, "height": 1},
            sticker_roi_meta={"x": 0, "y": 0, "width": 1, "height": 1},
            sticker_detection={},
        )

        # Simulate what process_frame_decoded does around each commit: the
        # "full cycle reset" already ran (current_event_id back to None)
        # before _maybe_persist is called, exactly like production.
        state.current_event_id = None
        first = service._maybe_persist(
            _sample_validation(), state, event_id="evt-00001", **common_kwargs
        )
        state.current_event_id = None
        second = service._maybe_persist(
            _sample_validation(), state, event_id="evt-00002", **common_kwargs
        )

        self.assertEqual(first["written"], True)
        self.assertEqual(second["written"], True, f"second commit was dropped: {second}")
        self.assertEqual(len(results_repo.created), 2)

    def test_replaying_the_same_event_id_is_still_deduped(self) -> None:
        """The dedup guard itself must still work for a genuine re-send of
        the same event (e.g. a retried frame for an already-committed event)."""
        service, state, results_repo = self._build_service_and_state()
        common_kwargs = dict(
            part_ready={"status": "ready"},
            part_ready_roi_meta={"x": 0, "y": 0, "width": 1, "height": 1},
            sticker_roi_meta={"x": 0, "y": 0, "width": 1, "height": 1},
            sticker_detection={},
        )

        state.current_event_id = None
        first = service._maybe_persist(
            _sample_validation(), state, event_id="evt-00001", **common_kwargs
        )
        state.current_event_id = None
        replay = service._maybe_persist(
            _sample_validation(), state, event_id="evt-00001", **common_kwargs
        )

        self.assertEqual(first["written"], True)
        self.assertEqual(replay["written"], False)
        self.assertEqual(replay["reason"], "duplicate_event")
        self.assertEqual(len(results_repo.created), 1)

    def test_persisted_record_carries_template_name_for_sql_push_partname(self) -> None:
        """PartName pushed to SQL is sourced from the template's own name
        (see build_sql_payload in the mirror repos) — _maybe_persist must
        include it on the local record it hands to create_result."""
        service, state, results_repo = self._build_service_and_state()
        common_kwargs = dict(
            part_ready={"status": "ready"},
            part_ready_roi_meta={"x": 0, "y": 0, "width": 1, "height": 1},
            sticker_roi_meta={"x": 0, "y": 0, "width": 1, "height": 1},
            sticker_detection={},
        )

        state.current_event_id = None
        service._maybe_persist(_sample_validation(), state, event_id="evt-00001", **common_kwargs)

        self.assertEqual(len(results_repo.created), 1)
        self.assertEqual(results_repo.created[0]["template_name"], "Sticker Template")


if __name__ == "__main__":
    unittest.main()
