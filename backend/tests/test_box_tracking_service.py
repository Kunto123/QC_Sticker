"""Unit tests for box_tracking_service.py against the local JSON-backed
repositories (no live DB needed). See CLAUDE.md: tests must never set
QC_SUITE_DATA_ROOT themselves — the shared root comes from conftest.py."""
from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from backend.tests._test_env import ensure_test_data_root

ensure_test_data_root()

from backend.app.models.session_state import SessionState
from backend.app.repositories.box_luggage_repository import BoxLuggageRepository
from backend.app.repositories.ok_data_part_repository import OkDataPartRepository
from backend.app.services.box_tracking_service import (
    BoxTrackingError,
    BoxTrackingService,
    parse_datapart_scan,
    to_int_qty,
)
from backend.app.services.inspection_session import InspectionSessionService
from shared.contracts.enums import DecisionCode, SessionStatus
from shared.contracts.templates import (
    BoxTrackingConfig,
    CameraDefaults,
    InspectionTemplate,
    PartReadyConfig,
    PersistenceConfig,
    RoiGeometry,
    StickerRule,
    VisionConfig,
)

SCAN_1 = "8125B-K2S -H3XX|1867490|5|B09SB1100748292049|QRTag"
SCAN_2 = "8125B-K2S -H3XX|1867490|5|GM1SL1202609210714450009E|QRTag"
SCAN_2_MISMATCH = "8125B-K2S -H3XX|9999999|5|GM1SL1202609210714450009E|QRTag"


@dataclass
class _FakeBoxTracking:
    enabled: bool = True
    min_age_hours: float = 1.0


@dataclass
class _FakeTemplate:
    name: str = "Sticker Template"
    box_tracking: _FakeBoxTracking = field(default_factory=_FakeBoxTracking)


@dataclass
class _FakeIdentity:
    line: str = "A1"


@dataclass
class _FakeMachineSettings:
    identity: _FakeIdentity = field(default_factory=_FakeIdentity)


class ParseScanTest(unittest.TestCase):
    def test_valid_scan_parses(self) -> None:
        r = parse_datapart_scan(SCAN_1)
        self.assertEqual(r["part_number"], "8125B-K2S -H3XX")
        self.assertEqual(r["company_id"], "1867490")
        self.assertEqual(r["qty_box"], "5")
        self.assertEqual(r["id_data_part"], "B09SB1100748292049")

    def test_wrong_field_count_raises(self) -> None:
        with self.assertRaises(BoxTrackingError) as ctx:
            parse_datapart_scan("a|b|c")
        self.assertEqual(ctx.exception.code, "INVALID_SCAN")

    def test_non_numeric_qty_raises(self) -> None:
        with self.assertRaises(BoxTrackingError):
            parse_datapart_scan("A|B|not-a-number|C|QRTag")

    def test_to_int_qty_rejects_zero(self) -> None:
        with self.assertRaises(BoxTrackingError):
            to_int_qty("0")


class BoxTrackingServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.box_repo = BoxLuggageRepository()
        self.box_repo.save([])
        self.ok_repo = OkDataPartRepository()
        self.ok_repo.save([])
        self.service = BoxTrackingService(self.box_repo, self.ok_repo, app_config=None, machine_settings=_FakeMachineSettings())
        self.template = _FakeTemplate()

    def _seed_ok_data_part(self, *, id_data_part: str, hours_old: float) -> None:
        prod_date = datetime.now(UTC) - timedelta(hours=hours_old)
        self.ok_repo.seed_row(
            {
                "No": 1,
                "NamaPart": "x",
                "ProdDate": prod_date.isoformat(),
                "IDDataPart": id_data_part,
                "QtyBox": 5,
                "MPID": "M1",
                "MCID": "",
                "Station": "",
            }
        )

    def test_start_box_not_found_in_ok_data_part(self) -> None:
        with self.assertRaises(BoxTrackingError) as ctx:
            self.service.start_box(template=self.template, raw_scan=SCAN_1, username="alice")
        self.assertEqual(ctx.exception.code, "NOT_FOUND")

    def test_age_gate_denies_when_too_young(self) -> None:
        self._seed_ok_data_part(id_data_part="B09SB1100748292049", hours_old=0.5)
        with self.assertRaises(BoxTrackingError) as ctx:
            self.service.start_box(template=self.template, raw_scan=SCAN_1, username="alice")
        self.assertEqual(ctx.exception.code, "PART_NOT_AGED")

    def test_age_gate_allows_at_exact_boundary(self) -> None:
        self._seed_ok_data_part(id_data_part="B09SB1100748292049", hours_old=1.0)
        result = self.service.start_box(template=self.template, raw_scan=SCAN_1, username="alice")
        self.assertEqual(result["qty_goal"], 5)

    def test_age_gate_allows_when_older(self) -> None:
        self._seed_ok_data_part(id_data_part="B09SB1100748292049", hours_old=10.0)
        result = self.service.start_box(template=self.template, raw_scan=SCAN_1, username="alice")
        self.assertEqual(result["qty_remaining"], 5)

    def test_full_happy_path_open_5_oks_then_close(self) -> None:
        self._seed_ok_data_part(id_data_part="B09SB1100748292049", hours_old=10.0)
        self.service.start_box(template=self.template, raw_scan=SCAN_1, username="alice")
        for _ in range(5):
            row = self.service.record_ok(datapart_pi=SCAN_1, part_ready_ratio=0.9, sticker_confidence=0.85, mp_check="alice")
            self.assertIsNotNone(row)
        rows = self.box_repo.load()
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(r["Data1"] is not None for r in rows))
        self.assertTrue(all(r["DatapartID"] is None for r in rows))
        result = self.service.close_box(datapart_pi=SCAN_1, raw_scan2=SCAN_2)
        self.assertEqual(result["closed_rows"], 5)
        closed = self.box_repo.load()
        self.assertTrue(all(r["DatapartID"] == "GM1SL1202609210714450009E" for r in closed))

    def test_record_ok_after_goal_reached_returns_none(self) -> None:
        self._seed_ok_data_part(id_data_part="B09SB1100748292049", hours_old=10.0)
        self.service.start_box(template=self.template, raw_scan=SCAN_1, username="alice")
        for _ in range(5):
            self.service.record_ok(datapart_pi=SCAN_1, part_ready_ratio=0.9, sticker_confidence=0.85, mp_check="alice")
        sixth = self.service.record_ok(datapart_pi=SCAN_1, part_ready_ratio=0.9, sticker_confidence=0.85, mp_check="alice")
        self.assertIsNone(sixth)

    def test_datapart2_mismatch_leaves_box_open(self) -> None:
        self._seed_ok_data_part(id_data_part="B09SB1100748292049", hours_old=10.0)
        self.service.start_box(template=self.template, raw_scan=SCAN_1, username="alice")
        with self.assertRaises(BoxTrackingError) as ctx:
            self.service.close_box(datapart_pi=SCAN_1, raw_scan2=SCAN_2_MISMATCH)
        self.assertEqual(ctx.exception.code, "MISMATCH")
        rows = self.box_repo.load()
        self.assertTrue(all(r["DatapartID"] is None for r in rows))

    def test_abandon_box_sets_sentinel(self) -> None:
        self._seed_ok_data_part(id_data_part="B09SB1100748292049", hours_old=10.0)
        self.service.start_box(template=self.template, raw_scan=SCAN_1, username="alice")
        result = self.service.abandon_box(datapart_pi=SCAN_1)
        self.assertEqual(result["abandoned_rows"], 5)
        rows = self.box_repo.load()
        self.assertTrue(all(r["DatapartID"] == "ABANDONED" for r in rows))

    def test_close_box_not_found_raises(self) -> None:
        # scan1/scan2 agree on part_number/company_id/qty_box (no mismatch),
        # but no box was ever opened for this datapart_pi.
        with self.assertRaises(BoxTrackingError) as ctx:
            self.service.close_box(datapart_pi=SCAN_1, raw_scan2=SCAN_2)
        self.assertEqual(ctx.exception.code, "NOT_FOUND")


def _make_session_state(*, box_tracking_enabled: bool) -> SessionState:
    return SessionState(
        session_id="test-sess",
        client_id="test-client",
        camera_index=0,
        template=InspectionTemplate(
            id=1,
            version_id=1,
            version_number=1,
            name="Test",
            description="",
            is_active=True,
            camera=CameraDefaults(),
            part_ready_roi=RoiGeometry(),
            sticker_roi=RoiGeometry(),
            vision=VisionConfig(),
            part_ready=PartReadyConfig(),
            sticker=StickerRule(part_name="P", expected_class="K"),
            persistence=PersistenceConfig(),
            box_tracking=BoxTrackingConfig(enabled=box_tracking_enabled),
        ),
        status=SessionStatus.IDLE,
    )


class MaybePersistBoxHookTest(unittest.TestCase):
    """_maybe_persist must call box_tracking_service.record_ok exactly once per
    genuine new ACCEPT, riding on the existing duplicate-event guard."""

    def setUp(self) -> None:
        self.box_tracking_service = MagicMock()
        self.box_tracking_service.record_ok.return_value = {"No": 1}
        self.service = InspectionSessionService(
            template_runtime=MagicMock(),
            results_repo=MagicMock(),
            sticker_inference=MagicMock(),
            box_tracking_service=self.box_tracking_service,
        )

    def _accept_validation(self) -> dict:
        return {
            "decision": DecisionCode.ACCEPT.value,
            "part_name": "P",
            "mp_check": "alice",
            "data1": 0.9,
            "data2": 0.85,
        }

    def test_record_ok_called_on_accept_with_open_box(self) -> None:
        state = _make_session_state(box_tracking_enabled=True)
        state.box_datapart_pi = "SCAN1"
        state.box_qty_remaining = 5
        self.service._maybe_persist(
            self._accept_validation(), state,
            part_ready={}, part_ready_roi_meta={}, sticker_roi_meta={}, sticker_detection={}, event_id="evt-1",
        )
        self.box_tracking_service.record_ok.assert_called_once_with(
            datapart_pi="SCAN1", part_ready_ratio=0.9, sticker_confidence=0.85, mp_check="alice",
        )
        self.assertEqual(state.box_qty_remaining, 4)

    def test_record_ok_not_called_when_box_tracking_disabled(self) -> None:
        state = _make_session_state(box_tracking_enabled=False)
        state.box_datapart_pi = "SCAN1"
        self.service._maybe_persist(
            self._accept_validation(), state,
            part_ready={}, part_ready_roi_meta={}, sticker_roi_meta={}, sticker_detection={}, event_id="evt-1",
        )
        self.box_tracking_service.record_ok.assert_not_called()

    def test_record_ok_not_called_when_no_open_box(self) -> None:
        state = _make_session_state(box_tracking_enabled=True)
        state.box_datapart_pi = None
        self.service._maybe_persist(
            self._accept_validation(), state,
            part_ready={}, part_ready_roi_meta={}, sticker_roi_meta={}, sticker_detection={}, event_id="evt-1",
        )
        self.box_tracking_service.record_ok.assert_not_called()

    def test_duplicate_event_does_not_double_call_record_ok(self) -> None:
        state = _make_session_state(box_tracking_enabled=True)
        state.box_datapart_pi = "SCAN1"
        state.box_qty_remaining = 5
        state.current_event_id = "evt-1"
        validation = self._accept_validation()
        self.service._maybe_persist(
            validation, state,
            part_ready={}, part_ready_roi_meta={}, sticker_roi_meta={}, sticker_detection={}, event_id="evt-1",
        )
        # Same event again (retried/duplicate frame) — existing persist_key guard
        # must short-circuit before the box hook runs a second time.
        self.service._maybe_persist(
            validation, state,
            part_ready={}, part_ready_roi_meta={}, sticker_roi_meta={}, sticker_detection={}, event_id="evt-1",
        )
        self.assertEqual(self.box_tracking_service.record_ok.call_count, 1)
        self.assertEqual(state.box_qty_remaining, 4)


if __name__ == "__main__":
    unittest.main()
