from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_ROOT = Path(tempfile.mkdtemp(prefix="qc-suite-persistence-tests-"))
atexit.register(lambda: shutil.rmtree(TEST_DATA_ROOT, ignore_errors=True))
os.environ["QC_SUITE_DATA_ROOT"] = str(TEST_DATA_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.repositories.hybrid_inspection_results_repository import HybridInspectionResultsRepository
from backend.app.repositories.inspection_results_repository import InspectionResultsRepository
from backend.app.repositories.postgres.inspection_mirror_repository import PostgresInspectionMirrorRepository
from backend.app.repositories.sqlserver.inspection_mirror_repository import SqlServerInspectionMirrorRepository


class _MirrorSuccess:
    def __init__(self) -> None:
        self.seen_payloads: list[dict] = []
        self.deleted_ids: list[int] = []

    def create_result(self, payload: dict) -> dict:
        self.seen_payloads.append(dict(payload))
        return {"id": 987}

    def delete_result(self, mirror_id: int) -> bool:
        self.deleted_ids.append(int(mirror_id))
        return True


class _MirrorFailure:
    def create_result(self, payload: dict) -> dict:
        raise RuntimeError("mirror failed")


class _MirrorFailThenSuccess:
    def __init__(self) -> None:
        self._attempt = 0

    def create_result(self, payload: dict) -> dict:
        self._attempt += 1
        if self._attempt == 1:
            raise RuntimeError("mirror failed once")
        return {"id": 654}


def _sample_payload() -> dict:
    return {
        "template_version_id": 1,
        "line_id": "LINE-A",
        "station_id": "ST-01",
        "part_name": "Part-A",
        "mp_check": "operator",
        "data1": 0.12,
        "data2": 0.34,
        "decision": "ACCEPT",
        "decision_code": "ACCEPT",
        "reject_reason_code": None,
        "retry_count": 0,
        "operator_user_id": 2,
        "part_ready_status": "ready",
        "part_ready_match_ratio": 0.93,
        "part_ready_distance": 1.1,
        "detected_class": "K0W-HB0",
        "expected_class": "K0W-HB0",
        "sticker_confidence": 0.81,
        "sticker_backend": "classic",
        "sticker_bbox": {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0},
        "validation_details": {"status": "accepted"},
        "part_ready_roi_meta": {"x": 1, "y": 2, "width": 3, "height": 4},
        "sticker_roi_meta": {"x": 5, "y": 6, "width": 7, "height": 8},
        "targets": [{"target_id": "target-1"}],
        "inspected_at": "2026-04-07T00:00:00+00:00",
    }


class HybridInspectionPersistenceTest(unittest.TestCase):
    def test_hybrid_repo_keeps_full_detail_locally_on_successful_sql_mirror(self) -> None:
        local_repo = InspectionResultsRepository()
        mirror = _MirrorSuccess()
        repo = HybridInspectionResultsRepository(local_repo, mirror)

        created = repo.create_result(_sample_payload())
        stored = repo.get_result(int(created["id"]))

        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored["push_status"], "sent")
        self.assertEqual(stored["sql_mirror_id"], 987)
        self.assertEqual(stored["validation_details"], {"status": "accepted"})
        self.assertEqual(stored["sticker_bbox"]["x1"], 1.0)
        self.assertEqual(stored["part_ready_roi_meta"]["width"], 3)
        self.assertEqual(len(mirror.seen_payloads), 1)

    def test_hybrid_repo_marks_failed_push_but_keeps_local_record(self) -> None:
        local_repo = InspectionResultsRepository()
        repo = HybridInspectionResultsRepository(local_repo, _MirrorFailure())

        created = repo.create_result(_sample_payload())
        stored = repo.get_result(int(created["id"]))

        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored["push_status"], "failed")
        self.assertEqual(stored["retry_count"], 1)
        self.assertIn("mirror failed", stored["last_push_error"])
        self.assertEqual(stored["part_name"], "Part-A")

    def test_retry_result_resends_failed_push_and_clears_error(self) -> None:
        local_repo = InspectionResultsRepository()
        repo = HybridInspectionResultsRepository(local_repo, _MirrorFailThenSuccess())

        created = repo.create_result(_sample_payload())
        self.assertEqual(created["push_status"], "failed")

        retried = repo.retry_result(int(created["id"]))
        stored = repo.get_result(int(created["id"]))

        self.assertEqual(retried["push_status"], "sent")
        self.assertEqual(retried["sql_mirror_id"], 654)
        self.assertIsNone(retried["last_push_error"])
        self.assertEqual(retried["retry_count"], 1)
        self.assertIsNotNone(retried.get("last_pushed_at"))
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored["push_status"], "sent")
        self.assertEqual(stored["sql_mirror_id"], 654)

    def test_retry_failed_retries_only_retryable_results(self) -> None:
        local_repo = InspectionResultsRepository()
        success_mirror = _MirrorSuccess()
        repo = HybridInspectionResultsRepository(local_repo, success_mirror)

        sent = repo.create_result(_sample_payload())
        failed_payload = dict(_sample_payload())
        failed_payload["part_name"] = "Part-B"
        failed_payload["inspected_at"] = "2026-04-07T00:10:00+00:00"
        repo._sql_mirror_repo = _MirrorFailure()
        failed = repo.create_result(failed_payload)

        repo._sql_mirror_repo = _MirrorSuccess()
        retried = repo.retry_failed(result_ids=[int(sent["id"]), int(failed["id"])], limit=10)

        self.assertEqual(len(retried), 1)
        self.assertEqual(retried[0]["id"], failed["id"])
        self.assertEqual(retried[0]["push_status"], "sent")

    def test_delete_result_deletes_local_and_mirror_when_available(self) -> None:
        local_repo = InspectionResultsRepository()
        mirror = _MirrorSuccess()
        repo = HybridInspectionResultsRepository(local_repo, mirror)

        created = repo.create_result(_sample_payload())
        result_id = int(created["id"])

        deleted = repo.delete_result(result_id)

        self.assertEqual(int(deleted["id"]), result_id)
        self.assertIsNone(repo.get_result(result_id))
        self.assertEqual(mirror.deleted_ids, [987])

    def test_sql_payload_uses_only_required_contract_fields(self) -> None:
        payload = SqlServerInspectionMirrorRepository.build_sql_payload(_sample_payload())

        self.assertEqual(
            payload,
            {
                "PartName": "K0W-HB0",
                "DateCheckMC": "2026-04-07T00:00:00+00:00",
                "MPCheck": "operator",
                "Data1": 0.93,   # part_ready_match_ratio
                "Data2": 0.81,   # sticker_confidence
                "Line": "LINE-A",
            },
        )

    def test_sql_payload_exactly_six_keys_no_extras(self) -> None:
        """The SQL push must never contain fields beyond the agreed contract."""
        payload = SqlServerInspectionMirrorRepository.build_sql_payload(_sample_payload())
        self.assertEqual(
            set(payload.keys()),
            {"PartName", "DateCheckMC", "MPCheck", "Data1", "Data2", "Line"},
        )

    def test_sql_payload_data1_is_part_ready_data2_is_sticker(self) -> None:
        """Data1 = confidence part ready, Data2 = confidence sticker — never inverted."""
        local = dict(_sample_payload())
        local["part_ready_match_ratio"] = 0.77
        local["sticker_confidence"] = 0.55

        payload = SqlServerInspectionMirrorRepository.build_sql_payload(local)

        self.assertEqual(payload["Data1"], 0.77, "Data1 must be part_ready_match_ratio")
        self.assertEqual(payload["Data2"], 0.55, "Data2 must be sticker_confidence")

    def test_sql_payload_partname_prefers_expected_class_with_part_name_fallback(self) -> None:
        local = dict(_sample_payload())

        sqlserver_payload = SqlServerInspectionMirrorRepository.build_sql_payload(local)
        postgres_payload = PostgresInspectionMirrorRepository.build_sql_payload(local)

        self.assertEqual(sqlserver_payload["PartName"], "K0W-HB0")
        self.assertEqual(postgres_payload["PartName"], "K0W-HB0")

        local["expected_class"] = ""
        fallback_payload = SqlServerInspectionMirrorRepository.build_sql_payload(local)

        self.assertEqual(fallback_payload["PartName"], "Part-A")

    def test_local_record_data1_data2_align_with_sql_contract(self) -> None:
        """Local data1/data2 must mirror the SQL contract so they are consistent.

        data1 = part_ready confidence (same value as part_ready_match_ratio)
        data2 = sticker confidence    (same value as sticker_confidence)
        """
        local_repo = InspectionResultsRepository()
        mirror = _MirrorSuccess()
        repo = HybridInspectionResultsRepository(local_repo, mirror)

        sample = dict(_sample_payload())
        # Explicitly set canonical values so the test is unambiguous
        sample["part_ready_match_ratio"] = 0.88
        sample["sticker_confidence"] = 0.66
        # Set data1/data2 consistent with contract (as produced by inspection_session)
        sample["data1"] = 0.88  # part_ready confidence
        sample["data2"] = 0.66  # sticker confidence

        created = repo.create_result(sample)
        stored = repo.get_result(int(created["id"]))

        self.assertIsNotNone(stored)
        assert stored is not None
        # data1 must be part_ready confidence
        self.assertAlmostEqual(float(stored["data1"]), 0.88, places=5,
                               msg="data1 must store part_ready confidence per contract")
        # data2 must be sticker confidence
        self.assertAlmostEqual(float(stored["data2"]), 0.66, places=5,
                               msg="data2 must store sticker confidence per contract")
