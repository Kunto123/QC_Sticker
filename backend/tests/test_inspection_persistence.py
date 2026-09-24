from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.tests._test_env import ensure_test_data_root  # noqa: E402

TEST_DATA_ROOT = ensure_test_data_root()

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
        "template_name": "Sticker Template A",
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

    def test_concurrent_create_and_update_does_not_lose_records(self) -> None:
        # Regression for a real production symptom: the UI's in-session accept
        # counter (pure in-memory increment) reached N, but only a handful of
        # results ended up in inspection_results.json / the SQL mirror. Root
        # cause was InspectionResultsRepository.create_result/update_result
        # doing load() -> mutate -> save() without holding the lock across the
        # whole sequence, so a concurrent writer (e.g. the push worker's
        # background thread updating push_status on other rows) could load a
        # stale snapshot between this thread's load() and save() and silently
        # clobber a just-appended record when it saved. Simulates that by
        # hammering create_result + update_result from many threads at once,
        # the same pattern HybridInspectionResultsRepository.create_result
        # does for every accepted commit (local create, then local update to
        # set push_status).
        local_repo = InspectionResultsRepository()
        created_ids: list[int] = []
        ids_lock = threading.Lock()
        errors: list[BaseException] = []

        def worker(index: int) -> None:
            try:
                payload = dict(_sample_payload())
                payload["part_name"] = f"Concurrent-{index}"
                record = local_repo.create_result(payload)
                local_repo.update_result(int(record["id"]), {"push_status": "sent"})
                with ids_lock:
                    created_ids.append(int(record["id"]))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(25)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(created_ids), 25)
        self.assertEqual(len(set(created_ids)), 25, "duplicate ids were assigned under concurrent writes")
        for result_id in created_ids:
            stored = local_repo.get_result(result_id)
            self.assertIsNotNone(stored, f"result #{result_id} was lost under concurrent writes")
            assert stored is not None
            self.assertEqual(stored["push_status"], "sent")

    def test_sql_payload_uses_only_required_contract_fields(self) -> None:
        payload = SqlServerInspectionMirrorRepository.build_sql_payload(_sample_payload())

        self.assertEqual(
            payload,
            {
                "PartName": "Sticker Template A",
                "DateCheckMC": "2026-04-07T00:00:00+00:00",
                # "operator" has no 4-digit run -> falls back to the raw value.
                "MPCheck": "operator",
                # part_ready_match_ratio 0.93 pushed as a whole-number percentage string.
                "Data1": "93%",
                # sticker_confidence 0.81 pushed as a whole-number percentage string.
                "Data2": "81%",
                # "LINE-A" -> only its last character is pushed.
                "Line": "A",
            },
        )

    def test_sql_payload_mp_check_extracts_4digit_member_id_code(self) -> None:
        """MPCheck = the 4-digit code in the middle of Member_ID
        (e.g. "ID 9101 PUTRA" -> "9101"), not the full string."""
        local = dict(_sample_payload())
        local["mp_check"] = "ID 9101 PUTRA"

        sqlserver_payload = SqlServerInspectionMirrorRepository.build_sql_payload(local)
        postgres_payload = PostgresInspectionMirrorRepository.build_sql_payload(local)

        self.assertEqual(sqlserver_payload["MPCheck"], "9101")
        self.assertEqual(postgres_payload["MPCheck"], "9101")

    def test_sql_payload_mp_check_falls_back_to_raw_value_without_4digit_code(self) -> None:
        local = dict(_sample_payload())
        local["mp_check"] = "PUTRA ONLY"

        payload = SqlServerInspectionMirrorRepository.build_sql_payload(local)

        self.assertEqual(payload["MPCheck"], "PUTRA ONLY")

    def test_sql_payload_line_uses_only_last_character(self) -> None:
        """Line = the last character of identity.line (e.g. "GB3" -> "3",
        "A01" -> "1"), not the full line code."""
        local = dict(_sample_payload())

        for line_value, expected in (("GB3", "3"), ("A01", "1"), ("A", "A")):
            local["line_id"] = line_value
            sqlserver_payload = SqlServerInspectionMirrorRepository.build_sql_payload(local)
            postgres_payload = PostgresInspectionMirrorRepository.build_sql_payload(local)
            self.assertEqual(sqlserver_payload["Line"], expected)
            self.assertEqual(postgres_payload["Line"], expected)

    def test_sql_payload_line_is_none_when_line_id_unset(self) -> None:
        local = dict(_sample_payload())
        local["line_id"] = None

        payload = SqlServerInspectionMirrorRepository.build_sql_payload(local)

        self.assertIsNone(payload["Line"])

    def test_sql_payload_exactly_six_keys_no_extras(self) -> None:
        """The SQL push must never contain fields beyond the agreed contract."""
        payload = SqlServerInspectionMirrorRepository.build_sql_payload(_sample_payload())
        self.assertEqual(
            set(payload.keys()),
            {"PartName", "DateCheckMC", "MPCheck", "Data1", "Data2", "Line"},
        )

    def test_sql_payload_data1_is_part_ready_data2_is_sticker(self) -> None:
        """Data1 = confidence part ready, Data2 = confidence sticker — never
        inverted. Both pushed as a whole-number percentage string ("77%"),
        not the raw 0..1 ratio."""
        local = dict(_sample_payload())
        local["part_ready_match_ratio"] = 0.77
        local["sticker_confidence"] = 0.55

        payload = SqlServerInspectionMirrorRepository.build_sql_payload(local)

        self.assertEqual(payload["Data1"], "77%", "Data1 must be part_ready_match_ratio as a percentage string")
        self.assertEqual(payload["Data2"], "55%", "Data2 must be sticker_confidence as a percentage string")

    def test_sql_payload_data1_data2_are_none_when_ratio_unset(self) -> None:
        local = dict(_sample_payload())
        local["part_ready_match_ratio"] = None
        local["sticker_confidence"] = None

        payload = SqlServerInspectionMirrorRepository.build_sql_payload(local)

        self.assertIsNone(payload["Data1"])
        self.assertIsNone(payload["Data2"])

    def test_sql_payload_data1_data2_percentage_rounds_to_whole_number(self) -> None:
        local = dict(_sample_payload())
        local["part_ready_match_ratio"] = 0.93261
        local["sticker_confidence"] = 0.8055555

        sqlserver_payload = SqlServerInspectionMirrorRepository.build_sql_payload(local)
        postgres_payload = PostgresInspectionMirrorRepository.build_sql_payload(local)

        self.assertEqual(sqlserver_payload["Data1"], "93%")
        self.assertEqual(sqlserver_payload["Data2"], "81%")
        self.assertEqual(postgres_payload["Data1"], "93%")
        self.assertEqual(postgres_payload["Data2"], "81%")

    def test_sql_payload_partname_prefers_template_name_with_expected_class_and_part_name_fallback(self) -> None:
        """PartName = the template's own name ("Preset Name" in Admin ->
        Templates), NOT the ML expected_class label — expected_class/part_name
        are only a safety-net fallback if template_name is somehow missing."""
        local = dict(_sample_payload())

        sqlserver_payload = SqlServerInspectionMirrorRepository.build_sql_payload(local)
        postgres_payload = PostgresInspectionMirrorRepository.build_sql_payload(local)

        self.assertEqual(sqlserver_payload["PartName"], "Sticker Template A")
        self.assertEqual(postgres_payload["PartName"], "Sticker Template A")

        local["template_name"] = ""
        expected_class_fallback_payload = SqlServerInspectionMirrorRepository.build_sql_payload(local)
        self.assertEqual(expected_class_fallback_payload["PartName"], "K0W-HB0")

        local["expected_class"] = ""
        part_name_fallback_payload = SqlServerInspectionMirrorRepository.build_sql_payload(local)
        self.assertEqual(part_name_fallback_payload["PartName"], "Part-A")

    def test_insert_plan_duplicates_value_across_multiple_configured_columns(self) -> None:
        """A field configured with QC_SUITE_INSPECTION_COL_*=ColA,ColB must
        write the same value into both physical columns on insert — this is
        how a target table with redundant/legacy columns (e.g. both
        `DateCheckMC` and `DateSendDB`) gets kept in sync."""
        record = SqlServerInspectionMirrorRepository.build_sql_payload(_sample_payload())

        for repo_cls in (PostgresInspectionMirrorRepository, SqlServerInspectionMirrorRepository):
            repo = repo_cls.__new__(repo_cls)
            repo._field_columns = [
                ("PartName", ["PartName"]),
                ("DateCheckMC", ["DateCheckMC", "DateSendDB"]),
                ("MPCheck", ["MPCheck"]),
                ("Data1", ["Data1"]),
                ("Data2", ["Data2"]),
                ("Line", ["Line"]),
            ]
            columns, values = repo._build_insert_plan(record)

            self.assertEqual(
                columns,
                ["PartName", "DateCheckMC", "DateSendDB", "MPCheck", "Data1", "Data2", "Line"],
            )
            date_check_mc_index = columns.index("DateCheckMC")
            date_send_db_index = columns.index("DateSendDB")
            self.assertEqual(values[date_check_mc_index], values[date_send_db_index])
            self.assertEqual(values[date_check_mc_index], record["DateCheckMC"])

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
