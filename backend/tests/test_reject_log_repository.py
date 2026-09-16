from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.repositories.reject_log_repository import RejectLogRepository


class RejectLogRepositoryTest(unittest.TestCase):
    def test_append_only_log_is_written_and_read_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = RejectLogRepository(Path(tmpdir))
            entry = repo.log_reject(
                {
                    "session_id": "sess-1",
                    "event_id": "evt-1",
                    "template_version_id": 3,
                    "line_id": "LINE-A",
                    "station_id": "ST-01",
                    "part_name": "Part-A",
                    "decision_code": "REJECT",
                    "reject_reason_code": "OUT_OF_POSITION",
                    "operator_user_id": 7,
                    "validation_details": {"status": "rejected"},
                    "part_ready": {"part_ready": True},
                    "sticker_detection": {"backend": "classic"},
                    "part_ready_roi_meta": {"x": 1, "y": 2, "width": 3, "height": 4},
                    "sticker_roi_meta": {"x": 5, "y": 6, "width": 7, "height": 8},
                }
            )

            log_path = Path(tmpdir) / "reject_log.jsonl"
            self.assertTrue(log_path.exists())
            self.assertEqual(entry["reject_reason_code"], "OUT_OF_POSITION")

            recent = repo.list_recent(limit=10)
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0]["session_id"], "sess-1")
            self.assertEqual(recent[0]["part_name"], "Part-A")
