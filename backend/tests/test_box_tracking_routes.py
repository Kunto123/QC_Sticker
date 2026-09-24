"""Route-level tests for the box/datapart1, box/datapart2 and box/abandon
endpoints. Runs against the local JSON-backed BoxLuggageRepository /
OkDataPartRepository (QC_SUITE_DATABASE_BACKEND unset -> "local" in tests)."""
from __future__ import annotations

import sys
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.tests._test_env import ensure_test_data_root  # noqa: E402

ensure_test_data_root()

from backend.app.core.container import box_luggage_repo, ok_data_part_repo, users_repo
from backend.app.factory import create_app
from shared.contracts.enums import UserRole

SCAN_1 = "8125B-K2S -H3XX|1867490|5|B09SB1100748292049|QRTag"
SCAN_2 = "8125B-K2S -H3XX|1867490|5|GM1SL1202609210714450009E|QRTag"
SCAN_2_MISMATCH = "8125B-K2S -H3XX|0000000|5|GM1SL1202609210714450009E|QRTag"


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class BoxTrackingRoutesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.app.testing = True
        cls.client = cls.app.test_client()
        cls.admin_token = None
        cls.operator_token = None
        suffix = uuid4().hex[:8]
        cls.admin_username = f"box-admin-{suffix}"
        cls.operator_username = f"box-operator-{suffix}"
        users_repo.create_user(cls.admin_username, "admin123", UserRole.ADMIN.value)
        users_repo.create_user(cls.operator_username, "operator123", UserRole.OPERATOR.value)
        cls.admin_token = cls._login(cls.admin_username, "admin123")
        cls.operator_token = cls._login(cls.operator_username, "operator123")

    @classmethod
    def _login(cls, username: str, password: str) -> str:
        response = cls.client.post("/auth/login", json={"username": username, "password": password})
        assert response.status_code == 200, response.get_json()
        return str(response.get_json()["token"])

    def setUp(self) -> None:
        box_luggage_repo.save([])
        ok_data_part_repo.save([])

    def _seed_ok_data_part(self, *, id_data_part: str = "B09SB1100748292049", hours_old: float = 10.0) -> None:
        prod_date = datetime.now(UTC) - timedelta(hours=hours_old)
        ok_data_part_repo.seed_row(
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

    def _create_box_tracking_template(self, *, min_age_hours: float = 0.0) -> dict:
        response = self.client.post(
            "/templates",
            json={
                "name": f"box-template-{uuid4().hex[:8]}",
                "description": "box tracking route test",
                "is_active": True,
                "camera": {"camera_index": 0},
                "part_ready_roi": {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5},
                "sticker_roi": {"x": 0.1, "y": 0.1, "w": 0.6, "h": 0.6},
                "vision": {"model_path": "models/dummy.pt", "classes": ["sample-sticker"]},
                "part_ready": {"enabled": True, "method": "mean_std_threshold"},
                "sticker": {
                    "part_name": "Box Test Part",
                    "expected_class": "sample-sticker",
                    "enabled": True,
                    "min_roi_confidence": 0.0,
                },
                "persistence": {"write_to_db": True},
                "box_tracking": {"enabled": True, "min_age_hours": min_age_hours},
            },
            headers=_headers(self.admin_token),
        )
        self.assertEqual(response.status_code, 201, response.get_json())
        return response.get_json()

    def _start_session(self, template_version_id: int) -> str:
        response = self.client.post(
            "/inspection/sessions/start",
            json={
                "client_id": f"box-route-test-{uuid4().hex[:8]}",
                "camera_index": 0,
                "template_version_id": template_version_id,
            },
            headers=_headers(self.operator_token),
        )
        self.assertEqual(response.status_code, 201, response.get_json())
        return response.get_json()["session_id"]

    def test_datapart1_happy_path_opens_box(self) -> None:
        self._seed_ok_data_part()
        template = self._create_box_tracking_template()
        session_id = self._start_session(template["version_id"])
        response = self.client.post(
            f"/inspection/sessions/{session_id}/box/datapart1",
            json={"raw_scan": SCAN_1},
            headers=_headers(self.operator_token),
        )
        self.assertEqual(response.status_code, 201, response.get_json())
        payload = response.get_json()
        self.assertEqual(payload["qty_goal"], 5)
        self.assertEqual(payload["qty_remaining"], 5)

    def test_datapart1_age_gate_denies(self) -> None:
        self._seed_ok_data_part(hours_old=0.1)
        template = self._create_box_tracking_template(min_age_hours=5.0)
        session_id = self._start_session(template["version_id"])
        response = self.client.post(
            f"/inspection/sessions/{session_id}/box/datapart1",
            json={"raw_scan": SCAN_1},
            headers=_headers(self.operator_token),
        )
        self.assertEqual(response.status_code, 422, response.get_json())
        self.assertEqual(response.get_json()["code"], "PART_NOT_AGED")

    def test_datapart1_unknown_id_data_part_404(self) -> None:
        template = self._create_box_tracking_template()
        session_id = self._start_session(template["version_id"])
        response = self.client.post(
            f"/inspection/sessions/{session_id}/box/datapart1",
            json={"raw_scan": SCAN_1},
            headers=_headers(self.operator_token),
        )
        self.assertEqual(response.status_code, 404, response.get_json())
        self.assertEqual(response.get_json()["code"], "NOT_FOUND")

    def test_datapart2_happy_path_closes_box(self) -> None:
        self._seed_ok_data_part()
        template = self._create_box_tracking_template()
        session_id = self._start_session(template["version_id"])
        self.client.post(
            f"/inspection/sessions/{session_id}/box/datapart1",
            json={"raw_scan": SCAN_1},
            headers=_headers(self.operator_token),
        )
        response = self.client.post(
            f"/inspection/sessions/{session_id}/box/datapart2",
            json={"raw_scan": SCAN_2},
            headers=_headers(self.operator_token),
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()["closed_rows"], 5)

    def test_datapart2_mismatch_warns_and_leaves_box_open(self) -> None:
        self._seed_ok_data_part()
        template = self._create_box_tracking_template()
        session_id = self._start_session(template["version_id"])
        self.client.post(
            f"/inspection/sessions/{session_id}/box/datapart1",
            json={"raw_scan": SCAN_1},
            headers=_headers(self.operator_token),
        )
        response = self.client.post(
            f"/inspection/sessions/{session_id}/box/datapart2",
            json={"raw_scan": SCAN_2_MISMATCH},
            headers=_headers(self.operator_token),
        )
        self.assertEqual(response.status_code, 422, response.get_json())
        self.assertEqual(response.get_json()["code"], "MISMATCH")
        # Box still open — a retry with the correct scan succeeds afterwards.
        retry = self.client.post(
            f"/inspection/sessions/{session_id}/box/datapart2",
            json={"raw_scan": SCAN_2},
            headers=_headers(self.operator_token),
        )
        self.assertEqual(retry.status_code, 200, retry.get_json())

    def test_abandon_requires_admin_role(self) -> None:
        self._seed_ok_data_part()
        template = self._create_box_tracking_template()
        session_id = self._start_session(template["version_id"])
        self.client.post(
            f"/inspection/sessions/{session_id}/box/datapart1",
            json={"raw_scan": SCAN_1},
            headers=_headers(self.operator_token),
        )
        operator_attempt = self.client.post(
            f"/inspection/sessions/{session_id}/box/abandon",
            headers=_headers(self.operator_token),
        )
        self.assertEqual(operator_attempt.status_code, 403, operator_attempt.get_json())
        admin_attempt = self.client.post(
            f"/inspection/sessions/{session_id}/box/abandon",
            headers=_headers(self.admin_token),
        )
        self.assertEqual(admin_attempt.status_code, 200, admin_attempt.get_json())
        self.assertEqual(admin_attempt.get_json()["abandoned_rows"], 5)


if __name__ == "__main__":
    unittest.main()
