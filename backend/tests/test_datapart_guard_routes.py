"""GET /datapart-guard/status and POST /datapart-guard/override."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.tests._test_env import ensure_test_data_root  # noqa: E402

ensure_test_data_root()

from backend.app.core.container import app_config, datapart_guard_service, users_repo  # noqa: E402
from backend.app.core.security import hash_rfid_uid, normalize_rfid_uid, rfid_uid_last4  # noqa: E402
from backend.app.factory import create_app  # noqa: E402
from shared.contracts.enums import UserRole  # noqa: E402


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class DatapartGuardRoutesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.app.testing = True
        cls.client = cls.app.test_client()

        cls.admin_username = f"dgroutes-admin-{uuid4().hex[:8]}"
        cls.operator_username = f"dgroutes-op-{uuid4().hex[:8]}"
        users_repo.create_user(cls.admin_username, "dgroutes-pass-123", UserRole.ADMIN.value)
        users_repo.create_user(cls.operator_username, "dgroutes-pass-123", UserRole.OPERATOR.value)
        cls.admin_token = cls._login(cls.admin_username, "dgroutes-pass-123")
        cls.operator_token = cls._login(cls.operator_username, "dgroutes-pass-123")

        cls.admin_id = users_repo.get_by_username(cls.admin_username)["id"]
        cls.operator_id = users_repo.get_by_username(cls.operator_username)["id"]
        cls.admin_rfid_uid = f"ADMINCARD{uuid4().hex[:6].upper()}"
        cls.operator_rfid_uid = f"OPCARD{uuid4().hex[:6].upper()}"
        cls._bind_rfid(cls.admin_id, cls.admin_rfid_uid)
        cls._bind_rfid(cls.operator_id, cls.operator_rfid_uid)

    @classmethod
    def _bind_rfid(cls, user_id: int, raw_uid: str) -> None:
        normalized = normalize_rfid_uid(raw_uid)
        users_repo.set_rfid_uid(
            user_id,
            normalized_uid=normalized,
            rfid_uid_hash=hash_rfid_uid(normalized, app_config.secret_key),
            rfid_uid_last4=rfid_uid_last4(normalized),
        )

    @classmethod
    def _login(cls, username: str, password: str) -> str:
        response = cls.client.post("/auth/login", json={"username": username, "password": password})
        assert response.status_code == 200, response.get_json()
        return str(response.get_json()["token"])

    def setUp(self) -> None:
        # The container's mirror repo is None under the local test backend,
        # so record_pushed() is always a no-op here — poke the service's
        # state directly to set up "locked"/"unlocked" scenarios for the
        # route tests. How locking actually happens is covered separately
        # in test_datapart_guard.py.
        datapart_guard_service._pending_ids = []
        datapart_guard_service._locked = False

    def test_status_requires_auth(self) -> None:
        response = self.client.get("/datapart-guard/status")
        self.assertEqual(response.status_code, 401, response.get_json())

    def test_status_reachable_by_any_authenticated_role(self) -> None:
        admin_response = self.client.get("/datapart-guard/status", headers=_headers(self.admin_token))
        operator_response = self.client.get("/datapart-guard/status", headers=_headers(self.operator_token))

        self.assertEqual(admin_response.status_code, 200, admin_response.get_json())
        self.assertEqual(operator_response.status_code, 200, operator_response.get_json())
        self.assertFalse(admin_response.get_json()["locked"])

    def test_override_requires_admin_role(self) -> None:
        response = self.client.post("/datapart-guard/override", headers=_headers(self.operator_token))
        self.assertEqual(response.status_code, 403, response.get_json())

    def test_override_requires_auth(self) -> None:
        response = self.client.post("/datapart-guard/override")
        self.assertEqual(response.status_code, 401, response.get_json())

    def test_admin_override_clears_lock_and_is_reflected_in_status(self) -> None:
        datapart_guard_service._pending_ids = [1, 2, 3, 4, 5]
        datapart_guard_service._locked = True

        override_response = self.client.post(
            "/datapart-guard/override", headers=_headers(self.admin_token)
        )

        self.assertEqual(override_response.status_code, 200, override_response.get_json())
        self.assertFalse(override_response.get_json()["locked"])
        self.assertEqual(override_response.get_json()["last_override_by"], self.admin_username)

        status_response = self.client.get("/datapart-guard/status", headers=_headers(self.operator_token))
        self.assertFalse(status_response.get_json()["locked"])

    def test_override_with_rfid_requires_auth(self) -> None:
        response = self.client.post(
            "/datapart-guard/override-with-rfid", json={"rfid_uid": self.admin_rfid_uid}
        )
        self.assertEqual(response.status_code, 401, response.get_json())

    def test_override_with_leaderpi_rfid_clears_lock_from_operator_session(self) -> None:
        """The whole point: an OPERATOR's own session can unlock by scanning
        a LEADERPI's card — no need to switch to an admin session."""
        datapart_guard_service._pending_ids = [1, 2, 3, 4, 5]
        datapart_guard_service._locked = True

        response = self.client.post(
            "/datapart-guard/override-with-rfid",
            json={"rfid_uid": self.admin_rfid_uid},
            headers=_headers(self.operator_token),
        )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertFalse(response.get_json()["locked"])
        self.assertEqual(response.get_json()["last_override_by"], self.admin_username)

    def test_override_with_non_leaderpi_rfid_is_rejected(self) -> None:
        datapart_guard_service._pending_ids = [1, 2, 3, 4, 5]
        datapart_guard_service._locked = True

        response = self.client.post(
            "/datapart-guard/override-with-rfid",
            json={"rfid_uid": self.operator_rfid_uid},
            headers=_headers(self.operator_token),
        )

        self.assertEqual(response.status_code, 403, response.get_json())
        self.assertTrue(datapart_guard_service.is_locked(), "lock must stay engaged on a rejected bypass attempt")

    def test_override_with_unknown_rfid_is_rejected(self) -> None:
        datapart_guard_service._pending_ids = [1, 2, 3, 4, 5]
        datapart_guard_service._locked = True

        response = self.client.post(
            "/datapart-guard/override-with-rfid",
            json={"rfid_uid": "NOSUCHCARD999"},
            headers=_headers(self.operator_token),
        )

        self.assertEqual(response.status_code, 401, response.get_json())
        self.assertTrue(datapart_guard_service.is_locked())

    def test_override_with_malformed_rfid_is_rejected(self) -> None:
        response = self.client.post(
            "/datapart-guard/override-with-rfid",
            json={"rfid_uid": "!!"},
            headers=_headers(self.operator_token),
        )

        self.assertEqual(response.status_code, 400, response.get_json())


if __name__ == "__main__":
    unittest.main()
