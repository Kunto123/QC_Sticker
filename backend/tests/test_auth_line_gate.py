"""Operator login is restricted to the machine's configured line.

An operator's MC_ID (set per-account, Admin -> Operators) must match
machine_settings.json -> identity.line (Admin -> Machine Settings ->
Identity) for /auth/login to succeed. LEADERPI/admin is never gated by this
— otherwise a wrong/blank Line value could lock every admin out of the one
screen that can fix it. The gate is also inactive while Line itself is
unset, so a fresh/unconfigured machine doesn't lock out every operator
before an admin first sets it.
"""
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

from backend.app.core.container import app_config, users_repo  # noqa: E402
from backend.app.factory import create_app  # noqa: E402
from shared.contracts.enums import UserRole  # noqa: E402


class OperatorLineGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.app.testing = True
        cls.client = cls.app.test_client()

    def setUp(self) -> None:
        self._original_machine_line_id = app_config.machine_line_id

    def tearDown(self) -> None:
        app_config.machine_line_id = self._original_machine_line_id

    def _create_operator(self, mc_id: str) -> tuple[str, str]:
        username = f"linegate-{uuid4().hex[:8]}"
        password = "linegate-pass-123"
        users_repo.create_user(username, password, UserRole.OPERATOR.value, mc_id=mc_id)
        return username, password

    def _login(self, username: str, password: str):
        return self.client.post("/auth/login", json={"username": username, "password": password})

    def test_operator_login_allowed_when_mc_id_matches_configured_line(self) -> None:
        app_config.machine_line_id = "GB3"
        username, password = self._create_operator("GB3")

        response = self._login(username, password)

        self.assertEqual(response.status_code, 200, response.get_json())

    def test_operator_login_rejected_when_mc_id_does_not_match_configured_line(self) -> None:
        app_config.machine_line_id = "GB3"
        username, password = self._create_operator("A01")

        response = self._login(username, password)

        self.assertEqual(response.status_code, 403, response.get_json())

    def test_operator_login_rejected_when_mc_id_blank_and_line_configured(self) -> None:
        app_config.machine_line_id = "GB3"
        username, password = self._create_operator("")

        response = self._login(username, password)

        self.assertEqual(response.status_code, 403, response.get_json())

    def test_operator_login_allowed_when_line_not_configured(self) -> None:
        """Fail-open: no restriction while identity.line itself is unset."""
        app_config.machine_line_id = ""
        username, password = self._create_operator("")

        response = self._login(username, password)

        self.assertEqual(response.status_code, 200, response.get_json())

    def test_mc_id_match_is_case_insensitive(self) -> None:
        app_config.machine_line_id = "GB3"
        username, password = self._create_operator("gb3")

        response = self._login(username, password)

        self.assertEqual(response.status_code, 200, response.get_json())

    def test_admin_login_never_gated_by_line(self) -> None:
        app_config.machine_line_id = "GB3"
        username = f"linegate-admin-{uuid4().hex[:8]}"
        password = "linegate-admin-pass-123"
        users_repo.create_user(username, password, UserRole.ADMIN.value, mc_id="")

        response = self._login(username, password)

        self.assertEqual(response.status_code, 200, response.get_json())

    def test_machine_settings_put_live_updates_login_gate_without_restart(self) -> None:
        """Real symptom: admin edits Identity -> Line (e.g. from a stale/
        wrong value left over from a previous boot to "GB3") and saves, then
        an operator whose MC_ID matches the *new* Line still gets rejected —
        because PUT /machine-settings only live-applied identity.line to
        InspectionSessionService (for session line_id), never to
        app_config.machine_line_id, which is what the login gate actually
        reads. Only a full backend restart picked up the change. Seed a
        deliberately different stale value first so a broken live-apply
        shows up as a mismatch instead of accidentally passing because the
        gate is inactive (empty configured_line fails open)."""
        app_config.machine_line_id = "STALE-FROM-BOOT"
        admin_username = f"linegate-admin2-{uuid4().hex[:8]}"
        admin_password = "linegate-admin2-pass-123"
        users_repo.create_user(admin_username, admin_password, UserRole.ADMIN.value, mc_id="")
        admin_login = self._login(admin_username, admin_password)
        self.assertEqual(admin_login.status_code, 200, admin_login.get_json())
        admin_headers = {"Authorization": f"Bearer {admin_login.get_json()['token']}"}

        original_settings = self.client.get("/machine-settings", headers=admin_headers).get_json()
        try:
            updated_settings = dict(original_settings)
            updated_settings["identity"] = {"line": "GB3"}
            put_response = self.client.put(
                "/machine-settings", json=updated_settings, headers=admin_headers
            )
            self.assertEqual(put_response.status_code, 200, put_response.get_json())

            username, password = self._create_operator("GB3")
            response = self._login(username, password)

            self.assertEqual(response.status_code, 200, response.get_json())
        finally:
            self.client.put("/machine-settings", json=original_settings, headers=admin_headers)
            app_config.machine_line_id = self._original_machine_line_id


if __name__ == "__main__":
    unittest.main()
