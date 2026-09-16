from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from backend.app.core.security import TokenStore
from backend.app.repositories.auth_audit_repository import AuthAuditRepository
from shared.contracts.auth import UserInfo
from shared.contracts.enums import UserRole


class TokenStoreTest(unittest.TestCase):
    def test_expired_token_is_rejected_and_purged(self) -> None:
        store = TokenStore(ttl_seconds=120)
        user = UserInfo(id=1, username="tester", role=UserRole.ADMIN)

        record = store.issue(user, client_name="unit-test")
        record.expires_at = record.issued_at - timedelta(seconds=1)

        self.assertIsNone(store.get(record.token))
        self.assertEqual(store.list_user_sessions(user.id), [])

    def test_revoke_user_clears_only_target_user_sessions(self) -> None:
        store = TokenStore(ttl_seconds=120)
        target_user = UserInfo(id=7, username="target", role=UserRole.OPERATOR)
        other_user = UserInfo(id=8, username="other", role=UserRole.ADMIN)

        target_session_1 = store.issue(target_user)
        target_session_2 = store.issue(target_user)
        other_session = store.issue(other_user)

        self.assertEqual(store.revoke_user(target_user.id), 2)
        self.assertIsNone(store.get(target_session_1.token))
        self.assertIsNone(store.get(target_session_2.token))
        self.assertIsNotNone(store.get(other_session.token))


class AuthAuditRepositoryTest(unittest.TestCase):
    def _make_repo(self) -> tuple[AuthAuditRepository, Path]:
        tmp = Path(tempfile.mkdtemp())
        return AuthAuditRepository(store_dir=tmp), tmp

    def test_log_and_list_recent_returns_newest_first(self) -> None:
        repo, _ = self._make_repo()
        repo.log("login_success", user_id=1, username="alice")
        repo.log("logout", user_id=1, username="alice")
        repo.log("login_failure", username="bad_actor", details="Invalid credentials")

        entries = repo.list_recent(limit=10)
        self.assertEqual(len(entries), 3)
        # newest first
        self.assertEqual(entries[0]["event_type"], "login_failure")
        self.assertEqual(entries[1]["event_type"], "logout")
        self.assertEqual(entries[2]["event_type"], "login_success")

    def test_list_recent_filters_by_user_id(self) -> None:
        repo, _ = self._make_repo()
        repo.log("login_success", user_id=1, username="alice")
        repo.log("login_success", user_id=2, username="bob")
        repo.log("logout", user_id=1, username="alice")

        alice_events = repo.list_recent(limit=10, user_id=1)
        self.assertEqual(len(alice_events), 2)
        self.assertTrue(all(e["user_id"] == 1 for e in alice_events))

    def test_list_recent_respects_limit(self) -> None:
        repo, _ = self._make_repo()
        for i in range(10):
            repo.log("login_success", user_id=i, username=f"user{i}")

        entries = repo.list_recent(limit=3)
        self.assertEqual(len(entries), 3)

    def test_list_recent_on_empty_file_returns_empty_list(self) -> None:
        repo, _ = self._make_repo()
        self.assertEqual(repo.list_recent(), [])

    def test_log_persists_all_optional_fields(self) -> None:
        repo, _ = self._make_repo()
        repo.log(
            "session_revoked",
            user_id=5,
            username="charlie",
            session_id="abc123",
            actor_id=1,
            actor_username="admin",
            ip_address="10.0.0.1",
            client_name="desktop-v1",
            details="manual revoke",
        )
        entries = repo.list_recent()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["event_type"], "session_revoked")
        self.assertEqual(entry["user_id"], 5)
        self.assertEqual(entry["actor_id"], 1)
        self.assertEqual(entry["actor_username"], "admin")
        self.assertEqual(entry["ip_address"], "10.0.0.1")
        self.assertEqual(entry["session_id"], "abc123")
        self.assertIn("created_at", entry)

    def test_log_with_none_user_id_for_failed_login(self) -> None:
        repo, _ = self._make_repo()
        repo.log("login_failure", username="ghost", details="Invalid credentials")
        entries = repo.list_recent()
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0]["user_id"])
        self.assertEqual(entries[0]["username"], "ghost")

    def test_file_survives_between_instances(self) -> None:
        """Entries written by one instance are readable by a second instance pointing to the same dir."""
        tmp = Path(tempfile.mkdtemp())
        repo1 = AuthAuditRepository(store_dir=tmp)
        repo1.log("login_success", user_id=10, username="persisted")

        repo2 = AuthAuditRepository(store_dir=tmp)
        entries = repo2.list_recent()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["username"], "persisted")
