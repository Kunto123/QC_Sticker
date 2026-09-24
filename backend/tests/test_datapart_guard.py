"""DatapartGuardService / DatapartGuardRepository.

Every DATAPART_GUARD_BATCH_SIZE accepted judgements pushed to the SQL
mirror, further commits must lock until a downstream MES process has filled
DatapartID for all of them. State must be durable (survive a backend
restart) — a purely in-memory cache would silently forget a pending batch
across a restart, which was the original design flaw flagged before this
was built.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.tests._test_env import ensure_test_data_root  # noqa: E402

TEST_DATA_ROOT = ensure_test_data_root()

from backend.app.repositories.datapart_guard_repository import DatapartGuardRepository  # noqa: E402
from backend.app.services.datapart_guard_service import (  # noqa: E402
    DATAPART_GUARD_BATCH_SIZE,
    DatapartGuardService,
)


class _FakeMirrorRepo:
    def __init__(self, unfilled: list[int] | None = None) -> None:
        self.unfilled = list(unfilled or [])
        self.calls: list[list[int]] = []

    def unfilled_datapart_ids(self, mirror_ids: list[int]) -> list[int]:
        self.calls.append(list(mirror_ids))
        return [i for i in mirror_ids if i in self.unfilled]


class DatapartGuardServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = DatapartGuardRepository()
        # Isolate each test from whatever a previous test left on disk.
        self.repo.save_state({"pending_ids": [], "locked": False})

    def test_batch_size_is_five(self) -> None:
        self.assertEqual(DATAPART_GUARD_BATCH_SIZE, 5)

    def test_locks_after_batch_size_pushes(self) -> None:
        mirror = _FakeMirrorRepo()
        service = DatapartGuardService(self.repo, mirror)

        for i in range(1, DATAPART_GUARD_BATCH_SIZE):
            service.record_pushed(i)
            self.assertFalse(service.is_locked(), f"locked too early at push #{i}")

        service.record_pushed(DATAPART_GUARD_BATCH_SIZE)

        self.assertTrue(service.is_locked())
        self.assertEqual(service.status()["pending_ids"], list(range(1, DATAPART_GUARD_BATCH_SIZE + 1)))

    def test_ignores_none_mirror_id(self) -> None:
        """A push that never made it to SQL (mirror_id None) must not count
        toward the batch — it never got a No/id to check DatapartID against."""
        mirror = _FakeMirrorRepo()
        service = DatapartGuardService(self.repo, mirror)

        for _ in range(DATAPART_GUARD_BATCH_SIZE):
            service.record_pushed(None)

        self.assertFalse(service.is_locked())
        self.assertEqual(service.status()["pending_ids"], [])

    def test_does_not_grow_batch_further_once_locked(self) -> None:
        mirror = _FakeMirrorRepo()
        service = DatapartGuardService(self.repo, mirror)
        for i in range(1, DATAPART_GUARD_BATCH_SIZE + 1):
            service.record_pushed(i)
        self.assertTrue(service.is_locked())

        service.record_pushed(999)

        self.assertEqual(service.status()["pending_ids"], list(range(1, DATAPART_GUARD_BATCH_SIZE + 1)))

    def test_state_survives_a_simulated_restart(self) -> None:
        """A new DatapartGuardService instance (simulating a fresh backend
        process) built from the same repo must still see the pending batch
        and lock — this is the whole point of not using an in-memory-only
        cache."""
        mirror = _FakeMirrorRepo()
        first_process = DatapartGuardService(self.repo, mirror)
        for i in range(1, DATAPART_GUARD_BATCH_SIZE + 1):
            first_process.record_pushed(i)
        self.assertTrue(first_process.is_locked())

        second_process = DatapartGuardService(self.repo, mirror)

        self.assertTrue(second_process.is_locked())
        self.assertEqual(second_process.status()["pending_ids"], list(range(1, DATAPART_GUARD_BATCH_SIZE + 1)))

    def test_poll_once_stays_locked_while_any_id_unfilled(self) -> None:
        ids = list(range(1, DATAPART_GUARD_BATCH_SIZE + 1))
        mirror = _FakeMirrorRepo(unfilled=[ids[-1]])
        service = DatapartGuardService(self.repo, mirror)
        for i in ids:
            service.record_pushed(i)
        self.assertTrue(service.is_locked())

        service.poll_once()

        self.assertTrue(service.is_locked())
        self.assertEqual(mirror.calls, [ids])

    def test_poll_once_unlocks_once_all_ids_filled(self) -> None:
        ids = list(range(1, DATAPART_GUARD_BATCH_SIZE + 1))
        mirror = _FakeMirrorRepo(unfilled=[])
        service = DatapartGuardService(self.repo, mirror)
        for i in ids:
            service.record_pushed(i)
        self.assertTrue(service.is_locked())

        service.poll_once()

        self.assertFalse(service.is_locked())
        self.assertEqual(service.status()["pending_ids"], [])

        # And the unlocked state is itself durable across a restart.
        after_restart = DatapartGuardService(self.repo, mirror)
        self.assertFalse(after_restart.is_locked())

    def test_poll_once_is_a_noop_while_unlocked(self) -> None:
        mirror = _FakeMirrorRepo()
        service = DatapartGuardService(self.repo, mirror)

        service.poll_once()

        self.assertEqual(mirror.calls, [])

    def test_override_clears_lock_and_records_who(self) -> None:
        mirror = _FakeMirrorRepo(unfilled=[1, 2, 3])
        service = DatapartGuardService(self.repo, mirror)
        for i in range(1, DATAPART_GUARD_BATCH_SIZE + 1):
            service.record_pushed(i)
        self.assertTrue(service.is_locked())

        service.override(by="admin_dwiku")

        self.assertFalse(service.is_locked())
        status = service.status()
        self.assertEqual(status["pending_ids"], [])
        self.assertEqual(status["last_override_by"], "admin_dwiku")
        self.assertIsNotNone(status["last_override_at"])

        # Override survives a restart too — not just cleared in memory.
        after_restart = DatapartGuardService(self.repo, mirror)
        self.assertFalse(after_restart.is_locked())
        self.assertEqual(after_restart.status()["last_override_by"], "admin_dwiku")

    def test_no_mirror_repo_means_never_locks(self) -> None:
        """Local-only backend (no SQL mirror configured) — the guard must
        stay a permanent no-op rather than lock a line that has nowhere to
        push judgements at all."""
        service = DatapartGuardService(self.repo, None)

        for i in range(1, DATAPART_GUARD_BATCH_SIZE + 5):
            service.record_pushed(i)

        self.assertFalse(service.is_locked())


if __name__ == "__main__":
    unittest.main()
