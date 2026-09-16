from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.app.repositories.inspection_results_repository import InspectionResultsRepository


class HybridInspectionResultsRepository:
    RETRYABLE_PUSH_STATUSES = frozenset({"failed", "pending"})

    def __init__(
        self,
        local_repo: InspectionResultsRepository,
        sql_mirror_repo=None,
    ) -> None:
        self._local_repo = local_repo
        self._sql_mirror_repo = sql_mirror_repo

    def _utcnow_iso(self) -> str:
        return datetime.now(UTC).isoformat()

    def _apply_mirror_result(self, record: dict[str, Any]) -> dict[str, Any]:
        if self._sql_mirror_repo is None:
            raise ValueError("SQL mirror push is not configured.")

        attempt_at = self._utcnow_iso()
        try:
            mirror_record = self._sql_mirror_repo.create_result(record)
        except Exception as exc:  # noqa: BLE001
            retry_count = int(record.get("retry_count") or 0) + 1
            patch = {
                "push_status": "failed",
                "retry_count": retry_count,
                "last_push_error": str(exc),
                "last_push_attempt_at": attempt_at,
            }
            updated = self._local_repo.update_result(int(record["id"]), patch)
            record.update(updated)
            return record

        patch = {
            "push_status": "sent",
            "last_push_error": None,
            "sql_mirror_id": mirror_record.get("id"),
            "last_push_attempt_at": attempt_at,
            "last_pushed_at": attempt_at,
        }
        updated = self._local_repo.update_result(int(record["id"]), patch)
        record.update(updated)
        return record

    def create_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        initial_status = "pending" if self._sql_mirror_repo is not None else "local_only"
        record = self._local_repo.create_result(
            {
                **dict(payload),
                "push_status": initial_status,
                "retry_count": int(payload.get("retry_count") or 0),
                "last_push_error": None,
            }
        )
        if self._sql_mirror_repo is None:
            return record
        return self._apply_mirror_result(record)

    def retry_result(self, result_id: int) -> dict[str, Any]:
        record = self._local_repo.get_result(int(result_id))
        if record is None:
            raise ValueError("Inspection result not found.")

        push_status = str(record.get("push_status") or "").strip().lower()
        if push_status not in self.RETRYABLE_PUSH_STATUSES:
            raise ValueError(
                f"Inspection result #{result_id} cannot be retried from push_status `{push_status or '-'}`."
            )
        return self._apply_mirror_result(record)

    def retry_failed(self, *, result_ids: list[int] | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if self._sql_mirror_repo is None:
            raise ValueError("SQL mirror push is not configured.")

        records: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        if result_ids:
            for result_id in result_ids:
                record = self._local_repo.get_result(int(result_id))
                if record is None:
                    continue
                push_status = str(record.get("push_status") or "").strip().lower()
                if push_status not in self.RETRYABLE_PUSH_STATUSES:
                    continue
                numeric_id = int(record["id"])
                if numeric_id in seen_ids:
                    continue
                records.append(record)
                seen_ids.add(numeric_id)
                if len(records) >= limit:
                    break
        else:
            pending_records = self._local_repo.list_results(push_status="pending", limit=limit, offset=0)
            failed_records = self._local_repo.list_results(push_status="failed", limit=limit, offset=0)
            for record in [*pending_records, *failed_records]:
                numeric_id = int(record["id"])
                if numeric_id in seen_ids:
                    continue
                records.append(record)
                seen_ids.add(numeric_id)
                if len(records) >= limit:
                    break

        return [self._apply_mirror_result(record) for record in records]

    def list_results(
        self,
        *,
        line_id: str | None = None,
        station_id: str | None = None,
        part_name: str | None = None,
        template_version_id: int | None = None,
        decision_code: str | None = None,
        push_status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self._local_repo.list_results(
            line_id=line_id,
            station_id=station_id,
            part_name=part_name,
            template_version_id=template_version_id,
            decision_code=decision_code,
            push_status=push_status,
            limit=limit,
            offset=offset,
        )

    def get_result(self, result_id: int) -> dict[str, Any] | None:
        return self._local_repo.get_result(result_id)

    def update_result(self, result_id: int, patch: dict[str, Any], *, requeue_mirror: bool = False) -> dict[str, Any]:
        result = self._local_repo.update_result(result_id, patch)
        if self._sql_mirror_repo is not None and requeue_mirror:
            result = self._local_repo.update_result(
                result_id,
                {
                    "push_status": "pending",
                    "retry_count": 0,
                    "last_push_error": None,
                },
            )
        return result

    def delete_result(self, result_id: int) -> dict[str, Any]:
        existing = self._local_repo.get_result(result_id)
        deleted = self._local_repo.delete_result(result_id)

        if existing is None or self._sql_mirror_repo is None:
            return deleted

        mirror_id = existing.get("sql_mirror_id")
        delete_mirror = getattr(self._sql_mirror_repo, "delete_result", None)
        if mirror_id is None or not callable(delete_mirror):
            return deleted

        try:
            delete_mirror(int(mirror_id))
        except Exception:  # noqa: BLE001
            # Mirror cleanup is best-effort; local source of truth is already deleted.
            pass
        return deleted

    def summary(
        self,
        *,
        line_id: str | None = None,
        station_id: str | None = None,
        part_name: str | None = None,
        template_version_id: int | None = None,
        decision_code: str | None = None,
    ) -> dict[str, Any]:
        return self._local_repo.summary(
            line_id=line_id,
            station_id=station_id,
            part_name=part_name,
            template_version_id=template_version_id,
            decision_code=decision_code,
        )

    def buckets(
        self,
        *,
        line_id: str | None = None,
        station_id: str | None = None,
        part_name: str | None = None,
        template_version_id: int | None = None,
        decision_code: str | None = None,
        granularity: str = "hour",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        return self._local_repo.buckets(
            line_id=line_id,
            station_id=station_id,
            part_name=part_name,
            template_version_id=template_version_id,
            decision_code=decision_code,
            granularity=granularity,
            limit=limit,
        )
