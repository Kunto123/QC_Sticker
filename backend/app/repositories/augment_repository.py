from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from backend.app.repositories.base_json import JsonRepository

_TRANSITIONS: dict[str, set[str]] = {
    "queued":    {"running", "cancelled"},
    "running":   {"completed", "failed", "cancelled"},
    "completed": set(),
    "failed":    {"queued"},
    "cancelled": {"queued"},
}
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class AugmentRepository(JsonRepository):
    def __init__(self) -> None:
        super().__init__("augment_jobs.json", {"jobs": []})

    def list_jobs(self) -> list[dict]:
        return self.load()["jobs"]

    def get_job(self, job_id: str) -> dict | None:
        return next((j for j in self.list_jobs() if j["id"] == job_id), None)

    def create_job(
        self,
        dataset_id: str,
        transforms: list[str],
        multiplier: int = 2,
        params: dict | None = None,
    ) -> dict:
        payload = self.load()
        now = datetime.now(UTC).isoformat()
        record: dict[str, Any] = {
            "id": uuid.uuid4().hex[:12],
            "dataset_id": dataset_id,
            "transforms": transforms,
            "multiplier": max(1, int(multiplier)),
            "params": params or {},
            "status": "queued",
            "source_image_count": 0,
            "augmented_image_count": 0,
            "output_dataset_id": None,
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "log": ["Augmentation job queued."],
        }
        payload["jobs"].append(record)
        self.save(payload)
        return record

    def transition(
        self,
        job_id: str,
        new_status: str,
        *,
        log_line: str | None = None,
        **extra_fields,
    ) -> dict:
        payload = self.load()
        for item in payload["jobs"]:
            if item["id"] != job_id:
                continue
            current = item["status"]
            allowed = _TRANSITIONS.get(current, set())
            if new_status not in allowed:
                raise ValueError(
                    f"Cannot transition augment job '{job_id}' from '{current}' to '{new_status}'. "
                    f"Allowed: {sorted(allowed) or 'none'}"
                )
            now = datetime.now(UTC).isoformat()
            item["status"] = new_status
            if new_status == "running" and item.get("started_at") is None:
                item["started_at"] = now
            if new_status in {"completed", "failed", "cancelled"}:
                item["finished_at"] = now
            for key, value in extra_fields.items():
                item[key] = value
            if log_line:
                item.setdefault("log", []).append(log_line)
            self.save(payload)
            return dict(item)
        raise ValueError(f"Augment job '{job_id}' not found.")

    def cancel_job(self, job_id: str) -> dict:
        job = self.get_job(job_id)
        if job is None:
            raise ValueError("Augment job not found.")
        if job["status"] not in {"queued", "running"}:
            raise ValueError(f"Cannot cancel job in status '{job['status']}'.")
        return self.transition(job_id, "cancelled", log_line="Job cancelled by user.")

    def delete_job(self, job_id: str) -> dict:
        payload = self.load()
        items = payload["jobs"]
        for index, item in enumerate(items):
            if item["id"] != job_id:
                continue
            status = str(item.get("status") or "").strip().lower()
            if status not in _TERMINAL_STATUSES:
                raise ValueError(f"Cannot delete augment job in status '{status}'.")
            removed = dict(item)
            del items[index]
            self.save(payload)
            return removed
        raise ValueError("Augment job not found.")
