from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from backend.app.repositories.base_json import JsonRepository

# Valid state machine transitions.
_TRANSITIONS: dict[str, set[str]] = {
    "queued":    {"running", "cancelled"},
    "running":   {"completed", "failed", "cancelled"},
    "completed": set(),
    "failed":    {"queued"},  # allow re-queue after failure
    "cancelled": {"queued"},  # allow re-queue after cancel
}
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def _coerce_int_param(params: dict, key: str, *, default: int, min_value: int, max_value: int) -> int:
    raw_value = params.get(key, default)
    if raw_value in (None, ""):
        value = default
    else:
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an integer") from exc
    if value < min_value or value > max_value:
        raise ValueError(f"{key} must be between {min_value} and {max_value}")
    params[key] = value
    return value


def _coerce_bool_param(params: dict, key: str, *, default: bool = False) -> bool:
    raw_value = params.get(key, default)
    if isinstance(raw_value, bool):
        value = raw_value
    else:
        value = str(raw_value or "").strip().lower() in {"1", "true", "yes", "on"}
    params[key] = value
    return value


def _normalize_progress(value: object, *, default: int = 0) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        parsed = default
    return min(100, max(0, parsed))


class TrainingRepository(JsonRepository):
    def __init__(self) -> None:
        super().__init__("training_jobs.json", {"jobs": []})

    def list_jobs(self) -> list[dict]:
        return self.load()["jobs"]

    def get_job(self, job_id: str) -> dict | None:
        return next((j for j in self.list_jobs() if j["id"] == job_id), None)

    def create_job(self, dataset_id: str, base_model: str, params: dict) -> dict:
        payload = self.load()
        items = payload["jobs"]
        now = datetime.now(UTC).isoformat()
        request_params = dict(params or {})
        _coerce_int_param(request_params, "epochs", default=1, min_value=1, max_value=1000)
        _coerce_int_param(request_params, "imgsz", default=320, min_value=64, max_value=2048)
        _coerce_int_param(request_params, "batch", default=4, min_value=1, max_value=256)
        _coerce_int_param(request_params, "patience", default=5, min_value=1, max_value=500)
        _coerce_int_param(request_params, "workers", default=0, min_value=0, max_value=32)
        _coerce_bool_param(request_params, "cache", default=False)
        base_model_spec = request_params.get("base_model_spec")
        if not isinstance(base_model_spec, dict):
            base_model_spec = {}
        dataset_version_spec = request_params.get("dataset_version_spec")
        if not isinstance(dataset_version_spec, dict):
            dataset_version_spec = {}
        requested_base_model = str(base_model or request_params.get("base_model") or "").strip()
        requested_device_mode = str(request_params.get("device_mode") or "auto").strip().lower() or "auto"
        if requested_device_mode not in {"auto", "gpu", "cpu"}:
            requested_device_mode = "auto"
        request_params["device_mode"] = requested_device_mode
        base_model_catalog_id = str(base_model_spec.get("id") or "").strip() or None
        base_model_family = str(request_params.get("base_model_family") or base_model_spec.get("family") or "").strip().lower() or None
        base_model_variant = str(request_params.get("base_model_variant") or base_model_spec.get("variant") or "").strip().lower() or None
        base_model_display_name = str(request_params.get("base_model_display_name") or base_model_spec.get("display_name") or "").strip() or None
        base_model_weights_name = str(request_params.get("base_model_weights_name") or base_model_spec.get("weights_name") or "").strip() or None
        base_model_runtime = str(request_params.get("base_model_runtime") or base_model_spec.get("runtime") or "").strip().lower() or None
        base_model_task = str(request_params.get("base_model_task") or base_model_spec.get("task") or "").strip().lower() or None
        base_model_source = str(request_params.get("base_model_source") or base_model_spec.get("source") or ("catalog" if base_model_spec else "legacy")).strip().lower() or ("catalog" if base_model_spec else "legacy")
        dataset_version_id = str(request_params.get("dataset_version_id") or dataset_version_spec.get("id") or "").strip() or None
        dataset_version_number = request_params.get("dataset_version_number") or dataset_version_spec.get("version_number")
        dataset_version_name = str(request_params.get("dataset_version_name") or dataset_version_spec.get("name") or "").strip() or None
        dataset_version_display_label = str(
            request_params.get("dataset_version_display_label") or dataset_version_spec.get("display_label") or ""
        ).strip() or None
        dataset_version_status = str(request_params.get("dataset_version_status") or dataset_version_spec.get("status") or "").strip().lower() or None
        dataset_version_export_format = str(
            request_params.get("dataset_version_export_format") or dataset_version_spec.get("export_format") or ""
        ).strip().lower() or None
        dataset_version_export_root = str(
            request_params.get("dataset_version_export_root") or dataset_version_spec.get("export_root") or ""
        ).strip() or None
        dataset_version_manifest_path = str(
            request_params.get("dataset_version_manifest_path") or dataset_version_spec.get("manifest_path") or ""
        ).strip() or None
        dataset_version_split_ratios = request_params.get("dataset_version_split_ratios") or dataset_version_spec.get("split_ratios")
        record: dict[str, Any] = {
            "id": uuid.uuid4().hex[:12],
            "dataset_id": dataset_id,
            "base_model": requested_base_model or base_model_catalog_id or "baseline",
            "base_model_catalog_id": base_model_catalog_id,
            "base_model_family": base_model_family,
            "base_model_variant": base_model_variant,
            "base_model_display_name": base_model_display_name,
            "base_model_weights_name": base_model_weights_name,
            "base_model_runtime": base_model_runtime,
            "base_model_task": base_model_task,
            "base_model_source": base_model_source,
            "dataset_version_id": dataset_version_id,
            "dataset_version_number": dataset_version_number,
            "dataset_version_name": dataset_version_name,
            "dataset_version_display_label": dataset_version_display_label,
            "dataset_version_status": dataset_version_status,
            "dataset_version_export_format": dataset_version_export_format,
            "dataset_version_export_root": dataset_version_export_root,
            "dataset_version_manifest_path": dataset_version_manifest_path,
            "dataset_version_split_ratios": dataset_version_split_ratios,
            "status": "queued",
            "trained_model_path": None,
            "params": request_params,
            "requested_device_mode": requested_device_mode,
            "effective_device": None,
            "device_backend": None,
            "device_fallback_reason": None,
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "progress_percent": 0,
            "progress_stage": "queued",
            "progress_message": "Training job queued.",
            "log": [
                f"Training job queued (device: {requested_device_mode}, base model: {base_model_display_name or base_model_catalog_id or requested_base_model or 'baseline'}"
                f"{', dataset version: ' + (dataset_version_display_label or dataset_version_name or dataset_version_id) if dataset_version_display_label or dataset_version_name or dataset_version_id else ''})."
            ],
        }
        items.append(record)
        self.save(payload)
        return record

    def update_job(self, job_id: str, *, log_line: str | None = None, **fields) -> dict:
        payload = self.load()
        for item in payload["jobs"]:
            if item["id"] != job_id:
                continue
            for key, value in fields.items():
                item[key] = value
            if log_line:
                item.setdefault("log", []).append(log_line)
            item["updated_at"] = datetime.now(UTC).isoformat()
            self.save(payload)
            return dict(item)
        raise ValueError(f"Training job '{job_id}' not found.")

    def transition(self, job_id: str, new_status: str, *, log_line: str | None = None, **extra_fields) -> dict:
        """Transition a job to new_status, raising ValueError if the transition is invalid."""
        payload = self.load()
        for item in payload["jobs"]:
            if item["id"] != job_id:
                continue
            current = item["status"]
            allowed = _TRANSITIONS.get(current, set())
            if new_status not in allowed:
                raise ValueError(
                    f"Cannot transition job '{job_id}' from '{current}' to '{new_status}'. "
                    f"Allowed: {sorted(allowed) or 'none'}"
                )
            now = datetime.now(UTC).isoformat()
            item["status"] = new_status
            if new_status == "running" and item.get("started_at") is None:
                item["started_at"] = now
                extra_fields.setdefault("progress_percent", 5)
                extra_fields.setdefault("progress_stage", "running")
                extra_fields.setdefault("progress_message", "Training job started.")
            if new_status in {"completed", "failed", "cancelled"}:
                item["finished_at"] = now
            if new_status == "completed":
                extra_fields.setdefault("progress_percent", 100)
                extra_fields.setdefault("progress_stage", "completed")
                extra_fields.setdefault("progress_message", "Training completed.")
            if new_status == "failed":
                extra_fields.setdefault("progress_percent", _normalize_progress(item.get("progress_percent"), default=95))
                extra_fields.setdefault("progress_stage", "failed")
                extra_fields.setdefault("progress_message", str(extra_fields.get("error") or "Training failed."))
            if new_status == "cancelled":
                extra_fields.setdefault("progress_percent", min(99, _normalize_progress(item.get("progress_percent"), default=0)))
                extra_fields.setdefault("progress_stage", "cancelled")
                extra_fields.setdefault("progress_message", "Training cancelled.")
            for key, value in extra_fields.items():
                item[key] = value
            if log_line:
                item.setdefault("log", []).append(log_line)
            self.save(payload)
            return dict(item)
        raise ValueError(f"Training job '{job_id}' not found.")

    def cancel_job(self, job_id: str) -> dict:
        job = self.get_job(job_id)
        if job is None:
            raise ValueError("Training job not found.")
        if job["status"] not in {"queued", "running"}:
            raise ValueError(f"Cannot cancel job in status '{job['status']}'.")
        return self.transition(
            job_id,
            "cancelled",
            log_line="Job cancelled by user.",
            progress_message="Job cancelled by user.",
        )

    def delete_job(self, job_id: str) -> dict:
        payload = self.load()
        items = payload["jobs"]
        for index, item in enumerate(items):
            if item["id"] != job_id:
                continue
            status = str(item.get("status") or "").strip().lower()
            if status not in _TERMINAL_STATUSES:
                raise ValueError(f"Cannot delete training job in status '{status}'.")
            removed = dict(item)
            del items[index]
            self.save(payload)
            return removed
        raise ValueError("Training job not found.")

