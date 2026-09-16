from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.core.config import DEFAULT_STICKER_MODEL_META_PATH, DEFAULT_STICKER_MODEL_PATH, MODELS_DIR, PROJECT_ROOT
from backend.app.repositories.base_json import JsonRepository

# Lifecycle: draft → validated → canary → production → retired
# Any state → retired; never go backwards (except re-draft after retired)
_MODEL_TRANSITIONS: dict[str, set[str]] = {
    "draft":      {"validated", "retired"},
    "validated":  {"canary", "draft", "retired"},
    "canary":     {"production", "validated", "retired"},
    "production": {"retired"},
    "retired":    {"draft"},
}
_ALL_MODEL_STATES = set(_MODEL_TRANSITIONS)


def _file_sha256(path: str) -> str | None:
    try:
        p = Path(path)
        if not p.exists():
            return None
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return None


def _path_candidates(raw_path: str) -> set[str]:
    value = str(raw_path or "").strip()
    if not value:
        return set()
    path = Path(value)
    candidates = {value.replace("/", "\\").lower()}
    if path.name:
        candidates.add(path.name.lower())
    resolved_candidates = [path] if path.is_absolute() else [PROJECT_ROOT / path, MODELS_DIR / path, MODELS_DIR / path.name]
    for candidate in resolved_candidates:
        try:
            candidates.add(str(candidate.resolve()).replace("/", "\\").lower())
        except Exception:  # noqa: BLE001
            continue
    return candidates


def _load_default_meta() -> dict[str, Any]:
    path = Path(DEFAULT_STICKER_MODEL_META_PATH)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _default_models_payload() -> dict[str, Any]:
    meta = _load_default_meta()
    now = datetime.now(UTC).isoformat()
    if not DEFAULT_STICKER_MODEL_PATH:
        return {"models": []}
    return {
        "models": [
            {
                "id": 1,
                "name": "AKH Sticker Detector",
                "path": DEFAULT_STICKER_MODEL_PATH,
                "meta_path": DEFAULT_STICKER_MODEL_META_PATH or None,
                "source": "seeded-default",
                "runtime": "ultralytics",
                "task": "detection",
                "architecture_family": meta.get("architecture_family") or "yolov5",
                "architecture_variant": meta.get("architecture_variant") or "unknown",
                "class_names": list(meta.get("class_names") or []),
                "lifecycle_status": "production",
                "checksum_sha256": _file_sha256(DEFAULT_STICKER_MODEL_PATH),
                "provenance": {"source_dataset_id": None, "training_job_id": None},
                "created_at": now,
            }
        ]
    }


class ModelsRepository(JsonRepository):
    def __init__(self) -> None:
        super().__init__("models.json", _default_models_payload())

    def list_models(self) -> list[dict]:
        return self.load()["models"]

    def get_model(self, model_id: int) -> dict | None:
        return next((item for item in self.list_models() if int(item["id"]) == int(model_id)), None)

    def find_by_path(self, path: str) -> dict | None:
        normalized = str(path or "").strip().lower()
        return next(
            (item for item in self.list_models() if str(item.get("path") or "").strip().lower() == normalized),
            None,
        )

    def find_by_name(self, name: str) -> dict | None:
        normalized = str(name or "").strip().lower()
        if not normalized:
            return None
        return next(
            (item for item in self.list_models() if str(item.get("name") or "").strip().lower() == normalized),
            None,
        )

    def find_active_model_conflict(self, model_path: str, *, templates_repo: Any, deployments_repo: Any) -> dict | None:
        model_candidates = _path_candidates(model_path)
        if not model_candidates:
            return None
        for deployment in deployments_repo.list_deployments():
            if not deployment.get("is_active"):
                continue
            version = templates_repo.get_version(int(deployment.get("template_version_id") or 0))
            if version is None:
                continue
            template_payload = dict(version.get("template") or {})
            vision_payload = dict(template_payload.get("vision") or {})
            deployed_model_path = str(vision_payload.get("model_path") or "").strip()
            if not deployed_model_path:
                continue
            if model_candidates & _path_candidates(deployed_model_path):
                return {
                    "deployment_id": deployment.get("id"),
                    "line_id": deployment.get("line_id"),
                    "station_id": deployment.get("station_id"),
                    "template_id": deployment.get("template_id"),
                    "template_version_id": deployment.get("template_version_id"),
                    "template_model_path": deployed_model_path,
                }
        return None

    def list_model_deployment_references(self, model_path: str, *, templates_repo: Any, deployments_repo: Any) -> list[dict[str, Any]]:
        model_candidates = _path_candidates(model_path)
        if not model_candidates:
            return []

        references: list[dict[str, Any]] = []
        for deployment in deployments_repo.list_deployments():
            version = templates_repo.get_version(int(deployment.get("template_version_id") or 0))
            if version is None:
                continue
            template_payload = dict(version.get("template") or {})
            vision_payload = dict(template_payload.get("vision") or {})
            deployed_model_path = str(vision_payload.get("model_path") or "").strip()
            if not deployed_model_path or not (model_candidates & _path_candidates(deployed_model_path)):
                continue
            references.append(
                {
                    "deployment_id": deployment.get("id"),
                    "line_id": deployment.get("line_id"),
                    "station_id": deployment.get("station_id"),
                    "template_id": deployment.get("template_id"),
                    "template_version_id": deployment.get("template_version_id"),
                    "template_name": deployment.get("template_name") or template_payload.get("name"),
                    "version_number": deployment.get("version_number"),
                    "is_active": bool(deployment.get("is_active")),
                    "effective_from": deployment.get("effective_from"),
                    "effective_until": deployment.get("effective_until"),
                    "model_path": deployed_model_path,
                }
            )
        return references

    def add_model(
        self,
        name: str,
        path: str,
        source: str = "manual",
        *,
        meta_path: str | None = None,
        runtime: str = "ultralytics",
        task: str = "detection",
        class_names: list[str] | None = None,
        architecture_family: str | None = None,
        architecture_variant: str | None = None,
        source_dataset_id: str | None = None,
        training_job_id: str | None = None,
    ) -> dict:
        payload = self.load()
        items = payload["models"]
        record = {
            "id": self.next_id(items),
            "name": name,
            "path": path,
            "meta_path": meta_path,
            "source": source,
            "runtime": runtime,
            "task": task,
            "class_names": list(class_names or []),
            "architecture_family": architecture_family,
            "architecture_variant": architecture_variant,
            "lifecycle_status": "draft",
            "checksum_sha256": _file_sha256(path),
            "provenance": {
                "source_dataset_id": source_dataset_id,
                "training_job_id": training_job_id,
            },
            "created_at": datetime.now(UTC).isoformat(),
        }
        items.append(record)
        self.save(payload)
        return record

    def transition_lifecycle(
        self,
        model_id: int,
        new_status: str,
        *,
        actor_id: int | None = None,
        note: str | None = None,
    ) -> dict:
        if new_status not in _ALL_MODEL_STATES:
            raise ValueError(f"Invalid lifecycle status '{new_status}'. Must be one of: {sorted(_ALL_MODEL_STATES)}")
        payload = self.load()
        for item in payload["models"]:
            if int(item["id"]) != int(model_id):
                continue
            current = item.get("lifecycle_status", "draft")
            allowed = _MODEL_TRANSITIONS.get(current, set())
            if new_status not in allowed:
                raise ValueError(
                    f"Cannot transition model {model_id} from '{current}' to '{new_status}'. "
                    f"Allowed: {sorted(allowed) or 'none'}"
                )
            now = datetime.now(UTC).isoformat()
            item["lifecycle_status"] = new_status
            item["updated_at"] = now
            if note:
                item.setdefault("lifecycle_notes", []).append({"status": new_status, "note": note, "at": now, "by": actor_id})
            self.save(payload)
            return dict(item)
        raise ValueError(f"Model {model_id} not found.")

    def update_model(self, model_id: int, *, name: str) -> dict:
        name = str(name or "").strip()
        if not name:
            raise ValueError("Model name must not be empty.")
        payload = self.load()
        for item in payload["models"]:
            if int(item["id"]) != int(model_id):
                continue
            if str(item.get("source") or "").strip().lower() == "seeded-default":
                raise ValueError(f"Model {model_id} is a seeded-default and cannot be renamed.")
            item["name"] = name
            item["updated_at"] = datetime.now(UTC).isoformat()
            self.save(payload)
            return dict(item)
        raise ValueError(f"Model {model_id} not found.")

    def delete_model(self, model_id: int) -> dict[str, Any]:
        payload = self.load()
        items = payload["models"]
        for index, item in enumerate(items):
            if int(item["id"]) != int(model_id):
                continue
            if str(item.get("source") or "").strip().lower() == "seeded-default":
                raise ValueError("Default seeded model cannot be deleted.")
            removed = dict(item)
            del items[index]
            self.save(payload)
            return removed
        raise ValueError(f"Model {model_id} not found.")
