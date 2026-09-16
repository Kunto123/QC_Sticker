from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.app.core.config import DEFAULT_STICKER_MODEL_META_PATH, DEFAULT_STICKER_MODEL_PATH
from backend.app.repositories.base_json import JsonRepository
from shared.contracts.templates import InspectionTemplate, template_from_dict

# Valid lifecycle states and the allowed forward transitions.
# "retired" can be reached from any state.
_LIFECYCLE_TRANSITIONS: dict[str, set[str]] = {
    "draft":     {"review", "retired"},
    "review":    {"approved", "draft", "retired"},
    "approved":  {"published", "review", "retired"},
    "published": {"retired"},
    "retired":   set(),
}
_ALL_LIFECYCLE_STATES = set(_LIFECYCLE_TRANSITIONS)


def _sample_template() -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "templates": [
            {
                "id": 1,
                "name": "QC Line A",
                "description": "Default sample template for operator flow.",
                "is_active": True,
                "lifecycle_status": "published",
                "current_version_id": 1,
                "created_at": now,
                "updated_at": now,
                "versions": [
                    {
                        "version_id": 1,
                        "version_number": 1,
                        "created_at": now,
                        "approved_by": None,
                        "approved_at": None,
                        "change_note": "Initial version",
                        "template": {
                            "id": 1,
                            "version_id": 1,
                            "version_number": 1,
                            "name": "QC Line A",
                            "description": "Default sample template for operator flow.",
                            "is_active": True,
                            "camera": {
                                "camera_index": 0,
                                "width": 640,
                                "height": 480,
                                "fps": 15,
                            },
                            "part_ready_roi": {
                                "x": 0.2,
                                "y": 0.2,
                                "w": 0.25,
                                "h": 0.25,
                                "width": 160,
                                "height": 120,
                            },
                            "sticker_roi": {
                                "x": 0.2,
                                "y": 0.2,
                                "w": 0.6,
                                "h": 0.6,
                                "width": 320,
                                "height": 240,
                            },
                            "vision": {
                                "model_path": DEFAULT_STICKER_MODEL_PATH or "models/dummy.pt",
                                "model_meta_path": DEFAULT_STICKER_MODEL_META_PATH or None,
                                "runtime": "ultralytics",
                                "conf_threshold": 0.25,
                                "stream_fps": 10,
                                "inference_fps": 4,
                                "imgsz": 640,
                                "classes": ["K0W-HB0", "K1Z-FA0", "K2S-H30"],
                                "enable_ergonomic_check": False,
                                "ergonomic_pose_model_path": None,
                                "ergonomic_min_keypoint_conf": 0.35,
                            },
                            "part_ready": {
                                "enabled": True,
                                "color_profile_id": None,
                                "colorspace": "LAB",
                                "distance_threshold": None,
                                "min_match_ratio": 0.75,
                            },
                            "sticker": {
                                "part_name": "Sample Part",
                                "expected_class": "K0W-HB0",
                                "line": "LINE-A",
                                "enabled": True,
                                "validator_mode": "ml_detection",
                                "min_roi_confidence": 0.0,
                                "min_class_confidence": None,
                                "max_offset_x": 80,
                                "max_offset_y": 80,
                                "tilt_gate_enabled": False,
                                "expected_tilt_degrees": 0.0,
                                "max_tilt_degrees": None,
                            },
                            "persistence": {"write_to_db": True},
                            "metadata": {},
                        },
                    }
                ],
            }
        ]
    }


class TemplatesRepository(JsonRepository):
    def __init__(self) -> None:
        super().__init__("templates.json", _sample_template())

    def _payload(self) -> dict[str, Any]:
        return self.load()

    def _normalize_template_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        template = template_from_dict(payload)
        return template.to_dict()

    def list_templates(self) -> list[dict[str, Any]]:
        return self._payload()["templates"]

    def list_summaries(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for template in self.list_templates():
            version = self.get_version(template["current_version_id"])
            # Extract mode from version template data for display
            _mode = "sticker"
            _validator_mode = "ml_detection"
            if version:
                _vt = version.get("template") or {}
                _mode = str(_vt.get("mode") or "sticker")
                _validator_mode = str(_vt.get("sticker", {}).get("validator_mode") or "ml_detection")
            items.append(
                {
                    "id": template["id"],
                    "name": template["name"],
                    "description": template.get("description"),
                    "is_active": bool(template.get("is_active", True)),
                    "lifecycle_status": template.get("lifecycle_status", "draft"),
                    "created_at": template.get("created_at"),
                    "updated_at": template.get("updated_at"),
                    "version_id": template.get("current_version_id"),
                    "version_number": version.get("version_number") if version else None,
                    "mode": _mode,
                    "sticker": {"validator_mode": _validator_mode},
                }
            )
        return items

    def get_template(self, template_id: int) -> dict[str, Any] | None:
        return next(
            (item for item in self.list_templates() if int(item["id"]) == int(template_id)),
            None,
        )

    def get_version(self, version_id: int | None) -> dict[str, Any] | None:
        if version_id is None:
            return None
        for template in self.list_templates():
            for version in template.get("versions") or []:
                if int(version["version_id"]) == int(version_id):
                    return version
        return None

    def get_template_detail(self, template_id: int) -> dict[str, Any] | None:
        template = self.get_template(template_id)
        if not template:
            return None
        version = self.get_version(template.get("current_version_id"))
        if not version:
            return None
        payload = self._normalize_template_payload(dict(version.get("template") or {}))
        payload["id"] = template["id"]
        payload["is_active"] = bool(template.get("is_active", True))
        return payload

    def get_version_detail(self, version_id: int) -> dict[str, Any] | None:
        version = self.get_version(version_id)
        if not version:
            return None

        raw_template = dict(version.get("template") or {})
        payload = self._normalize_template_payload(raw_template)
        template_id = int(payload.get("id") or raw_template.get("id") or 0)
        template_record = self.get_template(template_id) if template_id else None

        if template_id:
            payload["id"] = template_id
        if template_record is not None:
            payload["is_active"] = bool(template_record.get("is_active", payload.get("is_active", True)))
            payload["name"] = str(payload.get("name") or template_record.get("name") or "")
            payload["description"] = str(payload.get("description") or template_record.get("description") or "")
        return payload

    def get_by_version_id(self, version_id: int) -> InspectionTemplate | None:
        version = self.get_version(version_id)
        if not version:
            return None
        return template_from_dict(self._normalize_template_payload(version["template"]))

    def create_template(self, payload: dict[str, Any]) -> dict[str, Any]:
        store = self._payload()
        templates = store["templates"]
        template_id = self.next_id(templates)
        version_id = max(
            [int(version["version_id"]) for item in templates for version in item.get("versions", [])]
            or [0]
        ) + 1
        now = datetime.now(UTC).isoformat()
        change_note = str(payload.get("change_note") or "").strip() or "Initial version"
        template_payload = self._normalize_template_payload(
            {
                **dict(payload),
                "id": template_id,
                "version_id": version_id,
                "version_number": 1,
                "is_active": bool(payload.get("is_active", True)),
            }
        )
        record = {
            "id": template_id,
            "name": template_payload["name"],
            "description": template_payload.get("description", ""),
            "is_active": bool(template_payload.get("is_active", True)),
            "lifecycle_status": "draft",
            "current_version_id": version_id,
            "created_at": now,
            "updated_at": now,
            "versions": [
                {
                    "version_id": version_id,
                    "version_number": 1,
                    "created_at": now,
                    "approved_by": None,
                    "approved_at": None,
                    "change_note": change_note,
                    "template": template_payload,
                }
            ],
        }
        templates.append(record)
        self.save(store)
        return template_payload

    def update_template(self, template_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        store = self._payload()
        templates = store["templates"]
        version_id = max(
            [int(version["version_id"]) for item in templates for version in item.get("versions", [])]
            or [0]
        ) + 1
        now = datetime.now(UTC).isoformat()
        change_note = str(payload.get("change_note") or "").strip() or None
        for item in templates:
            if int(item["id"]) != int(template_id):
                continue
            current_versions = item.get("versions") or []
            version_number = max(int(v["version_number"]) for v in current_versions) + 1
            template_payload = self._normalize_template_payload(
                {
                    **dict(payload),
                    "id": int(template_id),
                    "version_id": version_id,
                    "version_number": version_number,
                    "is_active": bool(payload.get("is_active", item.get("is_active", True))),
                }
            )
            current_versions.append(
                {
                    "version_id": version_id,
                    "version_number": version_number,
                    "created_at": now,
                    "approved_by": None,
                    "approved_at": None,
                    "change_note": change_note,
                    "template": template_payload,
                }
            )
            item["name"] = template_payload["name"]
            item["description"] = template_payload.get("description", "")
            item["is_active"] = bool(template_payload.get("is_active", item.get("is_active", True)))
            item["lifecycle_status"] = "draft"  # new version resets to draft
            item["current_version_id"] = version_id
            item["updated_at"] = now
            self.save(store)
            return template_payload
        raise ValueError("Template not found.")

    def update_current_version(self, template_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        """Update the current version in-place without creating a new version entry."""
        store = self._payload()
        templates = store["templates"]
        now = datetime.now(UTC).isoformat()
        for item in templates:
            if int(item["id"]) != int(template_id):
                continue
            current_versions = item.get("versions") or []
            if not current_versions:
                raise ValueError("Template has no versions.")
            # Find the current version entry
            current_version_id = item.get("current_version_id")
            current_version = None
            for v in current_versions:
                if v["version_id"] == current_version_id:
                    current_version = v
                    break
            if current_version is None:
                # Fallback: use the last version
                current_version = current_versions[-1]
            # Normalize the updated payload
            template_payload = self._normalize_template_payload(
                {
                    **dict(payload),
                    "id": int(template_id),
                    "version_id": current_version["version_id"],
                    "version_number": current_version["version_number"],
                    "is_active": bool(payload.get("is_active", item.get("is_active", True))),
                }
            )
            # Replace the current version's template data in-place
            current_version["template"] = template_payload
            current_version["created_at"] = now
            # Update top-level item fields
            item["name"] = template_payload["name"]
            item["description"] = template_payload.get("description", "")
            item["is_active"] = bool(template_payload.get("is_active", item.get("is_active", True)))
            item["updated_at"] = now
            # Keep lifecycle_status as-is (don't reset to draft for in-place update)
            self.save(store)
            return template_payload
        raise ValueError("Template not found.")
        template = self.get_template(template_id)
        if not template:
            raise ValueError("Template not found.")
        return [
            {
                "version_id": v["version_id"],
                "version_number": v["version_number"],
                "created_at": v.get("created_at"),
                "approved_by": v.get("approved_by"),
                "approved_at": v.get("approved_at"),
                "change_note": v.get("change_note"),
                "is_current": v["version_id"] == template.get("current_version_id"),
            }
            for v in (template.get("versions") or [])
        ]

    def transition_lifecycle(
        self,
        template_id: int,
        new_status: str,
        *,
        actor_id: int | None = None,
        actor_username: str | None = None,
        change_note: str | None = None,
    ) -> dict[str, Any]:
        if new_status not in _ALL_LIFECYCLE_STATES:
            raise ValueError(f"Invalid lifecycle status '{new_status}'. Must be one of: {sorted(_ALL_LIFECYCLE_STATES)}")
        store = self._payload()
        for item in store["templates"]:
            if int(item["id"]) != int(template_id):
                continue
            current = item.get("lifecycle_status", "draft")
            allowed = _LIFECYCLE_TRANSITIONS.get(current, set())
            if new_status not in allowed:
                raise ValueError(
                    f"Cannot transition from '{current}' to '{new_status}'. "
                    f"Allowed transitions: {sorted(allowed) or 'none'}"
                )
            now = datetime.now(UTC).isoformat()
            item["lifecycle_status"] = new_status
            item["updated_at"] = now
            # Stamp approval metadata on the current version when approving/publishing
            if new_status in {"approved", "published"} and actor_id is not None:
                for version in item.get("versions") or []:
                    if version["version_id"] == item.get("current_version_id"):
                        version["approved_by"] = actor_id
                        version["approved_at"] = now
                        if change_note:
                            version["change_note"] = change_note
                        break
            self.save(store)
            return {
                "id": template_id,
                "lifecycle_status": new_status,
                "updated_at": now,
                "approved_by": actor_id,
                "actor_username": actor_username,
            }
        raise ValueError("Template not found.")

    def rollback_version(self, template_id: int, version_id: int) -> dict[str, Any]:
        """Set current_version_id to a previous version and reset lifecycle to 'draft'."""
        store = self._payload()
        for item in store["templates"]:
            if int(item["id"]) != int(template_id):
                continue
            version = next(
                (v for v in (item.get("versions") or []) if int(v["version_id"]) == int(version_id)),
                None,
            )
            if not version:
                raise ValueError(f"Version {version_id} not found for template {template_id}.")
            now = datetime.now(UTC).isoformat()
            item["current_version_id"] = int(version_id)
            item["lifecycle_status"] = "draft"
            item["updated_at"] = now
            self.save(store)
            return {
                "id": template_id,
                "current_version_id": int(version_id),
                "version_number": version["version_number"],
                "lifecycle_status": "draft",
                "updated_at": now,
            }
        raise ValueError("Template not found.")

    def delete_template(self, template_id: int) -> bool:
        store = self._payload()
        before = len(store["templates"])
        store["templates"] = [
            item for item in store["templates"] if int(item["id"]) != int(template_id)
        ]
        if len(store["templates"]) == before:
            return False
        self.save(store)
        return True
