from __future__ import annotations

from datetime import UTC, datetime

from backend.app.repositories.base_json import JsonRepository


_UNSET = object()


class DeploymentsRepository(JsonRepository):
    def __init__(self) -> None:
        super().__init__("deployments.json", {"deployments": []})

    def list_deployments(self) -> list[dict]:
        return self.load()["deployments"]

    def get_deployment(self, deployment_id: int) -> dict | None:
        return next(
            (item for item in self.list_deployments() if int(item.get("id") or 0) == int(deployment_id)),
            None,
        )

    def deploy(
        self,
        *,
        template_id: int,
        template_version_id: int,
        deployed_by: int | None,
        template_name: str,
        version_number: int,
    ) -> dict:
        payload = self.load()
        items = payload["deployments"]
        now = datetime.now(UTC).isoformat()
        record = {
            "id": self.next_id(items),
            "template_id": int(template_id),
            "template_version_id": int(template_version_id),
            "is_active": True,
            "deployed_by": deployed_by,
            "effective_from": now,
            "effective_until": None,
            "created_at": now,
            "template_name": template_name,
            "version_number": version_number,
        }
        items.append(record)
        self.save(payload)
        return record

    def get_active(self) -> dict | None:
        return next(
            (
                item
                for item in reversed(self.list_deployments())
                if item["is_active"]
            ),
            None,
        )

    def deactivate(self, deployment_id: int) -> bool:
        payload = self.load()
        now = datetime.now(UTC).isoformat()
        for item in payload["deployments"]:
            if int(item["id"]) == int(deployment_id) and item["is_active"]:
                item["is_active"] = False
                item["effective_until"] = now
                self.save(payload)
                return True
        return False

    def update_deployment(
        self,
        deployment_id: int,
        *,
        template_version_id: object = _UNSET,
        template_name: object = _UNSET,
        version_number: object = _UNSET,
        deployed_by: object = _UNSET,
    ) -> dict:
        if (
            template_version_id is _UNSET
            and template_name is _UNSET
            and version_number is _UNSET
            and deployed_by is _UNSET
        ):
            raise ValueError("At least one mutable field is required.")

        payload = self.load()
        items = payload["deployments"]
        now = datetime.now(UTC).isoformat()

        for item in items:
            if int(item.get("id") or 0) != int(deployment_id):
                continue
            if not bool(item.get("is_active")):
                raise ValueError("Inactive deployment cannot be updated.")

            if template_version_id is not _UNSET:
                item["template_version_id"] = int(template_version_id)

            if template_name is not _UNSET:
                normalized_template_name = str(template_name or "").strip()
                item["template_name"] = normalized_template_name or None

            if version_number is not _UNSET:
                item["version_number"] = int(version_number)

            if deployed_by is not _UNSET:
                item["deployed_by"] = None if deployed_by in (None, "") else int(deployed_by)

            item["updated_at"] = now
            self.save(payload)
            return dict(item)

        raise ValueError(f"Deployment {deployment_id} not found.")

    def rollback(self, deployment_id: int, *, rolled_back_by: int | None = None) -> dict:
        """Deactivate deployment_id and re-deploy the previous version."""
        payload = self.load()
        items = payload["deployments"]
        now = datetime.now(UTC).isoformat()

        target = next((d for d in items if int(d["id"]) == int(deployment_id)), None)
        if target is None:
            raise ValueError(f"Deployment {deployment_id} not found.")

        # Find the most recent *other* deployment
        previous = next(
            (
                d for d in reversed(items)
                if int(d["id"]) != int(deployment_id)
            ),
            None,
        )
        if previous is None:
            raise ValueError("No previous deployment found to roll back to.")

        target["is_active"] = False
        target["effective_until"] = now

        # Create a new deployment record based on the previous one
        record = {
            "id": self.next_id(items),
            "template_id": previous["template_id"],
            "template_version_id": previous["template_version_id"],
            "is_active": True,
            "deployed_by": rolled_back_by,
            "effective_from": now,
            "effective_until": None,
            "created_at": now,
            "template_name": previous.get("template_name"),
            "version_number": previous.get("version_number"),
            "rollback_from_deployment_id": int(deployment_id),
        }
        items.append(record)
        self.save(payload)
        return record

