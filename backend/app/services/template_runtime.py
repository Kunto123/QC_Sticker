from __future__ import annotations

from backend.app.repositories.deployments_repository import DeploymentsRepository
from backend.app.repositories.templates_repository import TemplatesRepository
from shared.contracts.templates import InspectionTemplate


class TemplateRuntimeService:
    def __init__(
        self,
        templates_repo: TemplatesRepository,
        deployments_repo: DeploymentsRepository,
    ) -> None:
        self._templates_repo = templates_repo
        self._deployments_repo = deployments_repo

    def resolve_template_by_version(self, version_id: int) -> InspectionTemplate:
        template = self._templates_repo.get_by_version_id(version_id)
        if not template:
            raise ValueError("Template version not found.")
        return template

    def get_active_deployment(self) -> dict | None:
        return self._deployments_repo.get_active()

