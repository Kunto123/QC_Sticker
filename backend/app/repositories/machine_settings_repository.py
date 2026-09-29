"""Repository Machine Settings — `data/json_store/machine_settings.json`.

Tidak ada seeding dari env: file yang tidak ada berarti pakai default dataclass.
Menyimpan selalu menulis schema version terkini, jadi file v1 otomatis
ter-upgrade pada PUT pertama.
"""
from __future__ import annotations

from backend.app.models.machine_settings import MachineSettings
from backend.app.repositories.base_json import JsonRepository


class MachineSettingsRepository(JsonRepository):
    def __init__(self) -> None:
        super().__init__("machine_settings.json", {})

    def load_settings(self) -> MachineSettings:
        raw = self.load()
        if not raw:
            return MachineSettings()
        return MachineSettings.from_dict(raw)

    def save_settings(self, settings: MachineSettings) -> None:
        self.save(settings.to_dict())
