"""Machine Settings repository — `data/json_store/machine_settings.json`.

No env seeding: a missing file means dataclass defaults. Saving always writes
the current schema version, so a v1 file is upgraded on the first PUT.
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
