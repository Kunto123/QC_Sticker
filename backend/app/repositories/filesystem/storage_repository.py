from __future__ import annotations

from pathlib import Path

from backend.app.core.config import DATASETS_DIR, MODELS_DIR


class FilesystemStorageRepository:
    def __init__(self, *, datasets_root: Path = DATASETS_DIR, models_root: Path = MODELS_DIR) -> None:
        self.datasets_root = Path(datasets_root)
        self.models_root = Path(models_root)
        self.datasets_root.mkdir(parents=True, exist_ok=True)
        self.models_root.mkdir(parents=True, exist_ok=True)

    def dataset_path(self, dataset_id: str, target: str = "images") -> Path:
        path = self.datasets_root / str(dataset_id) / str(target)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def model_path(self, relative_path: str) -> Path:
        path = self.models_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_dataset_file(self, dataset_id: str, file_name: str, content: bytes, *, target: str = "images") -> Path:
        destination = self.dataset_path(dataset_id, target) / file_name
        destination.write_bytes(content)
        return destination

    def write_model_file(self, relative_path: str, content: bytes) -> Path:
        destination = self.model_path(relative_path)
        destination.write_bytes(content)
        return destination

    def list_dataset_files(self, dataset_id: str, *, target: str = "images") -> list[dict[str, str | int]]:
        root = self.dataset_path(dataset_id, target)
        items: list[dict[str, str | int]] = []
        for path in sorted(root.glob("*")):
            if path.is_file():
                items.append({"name": path.name, "path": str(path), "size": path.stat().st_size})
        return items
