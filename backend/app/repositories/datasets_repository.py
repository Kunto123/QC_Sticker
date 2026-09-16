from __future__ import annotations

import json
import shutil
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from backend.app.core.config import DATASETS_DIR, ensure_data_dirs
from backend.app.repositories.base_json import JsonRepository


_ALLOWED_DATASET_TARGETS = {"images", "labels", "exports"}
_UNSET = object()


def _normalize_target(target: str) -> str:
    value = str(target or "").strip().lower()
    if value not in _ALLOWED_DATASET_TARGETS:
        raise ValueError(f"Invalid dataset target '{target}'. Must be one of: {sorted(_ALLOWED_DATASET_TARGETS)}")
    return value


class DatasetsRepository(JsonRepository):
    def __init__(self) -> None:
        ensure_data_dirs()
        super().__init__("datasets.json", {"datasets": []})

    def list_datasets(self) -> list[dict]:
        return [self._enrich_dataset(item) for item in self.load()["datasets"]]

    def create_dataset(self, name: str, description: str = "") -> dict:
        payload = self.load()
        items = payload["datasets"]
        dataset_id = uuid.uuid4().hex[:12]
        dataset_dir = DATASETS_DIR / dataset_id
        for subdir in ("images", "labels", "exports"):
            (dataset_dir / subdir).mkdir(parents=True, exist_ok=True)
        record = {
            "id": dataset_id,
            "name": name,
            "description": description,
            "folder_name": dataset_id,
            "created_at": datetime.now(UTC).isoformat(),
        }
        items.append(record)
        self.save(payload)
        return record

    def update_dataset(
        self,
        dataset_id: str,
        *,
        name: object = _UNSET,
        description: object = _UNSET,
    ) -> dict:
        payload = self.load()
        if name is _UNSET and description is _UNSET:
            raise ValueError("At least one of name or description is required")

        for item in payload["datasets"]:
            if item["id"] != dataset_id:
                continue

            if name is not _UNSET:
                normalized_name = str(name or "").strip()
                if not normalized_name:
                    raise ValueError("Dataset name is required")
                item["name"] = normalized_name

            if description is not _UNSET:
                item["description"] = str(description or "")

            item["updated_at"] = datetime.now(UTC).isoformat()
            self.save(payload)
            return self._enrich_dataset(item)

        raise ValueError("Dataset not found")

    def get_dataset(self, dataset_id: str) -> dict | None:
        return next((item for item in self.list_datasets() if item["id"] == dataset_id), None)

    def get_dataset_summary(self, dataset_id: str) -> dict:
        return self._dataset_summary(dataset_id)

    def dataset_dir(self, dataset_id: str) -> Path:
        return DATASETS_DIR / dataset_id

    def list_files(self, dataset_id: str, target: str) -> list[dict]:
        target_name = _normalize_target(target)
        folder = self.dataset_dir(dataset_id) / target_name
        folder.mkdir(parents=True, exist_ok=True)
        annotation_index = set()
        if target_name == "images":
            annotation_index = {item.stem for item in self._list_files(self.dataset_dir(dataset_id) / "labels") if item.suffix.lower() == ".json"}
        return [
            self._file_record(item, annotation_exists=item.stem in annotation_index if target_name == "images" else None)
            for item in self._list_files(folder)
        ]

    def save_file(self, dataset_id: str, target: str, file_name: str, content: bytes) -> dict:
        target_name = _normalize_target(target)
        folder = self.dataset_dir(dataset_id) / target_name
        folder.mkdir(parents=True, exist_ok=True)
        out_path = folder / Path(file_name).name
        out_path.write_bytes(content)
        return self._file_record(out_path)

    def save_files(self, dataset_id: str, target: str, files: list[tuple[str, bytes]]) -> list[dict]:
        saved: list[dict] = []
        for file_name, content in files:
            saved.append(self.save_file(dataset_id, target, file_name, content))
        return saved

    def save_annotation(self, dataset_id: str, image_name: str, labels: list[dict]) -> dict:
        folder = self.dataset_dir(dataset_id) / "labels"
        folder.mkdir(parents=True, exist_ok=True)
        out_path = folder / f"{Path(image_name).stem}.json"
        record = self._annotation_record(image_name, labels)
        out_path.write_text(json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8")
        return record

    def get_annotation(self, dataset_id: str, image_name: str) -> dict:
        out_path = self.dataset_dir(dataset_id) / "labels" / f"{Path(image_name).stem}.json"
        if not out_path.exists():
            return self._annotation_record(image_name, [])
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return self._annotation_record(image_name, payload)
        if isinstance(payload, dict):
            labels = payload.get("labels") or payload.get("annotations") or []
            if not isinstance(labels, list):
                labels = []
            record = self._annotation_record(image_name, labels)
            record.update({key: value for key, value in payload.items() if key not in {"labels", "annotations"}})
            record["labels"] = labels
            record["label_count"] = len(labels)
            return record
        return self._annotation_record(image_name, [])

    def delete_dataset(self, dataset_id: str) -> bool:
        payload = self.load()
        before = len(payload["datasets"])
        payload["datasets"] = [item for item in payload["datasets"] if item["id"] != dataset_id]
        if len(payload["datasets"]) == before:
            return False
        self.save(payload)
        shutil.rmtree(self.dataset_dir(dataset_id), ignore_errors=True)
        return True

    def _enrich_dataset(self, item: dict) -> dict:
        enriched = dict(item)
        enriched.update(self._dataset_summary(str(item["id"])))
        return enriched

    def _dataset_summary(self, dataset_id: str) -> dict:
        dataset_dir = self.dataset_dir(dataset_id)
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        exports_dir = dataset_dir / "exports"
        augmented_dir = dataset_dir / "augmented"

        image_files = self._list_files(images_dir)
        label_files = [item for item in self._list_files(labels_dir) if item.suffix.lower() == ".json"]
        export_files = self._list_files(exports_dir)
        augmented_files = self._list_files(augmented_dir, recursive=True)
        annotated_stems = {item.stem for item in label_files}
        annotated_image_count = sum(1 for item in image_files if item.stem in annotated_stems)
        image_count = len(image_files)
        annotation_coverage = round(annotated_image_count / image_count, 4) if image_count else 0.0

        return {
            "image_count": image_count,
            "label_count": len(label_files),
            "export_count": len(export_files),
            "augmented_count": len(augmented_files),
            "annotated_image_count": annotated_image_count,
            "annotation_coverage": annotation_coverage,
            "has_images": bool(image_files),
            "has_annotations": bool(label_files),
        }

    def _file_record(self, item: Path, *, annotation_exists: bool | None = None) -> dict:
        record = {"name": item.name, "path": str(item), "size": item.stat().st_size}
        if annotation_exists is not None:
            record["annotation_exists"] = annotation_exists
        return record

    def _list_files(self, folder: Path, *, recursive: bool = False) -> list[Path]:
        if not folder.exists():
            return []
        iterator = folder.rglob("*") if recursive else folder.iterdir()
        return [item for item in sorted(iterator) if item.is_file()]

    def _annotation_record(self, image_name: str, labels: list[dict]) -> dict:
        shape_counts: Counter[str] = Counter()
        for item in labels:
            if not isinstance(item, dict):
                continue
            shape = str(item.get("type") or item.get("shape_type") or item.get("kind") or "bbox").strip().lower() or "bbox"
            shape_counts[shape] += 1
        now = datetime.now(UTC).isoformat()
        return {
            "schema_version": 1,
            "image_name": image_name,
            "labels": labels,
            "label_count": len(labels),
            "shape_counts": dict(shape_counts),
            "updated_at": now,
        }
