from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from backend.app.repositories.models_repository import ModelsRepository
from backend.app.services import model_export_service as export_module
from backend.app.services.model_export_service import ModelExportService


class StubTemplatesRepo:
    def __init__(self, version_record: dict) -> None:
        self.version_record = dict(version_record)

    def get_version(self, version_id: int) -> dict | None:
        if int(version_id) != int(self.version_record.get("version_id") or 0):
            return None
        return dict(self.version_record)


class StubDeploymentsRepo:
    def __init__(self, deployments: list[dict]) -> None:
        self.deployments = [dict(item) for item in deployments]

    def list_deployments(self) -> list[dict]:
        return [dict(item) for item in self.deployments]


class StubModelsRepo:
    def __init__(self, model: dict | None = None, references: list[dict] | None = None, existing_names: list[str] | None = None) -> None:
        self.model = dict(model) if model is not None else None
        self.references = [dict(item) for item in (references or [])]
        self.existing_names = {str(name).strip().lower() for name in (existing_names or [])}
        self.add_calls: list[dict] = []
        self.transition_calls: list[tuple[int, str]] = []
        self.delete_calls: list[int] = []

    def get_model(self, model_id: int) -> dict | None:
        if self.model is None:
            return None
        if int(self.model.get("id") or 0) != int(model_id):
            return None
        return dict(self.model)

    def list_model_deployment_references(self, model_path: str, *, templates_repo, deployments_repo) -> list[dict]:  # noqa: ANN001
        return [dict(item) for item in self.references if str(item.get("model_path") or "") == str(model_path)]

    def find_by_name(self, name: str) -> dict | None:
        if str(name or "").strip().lower() in self.existing_names:
            return {"name": name}
        return None

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
        record = {
            "id": len(self.add_calls) + 100,
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
            "checksum_sha256": None,
            "provenance": {
                "source_dataset_id": source_dataset_id,
                "training_job_id": training_job_id,
            },
            "created_at": "2026-04-16T00:00:00+00:00",
        }
        self.add_calls.append(dict(record))
        self.model = dict(record)
        return dict(record)

    def transition_lifecycle(self, model_id: int, new_status: str, *, actor_id: int | None = None, note: str | None = None) -> dict:  # noqa: ARG002
        self.transition_calls.append((int(model_id), str(new_status)))
        if self.model is None:
            raise ValueError("Model not found.")
        updated = dict(self.model)
        updated["lifecycle_status"] = new_status
        updated["updated_at"] = "2026-04-16T00:00:00+00:00"
        if note:
            updated.setdefault("lifecycle_notes", []).append({"status": new_status, "note": note, "at": updated["updated_at"], "by": actor_id})
        self.model = updated
        return dict(updated)

    def delete_model(self, model_id: int) -> dict:  # noqa: ARG002
        self.delete_calls.append(int(model_id))
        if self.model is None:
            raise ValueError("Model not found.")
        return dict(self.model)


class ModelExportImportTest(unittest.TestCase):
    def test_repository_helpers_find_active_deployment_references(self) -> None:
        repo = ModelsRepository()
        model_path = r"D:\ProjectMagang\qc-suite-python\data\models\trained\demo-export.pt"
        version_record = {
            "version_id": 1,
            "template": {
                "name": "QC Line A",
                "vision": {
                    "model_path": model_path,
                },
            },
        }
        deployments = [
            {
                "id": 11,
                "line_id": "LINE-A",
                "station_id": "STATION-01",
                "template_id": 5,
                "template_version_id": 1,
                "template_name": "QC Line A",
                "version_number": 1,
                "is_active": True,
                "effective_from": "2026-04-16T00:00:00+00:00",
                "effective_until": None,
            },
            {
                "id": 12,
                "line_id": "LINE-B",
                "station_id": "STATION-02",
                "template_id": 6,
                "template_version_id": 2,
                "template_name": "Other Template",
                "version_number": 1,
                "is_active": False,
                "effective_from": "2026-04-15T00:00:00+00:00",
                "effective_until": "2026-04-16T00:00:00+00:00",
            },
        ]
        templates_repo = StubTemplatesRepo(version_record)
        deployments_repo = StubDeploymentsRepo(deployments)

        conflict = repo.find_active_model_conflict(model_path, templates_repo=templates_repo, deployments_repo=deployments_repo)
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict["deployment_id"], 11)
        self.assertEqual(conflict["template_model_path"], model_path)

        references = repo.list_model_deployment_references(model_path, templates_repo=templates_repo, deployments_repo=deployments_repo)
        self.assertEqual(len(references), 1)
        self.assertEqual(references[0]["deployment_id"], 11)
        self.assertTrue(references[0]["is_active"])

    def test_export_and_import_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            models_dir = temp_root / "models"
            weights_dir = models_dir / "trained"
            weights_dir.mkdir(parents=True, exist_ok=True)

            weights_path = weights_dir / "sticker-export.pt"
            metadata_path = weights_dir / "sticker-export.meta.json"
            weights_bytes = b"mock-weights-content"
            metadata_payload = {
                "dataset_id": "ds-001",
                "job_id": "job-001",
                "architecture_family": "yolov11",
                "architecture_variant": "n",
                "runtime": "ultralytics",
                "task": "detection",
                "class_names": ["A", "B"],
                "training_params": {
                    "epochs": 1,
                    "imgsz": 640,
                },
            }
            weights_path.write_bytes(weights_bytes)
            metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            model_record = {
                "id": 7,
                "name": "Sticker Model",
                "path": str(weights_path),
                "meta_path": str(metadata_path),
                "source": "training",
                "runtime": "ultralytics",
                "task": "detection",
                "class_names": ["A", "B"],
                "architecture_family": "yolov11",
                "architecture_variant": "n",
                "lifecycle_status": "draft",
                "checksum_sha256": hashlib.sha256(weights_bytes).hexdigest(),
                "provenance": {
                    "source_dataset_id": "ds-001",
                    "training_job_id": "job-001",
                },
                "lifecycle_notes": [{"status": "draft", "note": "seeded"}],
            }
            references = [
                {
                    "deployment_id": 11,
                    "line_id": "LINE-A",
                    "station_id": "STATION-01",
                    "template_id": 5,
                    "template_version_id": 1,
                    "template_name": "QC Line A",
                    "version_number": 1,
                    "is_active": True,
                    "effective_from": "2026-04-16T00:00:00+00:00",
                    "effective_until": None,
                    "model_path": str(weights_path),
                }
            ]
            templates_repo = StubTemplatesRepo(
                {
                    "version_id": 1,
                    "template": {
                        "name": "QC Line A",
                        "vision": {
                            "model_path": str(weights_path),
                        },
                    },
                }
            )
            deployments_repo = StubDeploymentsRepo(
                [
                    {
                        "id": 11,
                        "line_id": "LINE-A",
                        "station_id": "STATION-01",
                        "template_id": 5,
                        "template_version_id": 1,
                        "template_name": "QC Line A",
                        "version_number": 1,
                        "is_active": True,
                        "effective_from": "2026-04-16T00:00:00+00:00",
                        "effective_until": None,
                    }
                ]
            )

            export_repo = StubModelsRepo(model_record, references=references)
            import_repo = StubModelsRepo(existing_names=[])

            with patch.object(export_module, "MODELS_DIR", models_dir):
                export_service = ModelExportService(export_repo, templates_repo, deployments_repo)
                bundle = export_service.create_export(7)
                archive_path = Path(bundle["archive_path"])
                self.assertTrue(archive_path.exists())

                with ZipFile(archive_path, "r") as zip_file:
                    self.assertEqual({"weights.pt", "metadata.json", "EXPORT_MANIFEST.json", "deployment_references.json", "README.txt"}, set(zip_file.namelist()))
                    self.assertEqual(zip_file.read("metadata.json"), metadata_path.read_bytes())
                    manifest = json.loads(zip_file.read("EXPORT_MANIFEST.json").decode("utf-8"))
                    self.assertEqual(manifest["model"]["name"], "Sticker Model")
                    self.assertEqual(manifest["files"]["weights.pt"]["checksum_sha256"], hashlib.sha256(weights_bytes).hexdigest())
                    deployment_payload = json.loads(zip_file.read("deployment_references.json").decode("utf-8"))
                    self.assertEqual(len(deployment_payload["deployment_references"]), 1)
                    self.assertEqual(deployment_payload["deployment_references"][0]["deployment_id"], 11)

                import_service = ModelExportService(import_repo, templates_repo, deployments_repo)
                imported = import_service.import_model_archive(
                    archive_path,
                    target_lifecycle="validated",
                )

                self.assertEqual(imported["status"], "success")
                self.assertEqual(imported["lifecycle_status"], "validated")
                self.assertEqual(import_repo.add_calls[0]["name"], "Sticker Model")
                self.assertEqual(import_repo.transition_calls[0][1], "validated")
                self.assertTrue(Path(imported["model_path"]).exists())
                self.assertTrue(Path(imported["meta_path"]).exists())
                self.assertEqual(imported["file_checksums"]["weights"], hashlib.sha256(weights_bytes).hexdigest())
                self.assertEqual(imported["source_export"]["class_names"], ["A", "B"])

                archive_path.unlink(missing_ok=True)
                Path(imported["model_path"]).unlink(missing_ok=True)
                Path(imported["meta_path"]).unlink(missing_ok=True)
