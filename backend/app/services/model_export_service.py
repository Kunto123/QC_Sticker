from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import socket
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from backend.app.core.config import MODELS_DIR
from backend.app.repositories.models_repository import ModelsRepository


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dumps(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")


def _safe_component(value: str, fallback: str = "model") -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    normalized = normalized.strip("._-")
    return normalized or fallback


def _package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:  # noqa: BLE001
            continue
    return None


class ModelExportService:
    def __init__(self, models_repo: ModelsRepository, templates_repo: Any, deployments_repo: Any) -> None:
        self.models_repo = models_repo
        self.templates_repo = templates_repo
        self.deployments_repo = deployments_repo

    def _load_metadata(self, meta_path: Path | None, model: dict[str, Any]) -> tuple[dict[str, Any], bytes, str]:
        if meta_path is not None and meta_path.exists() and meta_path.is_file():
            raw_bytes = meta_path.read_bytes()
            try:
                metadata = json.loads(raw_bytes.decode("utf-8-sig"))
                return metadata, raw_bytes, "paired-file"
            except Exception:  # noqa: BLE001
                metadata = {
                    "name": str(model.get("name") or ""),
                    "path": str(model.get("path") or ""),
                    "meta_path": str(meta_path),
                    "source": str(model.get("source") or "manual"),
                    "runtime": str(model.get("runtime") or "ultralytics"),
                    "task": str(model.get("task") or "detection"),
                    "class_names": list(model.get("class_names") or []),
                    "architecture_family": model.get("architecture_family"),
                    "architecture_variant": model.get("architecture_variant"),
                    "provenance": dict(model.get("provenance") or {}),
                    "training_params": {},
                }
                return metadata, _json_dumps(metadata), "synthetic"

        metadata = {
            "name": str(model.get("name") or ""),
            "path": str(model.get("path") or ""),
            "meta_path": str(model.get("meta_path") or ""),
            "source": str(model.get("source") or "manual"),
            "runtime": str(model.get("runtime") or "ultralytics"),
            "task": str(model.get("task") or "detection"),
            "class_names": list(model.get("class_names") or []),
            "architecture_family": model.get("architecture_family"),
            "architecture_variant": model.get("architecture_variant"),
            "provenance": dict(model.get("provenance") or {}),
            "training_params": {},
            "export_note": "Generated from registry metadata because no paired .meta.json file was available.",
        }
        raw_bytes = _json_dumps(metadata)
        return metadata, raw_bytes, "synthetic"

    def _prepare_export_context(self, model_id: int) -> dict[str, Any]:
        model = self.models_repo.get_model(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found.")

        weights_path = Path(str(model.get("path") or "").strip())
        if not weights_path.exists() or not weights_path.is_file():
            raise ValueError(f"Model weights file not found: {weights_path}")

        meta_path_value = str(model.get("meta_path") or "").strip()
        meta_path = Path(meta_path_value) if meta_path_value else None
        metadata, metadata_bytes, metadata_source = self._load_metadata(meta_path, model)
        references = self.models_repo.list_model_deployment_references(
            str(weights_path),
            templates_repo=self.templates_repo,
            deployments_repo=self.deployments_repo,
        )
        manifest = {
            "export_version": "1.0",
            "export_timestamp": datetime.now(UTC).isoformat(),
            "source_device": {
                "hostname": socket.gethostname(),
                "qc_suite_version": _package_version("qc-suite-python", "qc_suite_python"),
                "python_version": platform.python_version(),
                "ultralytics_version": _package_version("ultralytics"),
            },
            "model": {
                "id": model.get("id"),
                "name": model.get("name"),
                "source": model.get("source"),
                "runtime": model.get("runtime"),
                "task": model.get("task"),
                "architecture_family": model.get("architecture_family"),
                "architecture_variant": model.get("architecture_variant"),
                "class_names": list(model.get("class_names") or []),
                "checksum_sha256": model.get("checksum_sha256"),
            },
            "provenance": dict(model.get("provenance") or {}),
            "training_params": dict(metadata.get("training_params") or metadata.get("params") or {}),
            "files": {
                "weights.pt": {
                    "original_path": str(weights_path),
                    "size_bytes": weights_path.stat().st_size,
                    "checksum_sha256": _sha256_file(weights_path),
                },
                "metadata.json": {
                    "original_path": str(meta_path) if meta_path is not None else None,
                    "size_bytes": len(metadata_bytes),
                    "checksum_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                    "source": metadata_source,
                },
            },
            "deployment_references": references,
            "lifecycle_status": model.get("lifecycle_status"),
            "lifecycle_notes": list(model.get("lifecycle_notes") or []),
        }
        return {
            "model": model,
            "weights_path": weights_path,
            "meta_path": meta_path,
            "metadata": metadata,
            "metadata_bytes": metadata_bytes,
            "metadata_source": metadata_source,
            "references": references,
            "manifest": manifest,
        }

    def build_export_manifest(self, model_id: int) -> dict[str, Any]:
        return dict(self._prepare_export_context(model_id)["manifest"])

    def _archive_name(self, model: dict[str, Any], manifest: dict[str, Any]) -> str:
        export_stamp = str(manifest.get("export_timestamp") or datetime.now(UTC).isoformat())
        timestamp_tag = export_stamp.replace(":", "-").replace("+00:00", "Z")
        model_name = _safe_component(str(model.get("name") or f"model-{model.get('id') or 'export'}"))
        return f"model-export-{model_name}-{timestamp_tag}.zip"

    def _deployment_summary(self, references: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "deployment_references": references,
            "active_deployments": [reference for reference in references if reference.get("is_active")],
        }

    def _readme_text(self, model: dict[str, Any], manifest: dict[str, Any], archive_name: str) -> str:
        file_info = dict(manifest.get("files") or {})
        weights_info = dict(file_info.get("weights.pt") or {})
        metadata_info = dict(file_info.get("metadata.json") or {})
        return "\n".join(
            [
                f"Model Export Package: {model.get('name') or 'Unnamed model'}",
                f"Archive: {archive_name}",
                f"Export timestamp: {manifest.get('export_timestamp')}",
                f"Source device: {dict(manifest.get('source_device') or {}).get('hostname') or 'unknown'}",
                "",
                "Contents:",
                "  - weights.pt",
                "  - metadata.json",
                "  - EXPORT_MANIFEST.json",
                "  - deployment_references.json",
                "",
                "Checksums:",
                f"  weights.pt: {weights_info.get('checksum_sha256') or 'unknown'}",
                f"  metadata.json: {metadata_info.get('checksum_sha256') or 'unknown'}",
                "",
                "Import notes:",
                "  - The import flow remaps the model to the current device data directory.",
                "  - Template references are informational; they are not rewritten automatically.",
            ]
        )

    def create_export(self, model_id: int, *, include_training_history: bool = False) -> dict[str, Any]:
        context = self._prepare_export_context(model_id)
        manifest = dict(context["manifest"])
        archive_name = self._archive_name(context["model"], manifest)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".zip", prefix="qc-suite-export-")
        temp_file.close()
        archive_path = Path(temp_file.name)

        deployment_payload = self._deployment_summary(context["references"])
        if include_training_history:
            deployment_payload["include_training_history"] = True

        try:
            with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as zip_file:
                zip_file.write(context["weights_path"], arcname="weights.pt")
                zip_file.writestr("metadata.json", context["metadata_bytes"])
                zip_file.writestr("EXPORT_MANIFEST.json", _json_dumps(manifest))
                zip_file.writestr("deployment_references.json", _json_dumps(deployment_payload))
                zip_file.writestr("README.txt", self._readme_text(context["model"], manifest, archive_name))
        except Exception:
            try:
                archive_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
            raise

        return {
            "archive_path": archive_path,
            "archive_name": archive_name,
            "manifest": manifest,
            "deployment_references": context["references"],
        }

    def _load_archive_bytes(self, archive_path: Path, member_name: str) -> bytes:
        with ZipFile(archive_path, "r") as zip_file:
            try:
                with zip_file.open(member_name) as file_handle:
                    return file_handle.read()
            except KeyError as exc:
                raise ValueError(f"Missing required archive member: {member_name}") from exc

    def import_model_archive(
        self,
        archive_path: str | Path,
        *,
        original_filename: str | None = None,
        skip_validation: bool = False,
        force_rename: bool = False,
        target_lifecycle: str = "draft",
    ) -> dict[str, Any]:
        archive_path = Path(archive_path)
        if not archive_path.exists() or not archive_path.is_file():
            raise ValueError("Archive file not found.")

        with ZipFile(archive_path, "r") as zip_file:
            members = set(zip_file.namelist())
            required_members = {"weights.pt", "metadata.json", "EXPORT_MANIFEST.json"}
            missing = sorted(required_members - members)
            if missing:
                raise ValueError(f"Archive is missing required file(s): {', '.join(missing)}")

        manifest_bytes = self._load_archive_bytes(archive_path, "EXPORT_MANIFEST.json")
        metadata_bytes = self._load_archive_bytes(archive_path, "metadata.json")
        weights_bytes = self._load_archive_bytes(archive_path, "weights.pt")
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
        metadata = json.loads(metadata_bytes.decode("utf-8-sig"))

        weights_expected = str(((manifest.get("files") or {}).get("weights.pt") or {}).get("checksum_sha256") or "")
        metadata_expected = str(((manifest.get("files") or {}).get("metadata.json") or {}).get("checksum_sha256") or "")
        weights_checksum = hashlib.sha256(weights_bytes).hexdigest()
        metadata_checksum = hashlib.sha256(metadata_bytes).hexdigest()
        if not skip_validation:
            if weights_expected and weights_checksum != weights_expected:
                raise ValueError("Archive checksum mismatch for weights.pt")
            if metadata_expected and metadata_checksum != metadata_expected:
                raise ValueError("Archive checksum mismatch for metadata.json")

        source_provenance = dict(manifest.get("provenance") or {})
        source_model = dict(manifest.get("model") or {})
        source_device = dict(manifest.get("source_device") or {})
        class_names = list(metadata.get("class_names") or source_model.get("class_names") or [])
        architecture_family = str(metadata.get("architecture_family") or source_model.get("architecture_family") or "").strip() or None
        architecture_variant = str(metadata.get("architecture_variant") or source_model.get("architecture_variant") or "").strip() or None
        runtime = str(metadata.get("runtime") or source_model.get("runtime") or "ultralytics").strip() or "ultralytics"
        task = str(metadata.get("task") or source_model.get("task") or "detection").strip() or "detection"
        imported_name = str(source_model.get("name") or metadata.get("name") or Path(original_filename or archive_path.name).stem or "Imported Model").strip() or "Imported Model"
        if self.models_repo.find_by_name(imported_name) is not None or force_rename:
            imported_name = f"{imported_name} [IMPORTED {datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}]"

        dataset_token = str(
            metadata.get("dataset_id")
            or metadata.get("job_id")
            or source_provenance.get("source_dataset_id")
            or "imported"
        ).strip()
        architecture_token = architecture_variant or architecture_family or "model"
        timestamp_tag = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        trained_dir = MODELS_DIR / "trained"
        trained_dir.mkdir(parents=True, exist_ok=True)

        weights_name = f"{_safe_component(dataset_token)}__{_safe_component(architecture_token)}__{timestamp_tag}__imported.pt"
        meta_name = weights_name.replace(".pt", ".meta.json")
        weights_dest = trained_dir / weights_name
        meta_dest = trained_dir / meta_name
        counter = 1
        while weights_dest.exists() or meta_dest.exists():
            suffix = f"_{counter}"
            weights_dest = trained_dir / f"{weights_name[:-3]}{suffix}.pt"
            meta_dest = trained_dir / f"{meta_name[:-10]}{suffix}.meta.json"
            counter += 1

        weights_dest.write_bytes(weights_bytes)
        meta_dest.write_bytes(metadata_bytes)

        added_model: dict[str, Any] | None = None
        try:
            added_model = self.models_repo.add_model(
                imported_name,
                str(weights_dest),
                "import",
                meta_path=str(meta_dest),
                runtime=runtime,
                task=task,
                class_names=class_names,
                architecture_family=architecture_family,
                architecture_variant=architecture_variant,
                source_dataset_id=str(source_provenance.get("source_dataset_id") or metadata.get("dataset_id") or metadata.get("job_id") or "").strip() or None,
                training_job_id=str(source_provenance.get("training_job_id") or metadata.get("training_job_id") or "").strip() or None,
            )
            normalized_target = str(target_lifecycle or "draft").strip().lower() or "draft"
            if normalized_target != "draft":
                added_model = self.models_repo.transition_lifecycle(int(added_model["id"]), normalized_target)
        except Exception:
            if added_model is not None:
                try:
                    self.models_repo.delete_model(int(added_model["id"]))
                except Exception:  # noqa: BLE001
                    pass
            for candidate in (weights_dest, meta_dest):
                try:
                    candidate.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass
            raise

        source_class_names = list(source_model.get("class_names") or [])
        warnings: list[str] = []
        if source_class_names and class_names and source_class_names != class_names:
            warnings.append("Class names differ between the exported model manifest and metadata.json.")

        return {
            "status": "success",
            "model_id": added_model["id"],
            "imported_name": added_model["name"],
            "model_path": str(weights_dest),
            "meta_path": str(meta_dest),
            "source_export": {
                "original_name": source_model.get("name"),
                "original_device": source_device.get("hostname"),
                "original_architecture": f"{source_model.get('architecture_family') or ''}{source_model.get('architecture_variant') or ''}".strip(),
                "class_names": class_names,
            },
            "source_deployments": [
                {
                    "line_id": reference.get("line_id"),
                    "station_id": reference.get("station_id"),
                    "template_name": reference.get("template_name"),
                    "template_version_id": reference.get("template_version_id"),
                    "deployment_id": reference.get("deployment_id"),
                    "is_active": reference.get("is_active"),
                    "note": "You may need to update template vision.model_path",
                }
                for reference in list(manifest.get("deployment_references") or [])
            ],
            "file_checksums": {
                "weights": weights_checksum,
                "metadata": metadata_checksum,
            },
            "warnings": warnings,
            "import_log": {
                "import_timestamp": datetime.now(UTC).isoformat(),
                "imported_on_device": socket.gethostname(),
                "source_export_timestamp": manifest.get("export_timestamp"),
            },
            "lifecycle_status": added_model.get("lifecycle_status"),
            "next_steps": [
                "Review class_names match your dataset labeling",
                "Test model with the inspection flow",
                "If satisfied, promote it to validated or production",
                "Update templates if replacing an existing model",
                "Deploy to stations as needed",
            ],
        }