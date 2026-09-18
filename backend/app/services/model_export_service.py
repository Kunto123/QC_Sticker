"""Model package import / export.

Import accepts any zip that contains exactly one detector model, at any depth:

* an Ultralytics OpenVINO export folder (`<name>_openvino_model/` with `<name>.xml`,
  `<name>.bin` and `metadata.yaml`) — the common case for this project;
* a single `.pt`, `.onnx` or `.tflite`, optionally with a `metadata.yaml`,
  `metadata.json` or `*.meta.json` next to it;
* a package produced by `create_export()` of this app (any version).

Every imported model lands in its own folder `data/models/<name>/` together with a
`<stem>.meta.json` carrying `class_names`, which is exactly where the inference
backends look for class names (`<model dir>/<model stem>.meta.json`). The registry
entry (`models.json`) points at the model file and the meta file.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import socket
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

from backend.app.core.config import MODELS_DIR
from backend.app.repositories.models_repository import ModelsRepository

MODEL_EXTENSIONS = {".xml": "openvino", ".pt": "ultralytics", ".onnx": "onnx", ".tflite": "tflite"}
_META_JSON_NAMES = ("metadata.json",)
_META_YAML_NAMES = ("metadata.yaml", "metadata.yml")
_IGNORED_PREFIXES = ("__MACOSX", ".")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dumps(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")


def _safe_component(value: str, fallback: str = "model") -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    normalized = normalized.strip("._-")
    return normalized or fallback


def runtime_for_path(path: str | Path) -> str:
    return MODEL_EXTENSIONS.get(Path(path).suffix.lower(), "ultralytics")


def class_names_from_yaml(raw: bytes | str) -> list[str]:
    """`names:` from an Ultralytics `metadata.yaml` — a {index: name} map or a list."""
    import yaml

    try:
        doc = yaml.safe_load(raw if isinstance(raw, str) else raw.decode("utf-8-sig"))
    except Exception:  # noqa: BLE001
        return []
    names = (doc or {}).get("names") if isinstance(doc, dict) else None
    if isinstance(names, dict):
        try:
            return [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
        except (TypeError, ValueError):
            return [str(v) for v in names.values()]
    if isinstance(names, list):
        return [str(v) for v in names]
    return []


def class_names_from_json(raw: bytes | str) -> list[str]:
    try:
        doc = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8-sig"))
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(doc, dict):
        return []
    names = doc.get("class_names")
    if names is None:
        names = doc.get("names")
    if isinstance(names, dict):
        try:
            return [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
        except (TypeError, ValueError):
            return [str(v) for v in names.values()]
    if isinstance(names, list):
        return [str(v) for v in names]
    return []


def write_meta_json(model_path: Path, class_names: list[str], **extra: Any) -> Path:
    """Write `<model dir>/<model stem>.meta.json` — the file the inference backends read."""
    meta_path = model_path.parent / f"{model_path.stem}.meta.json"
    payload = {"class_names": list(class_names), **extra}
    meta_path.write_bytes(_json_dumps(payload))
    return meta_path


def purge_model_files(model: dict[str, Any]) -> list[str]:
    """Delete a model's files from disk (only inside MODELS_DIR). A model that lives in
    its own sub-folder (everything imported by this service) has the folder removed."""
    removed: list[str] = []
    root = MODELS_DIR.resolve()
    path_value = str(model.get("path") or "").strip()
    if not path_value:
        return removed
    model_path = Path(path_value)
    if not model_path.is_absolute():
        model_path = MODELS_DIR / model_path
    try:
        model_path = model_path.resolve()
    except Exception:  # noqa: BLE001
        return removed
    if root not in model_path.parents:
        return removed
    folder = model_path.parent
    if folder != root and root in folder.parents:
        # Only report what is really gone: on Windows a model that is still loaded
        # (OpenVINO mmaps the .bin) cannot be deleted; the caller must unload first.
        shutil.rmtree(folder, ignore_errors=True)
        if not folder.exists():
            removed.append(str(folder))
        return removed
    for field in ("path", "meta_path"):
        raw = str(model.get(field) or "").strip()
        if not raw:
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = MODELS_DIR / candidate
        try:
            candidate = candidate.resolve()
        except Exception:  # noqa: BLE001
            continue
        if root in candidate.parents and candidate.is_file():
            try:
                candidate.unlink()
                removed.append(str(candidate))
            except OSError:
                pass
    if model_path.suffix.lower() == ".xml":
        bin_path = model_path.with_suffix(".bin")
        if bin_path.is_file():
            try:
                bin_path.unlink()
                removed.append(str(bin_path))
            except OSError:
                pass
    return sorted(set(removed))


class ModelExportService:
    def __init__(self, models_repo: ModelsRepository, templates_repo: Any, deployments_repo: Any) -> None:
        self.models_repo = models_repo
        self.templates_repo = templates_repo
        self.deployments_repo = deployments_repo

    # ── Export ──────────────────────────────────────────────────────────

    def _model_files(self, model: dict[str, Any]) -> list[Path]:
        """All files that make up the model: the weights, the `.bin` twin of an
        OpenVINO `.xml`, the paired meta json and (folder models) everything else in
        the folder such as `metadata.yaml`."""
        model_path = Path(str(model.get("path") or "").strip())
        if not model_path.exists() or not model_path.is_file():
            raise ValueError(f"Model file not found: {model_path}")
        files = {model_path.resolve()}
        meta_value = str(model.get("meta_path") or "").strip()
        if meta_value and Path(meta_value).is_file():
            files.add(Path(meta_value).resolve())
        if model_path.suffix.lower() == ".xml" and model_path.with_suffix(".bin").is_file():
            files.add(model_path.with_suffix(".bin").resolve())
        folder = model_path.parent.resolve()
        if folder != MODELS_DIR.resolve() and MODELS_DIR.resolve() in folder.parents:
            for extra in folder.iterdir():
                if extra.is_file():
                    files.add(extra.resolve())
        return sorted(files)

    def build_export_manifest(self, model_id: int) -> dict[str, Any]:
        model = self.models_repo.get_model(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found.")
        files = self._model_files(model)
        model_path = Path(str(model.get("path"))).resolve()
        return {
            "export_version": "2.0",
            "export_timestamp": datetime.now(UTC).isoformat(),
            "source_device": {
                "hostname": socket.gethostname(),
                "python_version": platform.python_version(),
            },
            "model": {
                "id": model.get("id"),
                "name": model.get("name"),
                "runtime": model.get("runtime"),
                "task": model.get("task"),
                "class_names": list(model.get("class_names") or []),
                "model_file": model_path.name,
            },
            "files": {
                path.name: {"size_bytes": path.stat().st_size, "checksum_sha256": _sha256_file(path)}
                for path in files
            },
            "deployment_references": self.models_repo.list_model_deployment_references(
                str(model_path), templates_repo=self.templates_repo, deployments_repo=self.deployments_repo,
            ),
            "lifecycle_status": model.get("lifecycle_status"),
        }

    def create_export(self, model_id: int) -> dict[str, Any]:
        model = self.models_repo.get_model(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found.")
        manifest = self.build_export_manifest(model_id)
        files = self._model_files(model)
        folder_name = _safe_component(str(model.get("name") or f"model-{model_id}"))
        stamp = str(manifest["export_timestamp"]).replace(":", "-").replace("+00:00", "Z")
        archive_name = f"model-export-{folder_name}-{stamp}.zip"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip", prefix="qc-suite-export-")
        tmp.close()
        archive_path = Path(tmp.name)
        try:
            with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as zf:
                for path in files:
                    zf.write(path, arcname=f"{folder_name}/{path.name}")
                zf.writestr(
                    f"{folder_name}/metadata.json",
                    _json_dumps({
                        "name": model.get("name"),
                        "runtime": model.get("runtime"),
                        "task": model.get("task"),
                        "class_names": list(model.get("class_names") or []),
                    }),
                )
                zf.writestr("EXPORT_MANIFEST.json", _json_dumps(manifest))
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise
        return {"archive_path": archive_path, "archive_name": archive_name, "manifest": manifest}

    # ── Import ──────────────────────────────────────────────────────────

    @staticmethod
    def _inspect_archive(zf: ZipFile) -> dict[str, Any]:
        """Find the one model inside the archive plus its companions."""
        members = [
            info for info in zf.infolist()
            if not info.is_dir()
            and not any(part.startswith(_IGNORED_PREFIXES) for part in PurePosixPath(info.filename).parts)
        ]
        by_ext: dict[str, list] = {}
        for info in members:
            ext = PurePosixPath(info.filename).suffix.lower()
            if ext in MODEL_EXTENSIONS:
                by_ext.setdefault(ext, []).append(info)
        # legacy app export: weights.pt is the model even if other .pt files exist
        legacy = [i for i in members if PurePosixPath(i.filename).name == "weights.pt"]
        if legacy and any(PurePosixPath(i.filename).name == "EXPORT_MANIFEST.json" for i in members):
            candidates = legacy
        else:
            candidates = [i for ext in (".xml", ".pt", ".onnx", ".tflite") for i in by_ext.get(ext, [])]
        if not candidates:
            raise ValueError(
                "Archive contains no model file (expected one of: .xml + .bin (OpenVINO), .pt, .onnx, .tflite)."
            )
        if len(candidates) > 1:
            names = ", ".join(PurePosixPath(i.filename).name for i in candidates)
            raise ValueError(f"Archive contains more than one model file ({names}); pack one model per zip.")
        model_info = candidates[0]
        model_dir = PurePosixPath(model_info.filename).parent
        model_stem = PurePosixPath(model_info.filename).stem
        runtime = MODEL_EXTENSIONS[PurePosixPath(model_info.filename).suffix.lower()]

        siblings = {PurePosixPath(i.filename).name: i for i in members if PurePosixPath(i.filename).parent == model_dir}
        bin_info = siblings.get(f"{model_stem}.bin") if runtime == "openvino" else None
        if runtime == "openvino" and bin_info is None:
            raise ValueError(f"OpenVINO model '{model_stem}.xml' has no matching '{model_stem}.bin' in the archive.")

        root_files = {PurePosixPath(i.filename).name: i for i in members if PurePosixPath(i.filename).parent == PurePosixPath(".")}
        meta_json = (
            siblings.get(f"{model_stem}.meta.json")
            or next((siblings[n] for n in _META_JSON_NAMES if n in siblings), None)
            or root_files.get(f"{model_stem}.meta.json")
            or next((root_files[n] for n in _META_JSON_NAMES if n in root_files), None)
        )
        meta_yaml = (
            next((siblings[n] for n in _META_YAML_NAMES if n in siblings), None)
            or next((root_files[n] for n in _META_YAML_NAMES if n in root_files), None)
        )
        manifest = root_files.get("EXPORT_MANIFEST.json") or siblings.get("EXPORT_MANIFEST.json")
        return {
            "model": model_info,
            "bin": bin_info,
            "meta_json": meta_json,
            "meta_yaml": meta_yaml,
            "manifest": manifest,
            "runtime": runtime,
            "stem": model_stem,
            "dir": model_dir,
            "extras": [
                i for name, i in siblings.items()
                if i not in (model_info, bin_info, meta_json, meta_yaml, manifest)
                and name not in _META_JSON_NAMES + _META_YAML_NAMES + ("EXPORT_MANIFEST.json", f"{model_stem}.meta.json")
                and PurePosixPath(name).suffix.lower() in {".yaml", ".yml", ".json", ".txt", ".md"}
            ],
        }

    def import_model_archive(
        self,
        archive_path: str | Path,
        *,
        name: str | None = None,
        original_filename: str | None = None,
        skip_validation: bool = False,
        force_rename: bool = False,
        target_lifecycle: str = "draft",
    ) -> dict[str, Any]:
        archive_path = Path(archive_path)
        if not archive_path.exists() or not archive_path.is_file():
            raise ValueError("Archive file not found.")
        try:
            zf = ZipFile(archive_path, "r")
        except BadZipFile as exc:
            raise ValueError("File is not a valid zip archive.") from exc

        warnings: list[str] = []
        with zf:
            found = self._inspect_archive(zf)
            model_bytes = zf.read(found["model"])
            bin_bytes = zf.read(found["bin"]) if found["bin"] is not None else None
            meta_json_bytes = zf.read(found["meta_json"]) if found["meta_json"] is not None else None
            meta_yaml_bytes = zf.read(found["meta_yaml"]) if found["meta_yaml"] is not None else None
            manifest = {}
            if found["manifest"] is not None:
                try:
                    manifest = json.loads(zf.read(found["manifest"]).decode("utf-8-sig"))
                except Exception:  # noqa: BLE001
                    warnings.append("EXPORT_MANIFEST.json could not be parsed; ignored.")
            extras = [(PurePosixPath(i.filename).name, zf.read(i)) for i in found["extras"]]

        # checksums from our own manifest (v1 or v2), when present
        if manifest and not skip_validation:
            files_info = dict(manifest.get("files") or {})
            model_member = PurePosixPath(found["model"].filename).name
            expected = str((files_info.get(model_member) or {}).get("checksum_sha256") or "")
            if expected and expected != _sha256_bytes(model_bytes):
                raise ValueError(f"Archive checksum mismatch for {model_member}")

        meta_doc: dict[str, Any] = {}
        if meta_json_bytes is not None:
            try:
                meta_doc = json.loads(meta_json_bytes.decode("utf-8-sig")) or {}
            except Exception:  # noqa: BLE001
                meta_doc = {}
        class_names = class_names_from_json(meta_json_bytes) if meta_json_bytes is not None else []
        if not class_names and meta_yaml_bytes is not None:
            class_names = class_names_from_yaml(meta_yaml_bytes)
        if not class_names and manifest:
            class_names = [str(c) for c in (manifest.get("model") or {}).get("class_names") or []]
        runtime = found["runtime"]
        if runtime != "ultralytics" and not class_names:
            warnings.append(
                "No class names found (metadata.yaml / *.meta.json). Detections will carry numeric "
                "labels until a <model>.meta.json with class_names is added."
            )

        display_name = (
            str(name or "").strip()
            or str((manifest.get("model") or {}).get("name") or meta_doc.get("name") or "").strip()
            or (str(found["dir"].name) if str(found["dir"]) not in ("", ".") else "")
            or Path(original_filename or archive_path.name).stem
            or found["stem"]
        )
        if self.models_repo.find_by_name(display_name) is not None or force_rename:
            display_name = f"{display_name} [IMPORTED {datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}]"

        folder_name = _safe_component(display_name)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dest_dir = MODELS_DIR / folder_name
        counter = 1
        while dest_dir.exists():
            dest_dir = MODELS_DIR / f"{folder_name}_{counter}"
            counter += 1
        dest_dir.mkdir(parents=True)

        stem = _safe_component(found["stem"], "model")
        model_dest = dest_dir / f"{stem}{PurePosixPath(found['model'].filename).suffix.lower()}"
        added: dict[str, Any] | None = None
        try:
            model_dest.write_bytes(model_bytes)
            if bin_bytes is not None:
                (dest_dir / f"{stem}.bin").write_bytes(bin_bytes)
            if meta_yaml_bytes is not None:
                (dest_dir / "metadata.yaml").write_bytes(meta_yaml_bytes)
            for extra_name, extra_bytes in extras:
                (dest_dir / _safe_component(extra_name, "extra")).write_bytes(extra_bytes)
            meta_dest = write_meta_json(
                model_dest,
                class_names,
                runtime=runtime,
                name=display_name,
                source_archive=str(original_filename or archive_path.name),
                imported_at=datetime.now(UTC).isoformat(),
            )
            added = self.models_repo.add_model(
                display_name,
                str(model_dest),
                "import",
                meta_path=str(meta_dest),
                runtime=runtime,
                task=str(meta_doc.get("task") or (manifest.get("model") or {}).get("task") or "detection"),
                class_names=class_names,
            )
            target = str(target_lifecycle or "draft").strip().lower() or "draft"
            if target != "draft":
                added = self.models_repo.transition_lifecycle(int(added["id"]), target)
        except Exception:
            if added is not None:
                try:
                    self.models_repo.delete_model(int(added["id"]))
                except Exception:  # noqa: BLE001
                    pass
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise

        return {
            "status": "success",
            "model_id": added["id"],
            "imported_name": added["name"],
            "runtime": runtime,
            "model_path": str(model_dest),
            "meta_path": str(added.get("meta_path")),
            "model_dir": str(dest_dir),
            "class_names": class_names,
            "lifecycle_status": added.get("lifecycle_status"),
            "warnings": warnings,
            "import_log": {
                "import_timestamp": datetime.now(UTC).isoformat(),
                "imported_on_device": socket.gethostname(),
                "source_archive": str(original_filename or archive_path.name),
            },
        }
