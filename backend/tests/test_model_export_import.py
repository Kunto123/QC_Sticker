"""Model package import / export (backend/app/services/model_export_service.py).

Training happens in other software, so importing a finished model is the only way
new models enter this app. The important case is an Ultralytics OpenVINO export
folder zipped as-is:

    best_openvino_model/
        best.xml
        best.bin
        metadata.yaml      # names: {0: sticker, ...}

It must land in `data/models/<name>/` with a `<stem>.meta.json` carrying the
class names, because that is the only file the non-.pt inference backends read
class names from.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from backend.app.repositories import base_json
from backend.app.repositories.models_repository import ModelsRepository
from backend.app.services import model_export_service as export_module
from backend.app.services.model_export_service import (
    ModelExportService,
    class_names_from_yaml,
    purge_model_files,
)


class StubTemplatesRepo:
    def __init__(self, version_record: dict | None = None) -> None:
        self.version_record = dict(version_record or {})

    def get_version(self, version_id: int) -> dict | None:
        if int(version_id) != int(self.version_record.get("version_id") or 0):
            return None
        return dict(self.version_record)


class StubDeploymentsRepo:
    def __init__(self, deployments: list[dict] | None = None) -> None:
        self.deployments = [dict(item) for item in (deployments or [])]

    def list_deployments(self) -> list[dict]:
        return [dict(item) for item in self.deployments]


def _write_zip(path: Path, members: dict[str, bytes]) -> Path:
    with ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


OPENVINO_YAML = b"""description: Ultralytics YOLOv8n model trained on sticker
author: Ultralytics
date: '2026-09-18'
version: 8.3.0
license: AGPL-3.0
stride: 32
task: detect
batch: 1
imgsz:
- 640
- 640
names:
  0: sticker
  1: sticker_bad
"""


def _openvino_zip(path: Path, *, folder: str = "best_openvino_model", stem: str = "best", with_yaml: bool = True) -> Path:
    members = {
        f"{folder}/{stem}.xml": b"<net name='best'/>",
        f"{folder}/{stem}.bin": b"\x00\x01\x02\x03",
    }
    if with_yaml:
        members[f"{folder}/metadata.yaml"] = OPENVINO_YAML
    return _write_zip(path, members)


class ImportTestBase(unittest.TestCase):
    """Every test gets a throwaway data root; MODELS_DIR and the registry file are patched."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.models_dir = self.root / "models"
        self.models_dir.mkdir()
        self.store_dir = self.root / "json_store"
        self.store_dir.mkdir()
        self._patches = [
            patch.object(export_module, "MODELS_DIR", self.models_dir),
            patch.object(base_json, "JSON_STORE_DIR", self.store_dir),
        ]
        for p in self._patches:
            p.start()
        self.models_repo = ModelsRepository()
        self.service = ModelExportService(self.models_repo, StubTemplatesRepo(), StubDeploymentsRepo())

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()


class OpenVinoZipImportTest(ImportTestBase):
    def test_openvino_export_folder_lands_in_models_dir_with_class_names(self) -> None:
        archive = _openvino_zip(self.root / "best_openvino_model.zip")

        result = self.service.import_model_archive(archive, original_filename="best_openvino_model.zip")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["runtime"], "openvino")
        self.assertEqual(result["class_names"], ["sticker", "sticker_bad"])
        self.assertEqual(result["warnings"], [])
        # name defaults to the folder inside the zip
        self.assertEqual(result["imported_name"], "best_openvino_model")

        model_dir = Path(result["model_dir"])
        self.assertEqual(model_dir.parent, self.models_dir)
        self.assertEqual(model_dir.name, "best_openvino_model")
        self.assertEqual(
            sorted(p.name for p in model_dir.iterdir()),
            ["best.bin", "best.meta.json", "best.xml", "metadata.yaml"],
        )
        self.assertEqual((model_dir / "best.bin").read_bytes(), b"\x00\x01\x02\x03")
        # the .meta.json the OpenVINO backend reads
        meta = json.loads((model_dir / "best.meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["class_names"], ["sticker", "sticker_bad"])
        self.assertEqual(meta["runtime"], "openvino")

        # registry entry points at the files
        record = self.models_repo.get_model(int(result["model_id"]))
        self.assertEqual(record["runtime"], "openvino")
        self.assertEqual(record["source"], "import")
        self.assertEqual(Path(record["path"]), model_dir / "best.xml")
        self.assertEqual(Path(record["meta_path"]), model_dir / "best.meta.json")
        self.assertEqual(record["class_names"], ["sticker", "sticker_bad"])
        self.assertEqual(record["lifecycle_status"], "draft")

    def test_explicit_name_wins_over_zip_folder_name(self) -> None:
        archive = _openvino_zip(self.root / "x.zip")
        result = self.service.import_model_archive(archive, name="Sticker Line A v3")
        self.assertEqual(result["imported_name"], "Sticker Line A v3")
        self.assertEqual(Path(result["model_dir"]).name, "Sticker_Line_A_v3")
        self.assertTrue((Path(result["model_dir"]) / "best.xml").is_file())

    def test_zip_with_files_at_root_uses_archive_filename(self) -> None:
        archive = _write_zip(self.root / "line-b-model.zip", {
            "best.xml": b"<net/>",
            "best.bin": b"\x00",
            "metadata.yaml": OPENVINO_YAML,
        })
        result = self.service.import_model_archive(archive, original_filename="line-b-model.zip")
        self.assertEqual(result["imported_name"], "line-b-model")
        self.assertEqual(result["class_names"], ["sticker", "sticker_bad"])

    def test_macosx_resource_fork_entries_are_ignored(self) -> None:
        archive = _write_zip(self.root / "mac.zip", {
            "best_openvino_model/best.xml": b"<net/>",
            "best_openvino_model/best.bin": b"\x00",
            "best_openvino_model/metadata.yaml": OPENVINO_YAML,
            "__MACOSX/best_openvino_model/._best.xml": b"junk",
            "__MACOSX/best_openvino_model/._best.bin": b"junk",
        })
        result = self.service.import_model_archive(archive)
        self.assertEqual(result["runtime"], "openvino")
        self.assertEqual(sorted(p.name for p in Path(result["model_dir"]).iterdir()),
                         ["best.bin", "best.meta.json", "best.xml", "metadata.yaml"])

    def test_missing_bin_is_rejected_and_leaves_no_folder(self) -> None:
        archive = _write_zip(self.root / "nobin.zip", {
            "m/best.xml": b"<net/>",
            "m/metadata.yaml": OPENVINO_YAML,
        })
        with self.assertRaisesRegex(ValueError, "no matching 'best.bin'"):
            self.service.import_model_archive(archive)
        self.assertEqual(list(self.models_dir.iterdir()), [])
        self.assertEqual(self.models_repo.list_models(), [])

    def test_missing_metadata_yaml_imports_with_warning(self) -> None:
        archive = _openvino_zip(self.root / "noyaml.zip", with_yaml=False)
        result = self.service.import_model_archive(archive)
        self.assertEqual(result["class_names"], [])
        self.assertEqual(len(result["warnings"]), 1)
        self.assertIn("No class names", result["warnings"][0])
        meta = json.loads((Path(result["model_dir"]) / "best.meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["class_names"], [])

    def test_two_models_in_one_zip_is_rejected(self) -> None:
        archive = _write_zip(self.root / "two.zip", {
            "a/best.xml": b"<net/>", "a/best.bin": b"\x00",
            "b/best.pt": b"pt",
        })
        with self.assertRaisesRegex(ValueError, "more than one model file"):
            self.service.import_model_archive(archive)

    def test_zip_without_model_is_rejected(self) -> None:
        archive = _write_zip(self.root / "empty.zip", {"readme.txt": b"hi"})
        with self.assertRaisesRegex(ValueError, "no model file"):
            self.service.import_model_archive(archive)

    def test_not_a_zip_is_rejected(self) -> None:
        bogus = self.root / "bogus.zip"
        bogus.write_bytes(b"not a zip")
        with self.assertRaisesRegex(ValueError, "not a valid zip"):
            self.service.import_model_archive(bogus)

    def test_duplicate_name_gets_imported_suffix_and_separate_folder(self) -> None:
        archive = _openvino_zip(self.root / "dup.zip")
        first = self.service.import_model_archive(archive)
        second = self.service.import_model_archive(archive)
        self.assertEqual(first["imported_name"], "best_openvino_model")
        self.assertTrue(second["imported_name"].startswith("best_openvino_model [IMPORTED "))
        self.assertNotEqual(first["model_dir"], second["model_dir"])
        self.assertTrue((Path(second["model_dir"]) / "best.xml").is_file())
        self.assertEqual(len(self.models_repo.list_models()), 2)

    def test_target_lifecycle_is_applied(self) -> None:
        archive = _openvino_zip(self.root / "lc.zip")
        result = self.service.import_model_archive(archive, target_lifecycle="validated")
        self.assertEqual(result["lifecycle_status"], "validated")

    def test_invalid_target_lifecycle_rolls_back_files_and_registry(self) -> None:
        archive = _openvino_zip(self.root / "bad-lc.zip")
        with self.assertRaises(ValueError):
            self.service.import_model_archive(archive, target_lifecycle="published")
        self.assertEqual(list(self.models_dir.iterdir()), [])
        self.assertEqual(self.models_repo.list_models(), [])


class OtherFormatsImportTest(ImportTestBase):
    def test_single_pt_in_zip(self) -> None:
        archive = _write_zip(self.root / "yolo.zip", {"best.pt": b"pt-bytes"})
        result = self.service.import_model_archive(archive, original_filename="yolo.zip")
        self.assertEqual(result["runtime"], "ultralytics")
        self.assertEqual(result["imported_name"], "yolo")
        # .pt carries its own names; no warning even without metadata
        self.assertEqual(result["warnings"], [])
        self.assertTrue((Path(result["model_dir"]) / "best.pt").is_file())

    def test_onnx_with_meta_json_sidecar(self) -> None:
        archive = _write_zip(self.root / "onnx.zip", {
            "best.onnx": b"onnx",
            "best.meta.json": json.dumps({"class_names": ["sticker"]}).encode(),
        })
        result = self.service.import_model_archive(archive)
        self.assertEqual(result["runtime"], "onnx")
        self.assertEqual(result["class_names"], ["sticker"])

    def test_tflite_with_metadata_json_names_map(self) -> None:
        archive = _write_zip(self.root / "tflite.zip", {
            "m/best.tflite": b"tfl",
            "m/metadata.json": json.dumps({"names": {"1": "b", "0": "a"}}).encode(),
        })
        result = self.service.import_model_archive(archive)
        self.assertEqual(result["runtime"], "tflite")
        self.assertEqual(result["class_names"], ["a", "b"])

    def test_legacy_v1_export_still_imports(self) -> None:
        """Archives made by the pre-2026-09-18 exporter: weights.pt + metadata.json + EXPORT_MANIFEST.json at root."""
        weights = b"legacy-weights"
        metadata = json.dumps({"class_names": ["A", "B"], "runtime": "ultralytics", "task": "detection"}).encode()
        manifest = {
            "export_version": "1.0",
            "model": {"name": "Legacy Sticker Model", "runtime": "ultralytics", "class_names": ["A", "B"]},
            "files": {
                "weights.pt": {"checksum_sha256": hashlib.sha256(weights).hexdigest()},
                "metadata.json": {"checksum_sha256": hashlib.sha256(metadata).hexdigest()},
            },
        }
        archive = _write_zip(self.root / "legacy.zip", {
            "weights.pt": weights,
            "metadata.json": metadata,
            "EXPORT_MANIFEST.json": json.dumps(manifest).encode(),
        })
        result = self.service.import_model_archive(archive)
        self.assertEqual(result["imported_name"], "Legacy Sticker Model")
        self.assertEqual(result["class_names"], ["A", "B"])
        self.assertTrue((Path(result["model_dir"]) / "weights.pt").is_file())

    def test_manifest_checksum_mismatch_is_rejected_unless_skipped(self) -> None:
        manifest = {"files": {"weights.pt": {"checksum_sha256": "0" * 64}}, "model": {"name": "X"}}
        archive = _write_zip(self.root / "tampered.zip", {
            "weights.pt": b"changed",
            "EXPORT_MANIFEST.json": json.dumps(manifest).encode(),
        })
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.service.import_model_archive(archive)
        self.assertEqual(list(self.models_dir.iterdir()), [])
        result = self.service.import_model_archive(archive, skip_validation=True)
        self.assertEqual(result["imported_name"], "X")


class ExportRoundTripTest(ImportTestBase):
    def test_export_then_import_reproduces_openvino_folder(self) -> None:
        imported = self.service.import_model_archive(_openvino_zip(self.root / "src.zip"), name="Line A")
        model_id = int(imported["model_id"])

        manifest = self.service.build_export_manifest(model_id)
        self.assertEqual(manifest["export_version"], "2.0")
        self.assertEqual(manifest["model"]["model_file"], "best.xml")
        self.assertEqual(sorted(manifest["files"]), ["best.bin", "best.meta.json", "best.xml", "metadata.yaml"])

        bundle = self.service.create_export(model_id)
        try:
            with ZipFile(bundle["archive_path"]) as zf:
                names = sorted(zf.namelist())
            self.assertEqual(names, [
                "EXPORT_MANIFEST.json",
                "Line_A/best.bin", "Line_A/best.meta.json", "Line_A/best.xml",
                "Line_A/metadata.json", "Line_A/metadata.yaml",
            ])
            # re-import on a "second PC": same class names, own folder, name kept (no clash: different registry)
            other_root = self.root / "pc2"
            (other_root / "models").mkdir(parents=True)
            (other_root / "json_store").mkdir()
            with patch.object(export_module, "MODELS_DIR", other_root / "models"), patch.object(base_json, "JSON_STORE_DIR", other_root / "json_store"):
                pc2 = ModelExportService(ModelsRepository(), StubTemplatesRepo(), StubDeploymentsRepo())
                again = pc2.import_model_archive(bundle["archive_path"], original_filename=bundle["archive_name"])
            self.assertEqual(again["imported_name"], "Line A")
            self.assertEqual(again["runtime"], "openvino")
            self.assertEqual(again["class_names"], ["sticker", "sticker_bad"])
            self.assertEqual(Path(again["model_dir"]).parent, other_root / "models")
            self.assertEqual(sorted(p.name for p in Path(again["model_dir"]).iterdir()),
                             ["best.bin", "best.meta.json", "best.xml", "metadata.yaml"])
        finally:
            Path(bundle["archive_path"]).unlink(missing_ok=True)

    def test_export_of_missing_file_fails_cleanly(self) -> None:
        record = self.models_repo.add_model("Ghost", str(self.models_dir / "ghost" / "best.xml"), "manual", runtime="openvino")
        with self.assertRaisesRegex(ValueError, "Model file not found"):
            self.service.create_export(int(record["id"]))


class PurgeTest(ImportTestBase):
    def test_purge_removes_imported_folder(self) -> None:
        result = self.service.import_model_archive(_openvino_zip(self.root / "p.zip"))
        record = self.models_repo.get_model(int(result["model_id"]))
        removed = purge_model_files(record)
        self.assertEqual(removed, [str(Path(result["model_dir"]))])
        self.assertFalse(Path(result["model_dir"]).exists())

    def test_purge_of_flat_openvino_model_removes_xml_bin_and_meta(self) -> None:
        (self.models_dir / "flat.xml").write_bytes(b"x")
        (self.models_dir / "flat.bin").write_bytes(b"b")
        (self.models_dir / "flat.meta.json").write_bytes(b"{}")
        record = {"path": str(self.models_dir / "flat.xml"), "meta_path": str(self.models_dir / "flat.meta.json")}
        removed = purge_model_files(record)
        self.assertEqual(sorted(Path(p).name for p in removed), ["flat.bin", "flat.meta.json", "flat.xml"])
        self.assertEqual(list(self.models_dir.iterdir()), [])

    def test_purge_never_touches_files_outside_models_dir(self) -> None:
        outside = self.root / "elsewhere" / "best.pt"
        outside.parent.mkdir()
        outside.write_bytes(b"keep")
        self.assertEqual(purge_model_files({"path": str(outside)}), [])
        self.assertTrue(outside.is_file())


class HelpersTest(unittest.TestCase):
    def test_class_names_from_yaml_orders_by_index(self) -> None:
        self.assertEqual(class_names_from_yaml(b"names:\n  2: c\n  0: a\n  1: b\n"), ["a", "b", "c"])
        self.assertEqual(class_names_from_yaml(b"names: [x, y]\n"), ["x", "y"])
        self.assertEqual(class_names_from_yaml(b"not: here\n"), [])
        self.assertEqual(class_names_from_yaml(b": : bad yaml ["), [])

    def test_repository_helpers_find_active_deployment_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(base_json, "JSON_STORE_DIR", Path(tmp)):
            repo = ModelsRepository()
            model_path = str(Path(tmp) / "models" / "demo" / "best.xml")
            templates_repo = StubTemplatesRepo({
                "version_id": 1,
                "template": {"name": "QC Line A", "vision": {"model_path": model_path}},
            })
            deployments_repo = StubDeploymentsRepo([
                {"id": 11, "template_id": 5, "template_version_id": 1, "template_name": "QC Line A",
                 "version_number": 1, "is_active": True, "effective_from": "2026-04-16T00:00:00+00:00", "effective_until": None},
                {"id": 12, "template_id": 6, "template_version_id": 2, "template_name": "Other",
                 "version_number": 1, "is_active": False, "effective_from": "2026-04-15T00:00:00+00:00", "effective_until": "2026-04-16T00:00:00+00:00"},
            ])
            conflict = repo.find_active_model_conflict(model_path, templates_repo=templates_repo, deployments_repo=deployments_repo)
            self.assertIsNotNone(conflict)
            self.assertEqual(conflict["deployment_id"], 11)
            references = repo.list_model_deployment_references(model_path, templates_repo=templates_repo, deployments_repo=deployments_repo)
            self.assertEqual([r["deployment_id"] for r in references], [11])


if __name__ == "__main__":
    unittest.main()
