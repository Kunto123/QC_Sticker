from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_ROOT = Path(tempfile.mkdtemp(prefix="qc-suite-versioning-tests-"))
atexit.register(lambda: shutil.rmtree(TEST_DATA_ROOT, ignore_errors=True))
os.environ["QC_SUITE_DATA_ROOT"] = str(TEST_DATA_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.repositories.dataset_versions_repository import DatasetVersionRepository
from backend.app.repositories.datasets_repository import DatasetsRepository


def _sample_image_bytes(color: tuple[int, int, int]) -> bytes:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :] = color
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Failed to encode sample image")
    return encoded.tobytes()


def _find_exported_label_path(export_root: Path, image_name: str) -> Path:
    label_name = f"{Path(image_name).stem}.txt"
    matches = list((export_root / "labels").rglob(label_name))
    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one exported label for {image_name}, found {len(matches)}")
    return matches[0]


def _read_single_yolo_row(label_path: Path) -> tuple[int, float, float, float, float]:
    rows = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1:
        raise AssertionError(f"Expected exactly one YOLO row in {label_path}, found {len(rows)}")
    parts = rows[0].split()
    if len(parts) != 5:
        raise AssertionError(f"Expected 5 YOLO columns in {label_path}, found {len(parts)}")
    return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])


class DatasetVersioningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.datasets_repo = DatasetsRepository()
        self.versions_repo = DatasetVersionRepository(self.datasets_repo)

    def test_create_version_exports_yolo_snapshot(self) -> None:
        dataset = self.datasets_repo.create_dataset("versioned-dataset", "smoke")
        self.datasets_repo.save_file(dataset["id"], "images", "a.jpg", _sample_image_bytes((10, 40, 80)))
        self.datasets_repo.save_file(dataset["id"], "images", "b.jpg", _sample_image_bytes((20, 60, 120)))
        self.datasets_repo.save_file(dataset["id"], "images", "c.jpg", _sample_image_bytes((30, 90, 160)))
        self.datasets_repo.save_annotation(
            dataset["id"],
            "a.jpg",
            [
                {
                    "type": "bbox",
                    "class": "bbox-class",
                    "bbox": {"x": 10, "y": 10, "w": 20, "h": 20},
                }
            ],
        )
        self.datasets_repo.save_annotation(
            dataset["id"],
            "b.jpg",
            [
                {
                    "type": "polygon",
                    "class": "poly-class",
                    "points": [
                        {"x": 30, "y": 40},
                        {"x": 60, "y": 40},
                        {"x": 60, "y": 70},
                    ],
                }
            ],
        )

        version = self.versions_repo.create_version(
            dataset["id"],
            {
                "name": "Smoke version",
                "description": "version export",
                "split_ratios": {"train": 0.6, "valid": 0.2, "test": 0.2},
            },
        )

        self.assertEqual(version["dataset_id"], dataset["id"])
        self.assertEqual(version["status"], "ready")
        self.assertTrue(version["ready_for_training"])
        self.assertEqual(version["image_count"], 3)
        self.assertEqual(version["annotated_image_count"], 2)
        self.assertEqual(len(version["class_names"]), 2)

        export_root = Path(version["export_root"])
        self.assertTrue(export_root.exists())
        self.assertTrue((export_root / "data.yaml").exists())
        self.assertTrue((export_root / "classes.txt").exists())
        data_yaml = (export_root / "data.yaml").read_text(encoding="utf-8")
        self.assertIn(f"path: {export_root.resolve().as_posix()}", data_yaml)
        self.assertIn("bbox-class", data_yaml)
        self.assertIn("poly-class", data_yaml)

        manifest = Path(version["manifest_path"])
        self.assertTrue(manifest.exists())
        self.assertIn("split_counts", manifest.read_text(encoding="utf-8"))

        image_exports = list((export_root / "images").rglob("*.jpg"))
        label_exports = list((export_root / "labels").rglob("*.txt"))
        self.assertEqual(len(image_exports), 3)
        self.assertEqual(len(label_exports), 3)

        # BBox labels in export must use YOLO center-based xywh, not top-left xywh.
        bbox_class_id, bbox_x, bbox_y, bbox_w, bbox_h = _read_single_yolo_row(
            _find_exported_label_path(export_root, "a.jpg")
        )
        self.assertEqual(bbox_class_id, 0)
        self.assertAlmostEqual(bbox_x, 0.0625, places=4)
        self.assertAlmostEqual(bbox_y, 0.083333, places=4)
        self.assertAlmostEqual(bbox_w, 0.0625, places=4)
        self.assertAlmostEqual(bbox_h, 0.083333, places=4)

        # Polygon annotations are converted to a bbox and must also be center-based.
        poly_class_id, poly_x, poly_y, poly_w, poly_h = _read_single_yolo_row(
            _find_exported_label_path(export_root, "b.jpg")
        )
        self.assertEqual(poly_class_id, 1)
        self.assertAlmostEqual(poly_x, 0.140625, places=4)
        self.assertAlmostEqual(poly_y, 0.229167, places=4)
        self.assertAlmostEqual(poly_w, 0.09375, places=4)
        self.assertAlmostEqual(poly_h, 0.125, places=4)

        refreshed = self.versions_repo.export_version(dataset["id"], version["id"])
        self.assertEqual(refreshed["id"], version["id"])
        self.assertTrue(Path(refreshed["export_root"]).exists())

    def test_create_version_converts_normalized_top_left_bbox_to_center(self) -> None:
        dataset = self.datasets_repo.create_dataset("normalized-bbox-dataset", "normalized")
        self.datasets_repo.save_file(dataset["id"], "images", "n.jpg", _sample_image_bytes((12, 34, 56)))
        self.datasets_repo.save_annotation(
            dataset["id"],
            "n.jpg",
            [
                {
                    "type": "bbox",
                    "class": "normalized-class",
                    "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
                    "normalized": True,
                }
            ],
        )

        version = self.versions_repo.create_version(dataset["id"])
        export_root = Path(version["export_root"])
        class_id, x, y, width, height = _read_single_yolo_row(_find_exported_label_path(export_root, "n.jpg"))

        self.assertEqual(class_id, 0)
        self.assertAlmostEqual(x, 0.25, places=4)
        self.assertAlmostEqual(y, 0.4, places=4)
        self.assertAlmostEqual(width, 0.3, places=4)
        self.assertAlmostEqual(height, 0.4, places=4)


# ---------------------------------------------------------------------------
# Augment integration — create_version with augment_jobs parameter
# ---------------------------------------------------------------------------

class AugmentIntegrationVersioningTest(unittest.TestCase):
    """create_version correctly integrates augmented images into the snapshot."""

    def setUp(self) -> None:
        self.datasets_repo = DatasetsRepository()
        self.versions_repo = DatasetVersionRepository(self.datasets_repo)
        # Create dataset with 2 original images and one annotation.
        self.dataset = self.datasets_repo.create_dataset("aug-integration-ds", "augment test")
        self.datasets_repo.save_file(self.dataset["id"], "images", "img1.jpg", _sample_image_bytes((10, 40, 80)))
        self.datasets_repo.save_file(self.dataset["id"], "images", "img2.jpg", _sample_image_bytes((20, 60, 120)))
        self.datasets_repo.save_annotation(
            self.dataset["id"],
            "img1.jpg",
            [{"type": "bbox", "class": "obj", "bbox": {"x": 10, "y": 10, "w": 20, "h": 20}}],
        )

    def _make_augment_job_stub(self, multiplier: int = 2, *, source_image: str = "img1.jpg") -> dict:
        """Create a synthetic completed augment job with photometric output files."""
        from backend.app.repositories.datasets_repository import DatasetsRepository as _DR
        ds_dir = _DR().dataset_dir(self.dataset["id"])
        job_id = "augtest001"
        output_dir = ds_dir / "augmented" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        source_stem = Path(source_image).stem
        # Write augmented copies like img1_aug001.jpg, img1_aug002.jpg.
        for i in range(1, multiplier + 1):
            shutil.copy2(ds_dir / "images" / source_image, output_dir / f"{source_stem}_aug{i:03d}.jpg")
        return {
            "id": job_id,
            "dataset_id": self.dataset["id"],
            "status": "completed",
            "transforms": ["brightness", "contrast"],
            "multiplier": multiplier,
            "output_dataset_id": str(output_dir),
            "augmented_image_count": multiplier,
        }

    def test_create_version_without_augment_unchanged(self) -> None:
        """create_version with no augment_jobs behaves identically to before."""
        version = self.versions_repo.create_version(self.dataset["id"])
        self.assertEqual(version["image_count"], 2)
        self.assertEqual(version["label_count"], 1)
        self.assertEqual(version["annotated_image_count"], 1)
        self.assertEqual(version["annotation_coverage"], 0.5)
        self.assertEqual(version["selected_augment_job_ids"], [])
        self.assertEqual(version["augmented_image_count_in_version"], 0)
        self.assertEqual(version["total_source_image_count"], 2)
        self.assertIn("1/2 ann", version["display_label"])

    def test_create_version_with_augment_increases_image_count(self) -> None:
        """image_count in version record = original + augmented when augment_jobs are provided."""
        job = self._make_augment_job_stub(multiplier=2)
        version = self.versions_repo.create_version(self.dataset["id"], augment_jobs=[job])
        # 2 original + 2 augmented copies of img1
        self.assertEqual(version["image_count"], 4)
        self.assertEqual(version["label_count"], 3)
        self.assertEqual(version["annotated_image_count"], 3)
        self.assertEqual(version["annotation_coverage"], 0.75)
        self.assertEqual(version["augmented_image_count_in_version"], 2)
        self.assertEqual(version["total_source_image_count"], 2)
        self.assertIn(job["id"], version["selected_augment_job_ids"])
        self.assertIn("3/4 ann", version["display_label"])

    def test_create_version_augmenting_unannotated_image_keeps_annotation_counts(self) -> None:
        """Augmenting an unannotated source image must not inflate annotated-image metrics."""
        job = self._make_augment_job_stub(multiplier=1, source_image="img2.jpg")
        version = self.versions_repo.create_version(self.dataset["id"], augment_jobs=[job])
        self.assertEqual(version["image_count"], 3)
        self.assertEqual(version["label_count"], 1)
        self.assertEqual(version["annotated_image_count"], 1)
        self.assertEqual(version["annotation_coverage"], round(1 / 3, 4))
        self.assertIn("1/3 ann", version["display_label"])

    def test_create_version_augment_export_includes_augmented_images(self) -> None:
        """Augmented images are stored in version_root/augmented/images (not in export)."""
        job = self._make_augment_job_stub(multiplier=2)
        version = self.versions_repo.create_version(self.dataset["id"], augment_jobs=[job])
        # Augmented images go to augmented/ folder, not export
        version_root = Path(version["source_root"]).parent
        aug_images = list((version_root / "augmented" / "images").rglob("*.jpg"))
        self.assertEqual(len(aug_images), 2)
        aug_names = {p.name for p in aug_images}
        self.assertIn("img1_aug001.jpg", aug_names)
        self.assertIn("img1_aug002.jpg", aug_names)
        # Source labels dir has augmented JSON labels
        source_labels_dir = Path(version["source_root"]) / "labels"
        aug_label = source_labels_dir / "img1_aug001.json"
        self.assertTrue(aug_label.exists(), f"Expected {aug_label} to exist")

    def test_create_version_augmented_image_gets_original_annotation(self) -> None:
        """Augmented images inherit their original image's annotation in the snapshot labels dir."""
        job = self._make_augment_job_stub(multiplier=1)
        version = self.versions_repo.create_version(self.dataset["id"], augment_jobs=[job])
        source_labels_dir = Path(version["source_root"]) / "labels"
        aug_label = source_labels_dir / "img1_aug001.json"
        self.assertTrue(aug_label.exists(), f"Expected {aug_label} to exist")
        import json as _json
        payload = _json.loads(aug_label.read_text(encoding="utf-8"))
        labels = payload.get("labels") or []
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].get("class"), "obj")

    def test_create_version_missing_augment_output_dir_is_skipped(self) -> None:
        """If the augment job output_dataset_id dir doesn't exist, it's silently skipped."""
        job = {
            "id": "missing-aug",
            "dataset_id": self.dataset["id"],
            "status": "completed",
            "transforms": ["brightness"],
            "multiplier": 1,
            "output_dataset_id": "/nonexistent/path/to/augmented/missing-aug",
            "augmented_image_count": 0,
        }
        version = self.versions_repo.create_version(self.dataset["id"], augment_jobs=[job])
        # No augmented images should be added
        self.assertEqual(version["augmented_image_count_in_version"], 0)
        self.assertEqual(version["image_count"], 2)


# ---------------------------------------------------------------------------
# Phase 0 — Baseline Freeze
# ---------------------------------------------------------------------------

class Phase0BaselineTest(unittest.TestCase):
    """Baseline freeze: lock current behavior before geometric label transforms are added.

    These tests define the expected behavior when ``geometric_augment_enabled=False``
    (the default that will be introduced in Phase 1).  They must continue to pass
    unchanged throughout all subsequent phases.
    """

    def setUp(self) -> None:
        self.datasets_repo = DatasetsRepository()
        self.versions_repo = DatasetVersionRepository(self.datasets_repo)
        self.dataset = self.datasets_repo.create_dataset("phase0-baseline-ds", "phase0")
        # One annotated source image (320×240 px solid colour).
        self.datasets_repo.save_file(
            self.dataset["id"], "images", "src.jpg", _sample_image_bytes((128, 64, 32))
        )
        self.datasets_repo.save_annotation(
            self.dataset["id"],
            "src.jpg",
            [{"type": "bbox", "class": "obj", "bbox": {"x": 10, "y": 20, "w": 40, "h": 30}}],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_aug_output(self, transforms: list[str], multiplier: int = 1, *, job_id: str = "p0-aug-001") -> dict:
        """Apply ``_apply_transforms`` and write output files; return a completed job stub."""
        import cv2
        from backend.app.workers.augment_worker import _apply_transforms

        ds_dir = self.datasets_repo.dataset_dir(self.dataset["id"])
        output_dir = ds_dir / "augmented" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        src = cv2.imread(str(ds_dir / "images" / "src.jpg"))
        for i in range(1, multiplier + 1):
            out = _apply_transforms(src.copy(), transforms)
            cv2.imwrite(str(output_dir / f"src_aug{i:03d}.jpg"), out)
        return {
            "id": job_id,
            "dataset_id": self.dataset["id"],
            "status": "completed",
            "transforms": transforms,
            "multiplier": multiplier,
            "output_dataset_id": str(output_dir),
            "augmented_image_count": multiplier,
        }

    # ------------------------------------------------------------------
    # _apply_transforms — image-level invariants
    # ------------------------------------------------------------------

    def test_apply_transforms_preserves_shape_for_all_transforms(self) -> None:
        """Every supported transform must preserve image height×width×channels."""
        import cv2
        from backend.app.workers.augment_worker import _apply_transforms

        ds_dir = self.datasets_repo.dataset_dir(self.dataset["id"])
        src = cv2.imread(str(ds_dir / "images" / "src.jpg"))
        original_shape = src.shape
        all_transforms = ["flip_h", "flip_v", "brightness", "contrast", "blur", "rotate", "noise"]
        for t in all_transforms:
            out = _apply_transforms(src.copy(), [t])
            self.assertEqual(
                out.shape, original_shape,
                f"_apply_transforms changed image shape for transform '{t}': "
                f"{original_shape} → {out.shape}",
            )

    def test_apply_transforms_flip_h_mirrors_image(self) -> None:
        """flip_h must produce the horizontally mirrored image (cv2.flip axis=1)."""
        import cv2
        from backend.app.workers.augment_worker import _apply_transforms

        ds_dir = self.datasets_repo.dataset_dir(self.dataset["id"])
        src = cv2.imread(str(ds_dir / "images" / "src.jpg"))
        out = _apply_transforms(src.copy(), ["flip_h"])
        expected = cv2.flip(src, 1)
        self.assertTrue(
            (out == expected).all(),
            "flip_h output does not match cv2.flip(img, 1)",
        )

    def test_apply_transforms_flip_v_mirrors_image(self) -> None:
        """flip_v must produce the vertically mirrored image (cv2.flip axis=0)."""
        import cv2
        from backend.app.workers.augment_worker import _apply_transforms

        ds_dir = self.datasets_repo.dataset_dir(self.dataset["id"])
        src = cv2.imread(str(ds_dir / "images" / "src.jpg"))
        out = _apply_transforms(src.copy(), ["flip_v"])
        expected = cv2.flip(src, 0)
        self.assertTrue(
            (out == expected).all(),
            "flip_v output does not match cv2.flip(img, 0)",
        )

    # ------------------------------------------------------------------
    # Snapshot annotation behaviour — flag OFF (verbatim copy)
    # ------------------------------------------------------------------

    def _read_aug_label(self, version: dict, aug_name: str = "src_aug001.json") -> dict:
        import json as _json
        label_path = Path(version["source_root"]) / "labels" / aug_name
        self.assertTrue(label_path.exists(), f"Expected label file {label_path}")
        return _json.loads(label_path.read_text(encoding="utf-8"))

    def test_photometric_augment_copies_annotation_verbatim(self) -> None:
        """Photometric transforms (brightness/contrast/blur/noise) → annotation coords unchanged."""
        job = self._write_aug_output(["brightness", "contrast", "blur", "noise"])
        version = self.versions_repo.create_version(self.dataset["id"], augment_jobs=[job])
        payload = self._read_aug_label(version)
        labels = payload.get("labels") or []
        self.assertEqual(len(labels), 1)
        bbox = labels[0].get("bbox") or {}
        self.assertEqual(bbox.get("x"), 10)
        self.assertEqual(bbox.get("y"), 20)
        self.assertEqual(bbox.get("w"), 40)
        self.assertEqual(bbox.get("h"), 30)

    def test_geometric_flip_h_copies_annotation_verbatim_when_flag_off(self) -> None:
        """flip_h with no label-transform engine → annotation copied verbatim (flag OFF baseline).

        When the geometric_augment feature flag is OFF (default), this verbatim-copy
        behaviour is preserved — the snapshot does NOT attempt to mirror the bbox.
        """
        job = self._write_aug_output(["flip_h"], job_id="p0-flip-h-001")
        version = self.versions_repo.create_version(self.dataset["id"], augment_jobs=[job])
        payload = self._read_aug_label(version)
        labels = payload.get("labels") or []
        self.assertEqual(len(labels), 1)
        bbox = labels[0].get("bbox") or {}
        # Verbatim: coords must equal original, NOT mirror-transformed
        self.assertEqual(bbox.get("x"), 10)
        self.assertEqual(bbox.get("y"), 20)
        self.assertEqual(bbox.get("w"), 40)
        self.assertEqual(bbox.get("h"), 30)

    def test_geometric_flip_v_copies_annotation_verbatim_when_flag_off(self) -> None:
        """flip_v with no label-transform engine → annotation copied verbatim (flag OFF baseline)."""
        job = self._write_aug_output(["flip_v"], job_id="p0-flip-v-001")
        version = self.versions_repo.create_version(self.dataset["id"], augment_jobs=[job])
        payload = self._read_aug_label(version)
        labels = payload.get("labels") or []
        self.assertEqual(len(labels), 1)
        bbox = labels[0].get("bbox") or {}
        self.assertEqual(bbox.get("x"), 10)
        self.assertEqual(bbox.get("y"), 20)
        self.assertEqual(bbox.get("w"), 40)
        self.assertEqual(bbox.get("h"), 30)

    def test_geometric_rotate_copies_annotation_verbatim_when_flag_off(self) -> None:
        """rotate with no label-transform engine → annotation copied verbatim (flag OFF baseline)."""
        job = self._write_aug_output(["rotate"], job_id="p0-rotate-001")
        version = self.versions_repo.create_version(self.dataset["id"], augment_jobs=[job])
        payload = self._read_aug_label(version)
        labels = payload.get("labels") or []
        self.assertEqual(len(labels), 1)
        bbox = labels[0].get("bbox") or {}
        self.assertEqual(bbox.get("x"), 10)
        self.assertEqual(bbox.get("y"), 20)
        self.assertEqual(bbox.get("w"), 40)
        self.assertEqual(bbox.get("h"), 30)

    # ------------------------------------------------------------------
    # Version metrics invariants
    # ------------------------------------------------------------------

    def test_no_augment_version_metrics_unchanged(self) -> None:
        """create_version without augment_jobs: metrics identical to pre-augment baseline."""
        version = self.versions_repo.create_version(self.dataset["id"])
        self.assertEqual(version["image_count"], 1)
        self.assertEqual(version["annotated_image_count"], 1)
        self.assertEqual(version["label_count"], 1)
        self.assertEqual(version["annotation_coverage"], 1.0)
        self.assertEqual(version["selected_augment_job_ids"], [])
        self.assertEqual(version["augmented_image_count_in_version"], 0)

    def test_photometric_augment_version_metrics_correct(self) -> None:
        """Photometric augment: image_count = original + augmented; annotation_coverage correct."""
        job = self._write_aug_output(["brightness"], multiplier=2, job_id="p0-photo-metrics")
        version = self.versions_repo.create_version(self.dataset["id"], augment_jobs=[job])
        # 1 original + 2 augmented
        self.assertEqual(version["image_count"], 3)
        self.assertEqual(version["annotated_image_count"], 3)
        self.assertEqual(version["label_count"], 3)
        self.assertEqual(version["augmented_image_count_in_version"], 2)
        self.assertEqual(version["total_source_image_count"], 1)

    def test_photometric_augment_export_contains_augmented_images(self) -> None:
        """YOLO export must include both original and augmented images for photometric transforms."""
        job = self._write_aug_output(["blur"], multiplier=1, job_id="p0-export-check")
        version = self.versions_repo.create_version(self.dataset["id"], augment_jobs=[job])
        export_root = Path(version["export_root"])
        exported = {p.name for p in (export_root / "images").rglob("*.jpg")}
        self.assertIn("src.jpg", exported)
        self.assertIn("src_aug001.jpg", exported)


# ---------------------------------------------------------------------------
# Phase 8 — Verification Matrix
# ---------------------------------------------------------------------------

class Phase8VerificationTest(unittest.TestCase):
    """Verification matrix: dual-mode ON/OFF flag, per-transform label correctness, mixed chains."""

    # Image is 320×240 px.  Bbox is at (10, 20, 40, 30) → right edge at 50, bottom edge at 50.
    _IMG_W = 320
    _IMG_H = 240
    _ORIG_X = 10.0
    _ORIG_Y = 20.0
    _ORIG_W = 40.0
    _ORIG_H = 30.0

    def setUp(self) -> None:
        self.datasets_repo = DatasetsRepository()
        self.versions_repo_off = DatasetVersionRepository(
            self.datasets_repo, geometric_augment_enabled=False
        )
        self.versions_repo_on = DatasetVersionRepository(
            self.datasets_repo, geometric_augment_enabled=True
        )
        self.dataset = self.datasets_repo.create_dataset("p8-verify-ds", "phase8")
        self.datasets_repo.save_file(
            self.dataset["id"], "images", "src.jpg",
            _sample_image_bytes((100, 150, 200)),
        )
        self.datasets_repo.save_annotation(
            self.dataset["id"],
            "src.jpg",
            [{
                "type": "bbox",
                "class": "obj",
                "bbox": {
                    "x": self._ORIG_X, "y": self._ORIG_Y,
                    "w": self._ORIG_W, "h": self._ORIG_H,
                },
            }],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_aug_with_trace(self, transforms: list[str], job_id: str) -> dict:
        """Apply transforms, write image + trace sidecar; return completed job stub."""
        import cv2
        from backend.app.workers.augment_worker import _apply_transforms_traced

        ds_dir = self.datasets_repo.dataset_dir(self.dataset["id"])
        output_dir = ds_dir / "augmented" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        src = cv2.imread(str(ds_dir / "images" / "src.jpg"))
        h, w = src.shape[:2]

        import json as _json
        out_img, trace = _apply_transforms_traced(src.copy(), transforms)
        cv2.imwrite(str(output_dir / "src_aug001.jpg"), out_img)
        (_json.dumps({
            "source_image": "src.jpg",
            "aug_index": 1,
            "image_width": w,
            "image_height": h,
            "transforms": trace,
        }, ensure_ascii=True, indent=2))
        trace_path = output_dir / "src_aug001.trace.json"
        trace_path.write_text(
            _json.dumps({
                "source_image": "src.jpg",
                "aug_index": 1,
                "image_width": w,
                "image_height": h,
                "transforms": trace,
            }, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return {
            "id": job_id,
            "dataset_id": self.dataset["id"],
            "status": "completed",
            "transforms": transforms,
            "multiplier": 1,
            "output_dataset_id": str(output_dir),
            "augmented_image_count": 1,
        }

    def _read_aug_bbox(self, version: dict, aug_stem: str = "src_aug001") -> dict:
        import json as _json
        label_path = Path(version["source_root"]) / "labels" / f"{aug_stem}.json"
        payload = _json.loads(label_path.read_text(encoding="utf-8"))
        labels = payload.get("labels") or []
        self.assertEqual(len(labels), 1)
        return labels[0].get("bbox") or {}

    # ------------------------------------------------------------------
    # Compatibility: flag OFF preserves baseline (verbatim copy)
    # ------------------------------------------------------------------

    def test_flag_off_flip_h_annotation_verbatim(self) -> None:
        """flag=OFF + flip_h → annotation verbatim (baseline compat)."""
        job = self._write_aug_with_trace(["flip_h"], "p8-off-flip-h")
        version = self.versions_repo_off.create_version(self.dataset["id"], augment_jobs=[job])
        bbox = self._read_aug_bbox(version)
        self.assertAlmostEqual(bbox["x"], self._ORIG_X, places=3)
        self.assertAlmostEqual(bbox["y"], self._ORIG_Y, places=3)

    def test_flag_off_rotate_annotation_verbatim(self) -> None:
        """flag=OFF + rotate → annotation verbatim (baseline compat)."""
        job = self._write_aug_with_trace(["rotate"], "p8-off-rotate")
        version = self.versions_repo_off.create_version(self.dataset["id"], augment_jobs=[job])
        bbox = self._read_aug_bbox(version)
        self.assertAlmostEqual(bbox["x"], self._ORIG_X, places=3)

    def test_flag_off_photometric_annotation_verbatim(self) -> None:
        """flag=OFF + brightness → annotation verbatim (correct for photometric)."""
        job = self._write_aug_with_trace(["brightness"], "p8-off-brightness")
        version = self.versions_repo_off.create_version(self.dataset["id"], augment_jobs=[job])
        bbox = self._read_aug_bbox(version)
        self.assertAlmostEqual(bbox["x"], self._ORIG_X, places=3)

    # ------------------------------------------------------------------
    # Geometry math: flag ON, per-transform label correctness
    # ------------------------------------------------------------------

    def test_flag_on_flip_h_transforms_bbox_correctly(self) -> None:
        """flag=ON + flip_h → x' = W - x - w; y/w/h unchanged."""
        job = self._write_aug_with_trace(["flip_h"], "p8-on-flip-h")
        version = self.versions_repo_on.create_version(self.dataset["id"], augment_jobs=[job])
        bbox = self._read_aug_bbox(version)
        expected_x = self._IMG_W - self._ORIG_X - self._ORIG_W  # 320 - 10 - 40 = 270
        self.assertAlmostEqual(bbox["x"], expected_x, places=2,
            msg=f"flip_h x: expected {expected_x}, got {bbox['x']}")
        self.assertAlmostEqual(bbox["y"], self._ORIG_Y, places=2)
        self.assertAlmostEqual(bbox["w"], self._ORIG_W, places=2)
        self.assertAlmostEqual(bbox["h"], self._ORIG_H, places=2)

    def test_flag_on_flip_v_transforms_bbox_correctly(self) -> None:
        """flag=ON + flip_v → y' = H - y - h; x/w/h unchanged."""
        job = self._write_aug_with_trace(["flip_v"], "p8-on-flip-v")
        version = self.versions_repo_on.create_version(self.dataset["id"], augment_jobs=[job])
        bbox = self._read_aug_bbox(version)
        expected_y = self._IMG_H - self._ORIG_Y - self._ORIG_H  # 240 - 20 - 30 = 190
        self.assertAlmostEqual(bbox["x"], self._ORIG_X, places=2)
        self.assertAlmostEqual(bbox["y"], expected_y, places=2,
            msg=f"flip_v y: expected {expected_y}, got {bbox['y']}")
        self.assertAlmostEqual(bbox["w"], self._ORIG_W, places=2)
        self.assertAlmostEqual(bbox["h"], self._ORIG_H, places=2)

    def test_flag_on_photometric_coords_unchanged(self) -> None:
        """flag=ON + brightness → annotation coords unchanged (photometric regression)."""
        job = self._write_aug_with_trace(["brightness"], "p8-on-brightness")
        version = self.versions_repo_on.create_version(self.dataset["id"], augment_jobs=[job])
        bbox = self._read_aug_bbox(version)
        self.assertAlmostEqual(bbox["x"], self._ORIG_X, places=3)
        self.assertAlmostEqual(bbox["y"], self._ORIG_Y, places=3)
        self.assertAlmostEqual(bbox["w"], self._ORIG_W, places=3)
        self.assertAlmostEqual(bbox["h"], self._ORIG_H, places=3)

    def test_flag_on_rotate_bbox_stays_within_image(self) -> None:
        """flag=ON + rotate → transformed bbox clipped inside image bounds."""
        job = self._write_aug_with_trace(["rotate"], "p8-on-rotate")
        version = self.versions_repo_on.create_version(self.dataset["id"], augment_jobs=[job])
        bbox = self._read_aug_bbox(version)
        self.assertGreaterEqual(bbox["x"], 0.0)
        self.assertGreaterEqual(bbox["y"], 0.0)
        self.assertGreater(bbox["w"], 0.0)
        self.assertGreater(bbox["h"], 0.0)
        self.assertLessEqual(bbox["x"] + bbox["w"], self._IMG_W + 1e-3)
        self.assertLessEqual(bbox["y"] + bbox["h"], self._IMG_H + 1e-3)

    # ------------------------------------------------------------------
    # Mixed chain: photometric + geometric
    # ------------------------------------------------------------------

    def test_flag_on_mixed_brightness_flip_h(self) -> None:
        """flag=ON + [brightness, flip_h] → only the flip_h changes coords."""
        job = self._write_aug_with_trace(["brightness", "flip_h"], "p8-on-mixed-bfh")
        version = self.versions_repo_on.create_version(self.dataset["id"], augment_jobs=[job])
        bbox = self._read_aug_bbox(version)
        # brightness is photometric — no coord change; flip_h mirrors x
        expected_x = self._IMG_W - self._ORIG_X - self._ORIG_W
        self.assertAlmostEqual(bbox["x"], expected_x, places=2)
        self.assertAlmostEqual(bbox["y"], self._ORIG_Y, places=2)

    def test_flag_on_flip_h_then_flip_v_double_flip(self) -> None:
        """flag=ON + [flip_h, flip_v] → both flips applied sequentially."""
        job = self._write_aug_with_trace(["flip_h", "flip_v"], "p8-on-double-flip")
        version = self.versions_repo_on.create_version(self.dataset["id"], augment_jobs=[job])
        bbox = self._read_aug_bbox(version)
        expected_x = self._IMG_W - self._ORIG_X - self._ORIG_W  # 270
        expected_y = self._IMG_H - self._ORIG_Y - self._ORIG_H  # 190
        self.assertAlmostEqual(bbox["x"], expected_x, places=2)
        self.assertAlmostEqual(bbox["y"], expected_y, places=2)
        self.assertAlmostEqual(bbox["w"], self._ORIG_W, places=2)
        self.assertAlmostEqual(bbox["h"], self._ORIG_H, places=2)

    # ------------------------------------------------------------------
    # Snapshot / export integrity with geometric transforms
    # ------------------------------------------------------------------

    def test_flag_on_flip_h_version_metrics_consistent(self) -> None:
        """flag=ON + flip_h: image_count, annotated_count, source labels all consistent."""
        job = self._write_aug_with_trace(["flip_h"], "p8-on-metrics-flip")
        version = self.versions_repo_on.create_version(self.dataset["id"], augment_jobs=[job])
        self.assertEqual(version["image_count"], 2)
        self.assertEqual(version["annotated_image_count"], 2)
        self.assertEqual(version["annotation_coverage"], 1.0)
        # Augmented labels are in source_labels/ as JSON
        source_labels_dir = Path(version["source_root"]) / "labels"
        aug_label = source_labels_dir / "src_aug001.json"
        self.assertTrue(aug_label.exists(), f"Expected {aug_label} to exist")
        import json as _json
        payload = _json.loads(aug_label.read_text(encoding="utf-8"))
        labels = payload.get("labels") or []
        self.assertEqual(len(labels), 1)

    def test_flag_off_no_trace_file_still_works_verbatim(self) -> None:
        """flag=OFF with no trace sidecar: annotation copied verbatim (safe fallback)."""
        import cv2
        ds_dir = self.datasets_repo.dataset_dir(self.dataset["id"])
        job_id = "p8-no-trace"
        output_dir = ds_dir / "augmented" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        src = cv2.imread(str(ds_dir / "images" / "src.jpg"))
        cv2.imwrite(str(output_dir / "src_aug001.jpg"), src)
        # Deliberately do NOT write a .trace.json
        job = {
            "id": job_id,
            "dataset_id": self.dataset["id"],
            "status": "completed",
            "transforms": ["brightness"],
            "multiplier": 1,
            "output_dataset_id": str(output_dir),
            "augmented_image_count": 1,
        }
        version = self.versions_repo_off.create_version(self.dataset["id"], augment_jobs=[job])
        bbox = self._read_aug_bbox(version)
        self.assertAlmostEqual(bbox["x"], self._ORIG_X, places=3)

    def test_flag_on_missing_trace_falls_back_to_verbatim(self) -> None:
        """flag=ON but no .trace.json present: graceful fallback to verbatim copy (no crash)."""
        import cv2
        ds_dir = self.datasets_repo.dataset_dir(self.dataset["id"])
        job_id = "p8-on-no-trace"
        output_dir = ds_dir / "augmented" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        src = cv2.imread(str(ds_dir / "images" / "src.jpg"))
        cv2.imwrite(str(output_dir / "src_aug001.jpg"), src)
        # No trace file
        job = {
            "id": job_id,
            "dataset_id": self.dataset["id"],
            "status": "completed",
            "transforms": ["flip_h"],
            "multiplier": 1,
            "output_dataset_id": str(output_dir),
            "augmented_image_count": 1,
        }
        # Should not raise; annotation is verbatim (no trace = no transform)
        version = self.versions_repo_on.create_version(self.dataset["id"], augment_jobs=[job])
        bbox = self._read_aug_bbox(version)
        self.assertAlmostEqual(bbox["x"], self._ORIG_X, places=3)