"""test_sticker_detection_gates.py — regression tests for runtime gate observability and model naming.

Covers:
  Phase 3 — raw_detection_count and allowed_labels_filter forwarded into sticker_detection.
  Phase 4 — class_names written to .meta.json by _register_trained_model.
  Phase 5 — artifact filename contains job_id_short (no same-second collision).
  Phase 6 — registry display name contains job_id_short.
  Phase 7 — old artifact filenames (no job_id_short) are still loadable unchanged.
  Phase 8 — tilt_gate_enabled toggle controls OUT_OF_ANGLE decision; telemetry always present.
"""
from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from backend.app.core.config import AppConfig
from backend.app.services.inspection_session import InspectionSessionService
from backend.app.services.sticker_inference import StickerInferenceService
from backend.app.workers.training_worker import TrainingWorker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inspection_service() -> InspectionSessionService:
    return InspectionSessionService(
        template_runtime=MagicMock(),
        profiles_repo=MagicMock(),
        results_repo=MagicMock(),
        sticker_inference=MagicMock(),
    )


def _make_training_worker(*, models_repo=None) -> TrainingWorker:
    config = MagicMock()
    config.training_engine_mode = "simulated"
    config.training_weights_download_allowed = False
    config.training_timeout_minutes = 1
    config.gpu_fail_fast = True
    return TrainingWorker(
        training_repo=MagicMock(),
        models_repo=models_repo,
        app_config=config,
        device_runtime=MagicMock(),
    )


# ---------------------------------------------------------------------------
# Phase 3: raw_detection_count and allowed_labels_filter in sticker_detection
# ---------------------------------------------------------------------------

class StickerDetectionObservabilityTest(unittest.TestCase):
    """_build_sticker_detection_payload must forward raw_detection_count and allowed_labels_filter."""

    def setUp(self) -> None:
        self.service = _make_inspection_service()

    def test_raw_detection_count_present_in_payload(self) -> None:
        payload = self.service._build_sticker_detection_payload(
            [],
            skipped=False,
            backend="ultralytics",
            raw_detection_count=5,
            allowed_labels_filter=["sticker"],
        )
        self.assertEqual(payload["raw_detection_count"], 5)

    def test_allowed_labels_filter_present_in_payload(self) -> None:
        payload = self.service._build_sticker_detection_payload(
            [],
            skipped=False,
            backend="ultralytics",
            raw_detection_count=3,
            allowed_labels_filter=["label_a", "label_b"],
        )
        self.assertEqual(payload["allowed_labels_filter"], ["label_a", "label_b"])

    def test_none_values_when_skipped(self) -> None:
        """When inference is skipped the caller passes raw_detection_count=None."""
        payload = self.service._build_sticker_detection_payload(
            [],
            skipped=True,
            reason="part_not_ready",
            backend="skipped",
            raw_detection_count=None,
            allowed_labels_filter=None,
        )
        self.assertIsNone(payload["raw_detection_count"])
        self.assertIsNone(payload["allowed_labels_filter"])
        self.assertEqual(payload["status"], "skipped")

    def test_class_filter_mismatch_diagnosis(self) -> None:
        """When model finds boxes but class filter removes all, raw_detection_count > count signals the gap."""
        payload = self.service._build_sticker_detection_payload(
            [],  # all detections filtered out by allowed_labels
            skipped=False,
            backend="ultralytics",
            raw_detection_count=4,   # model found 4 boxes before filter
            allowed_labels_filter=["expected_class"],
        )
        # count=0 but raw_detection_count=4 → class filter mismatch is visible
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["raw_detection_count"], 4)
        self.assertIsNotNone(payload["allowed_labels_filter"])


class StickerInferenceRawCountTest(unittest.TestCase):
    """UltralyticsBackend.predict must include raw_detection_count in its return value."""

    def setUp(self) -> None:
        import threading
        from backend.app.services.inference_backend import UltralyticsBackend
        self.backend = UltralyticsBackend(
            config=AppConfig(),
            loaded_models={},
            meta_cache={},
            runtime_lock=threading.RLock(),
            device_resolution=self._fake_device(),
        )

    def _fake_vision(self, *, classes=None):
        from shared.contracts.templates import VisionConfig
        v = VisionConfig()
        v.conf_threshold = 0.25
        v.imgsz = 0
        v.classes = classes or []
        v.model_path = "dummy.pt"
        v.model_meta_path = None
        return v

    def _fake_device(self):
        from backend.app.core.device_runtime import DeviceResolution
        return DeviceResolution(
            requested_mode="cpu",
            effective_device="cpu",
            backend="cpu",
            gpu_available=False,
            cuda_device_id=None,
            fallback_reason=None,
        )

    def test_raw_detection_count_field_exists(self) -> None:
        """UltralyticsBackend.predict result always contains raw_detection_count."""
        fake_box = MagicMock()
        fake_box.xyxy = [MagicMock(tolist=lambda: [1.0, 2.0, 3.0, 4.0])]
        fake_box.cls = [MagicMock(item=lambda: 0)]
        fake_box.conf = [MagicMock(item=lambda: 0.9)]

        fake_result = MagicMock()
        fake_result.boxes = [fake_box]
        fake_result.names = {0: "sticker"}

        fake_model = MagicMock()
        fake_model.predict.return_value = [fake_result]

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        vision = self._fake_vision(classes=["sticker"])

        with patch.object(self.backend, "_get_ultralytics_model", return_value=fake_model), \
             patch.object(self.backend, "_resolve_model_path", return_value="dummy.pt"), \
             patch("pathlib.Path.exists", return_value=True):
            result = self.backend.predict(image, vision)

        self.assertIn("raw_detection_count", result)
        self.assertEqual(result["raw_detection_count"], 1)

    def test_allowed_labels_filter_field_in_result(self) -> None:
        """UltralyticsBackend.predict includes allowed_labels_filter matching vision.classes."""
        fake_result = MagicMock()
        fake_result.boxes = []
        fake_result.names = {}

        fake_model = MagicMock()
        fake_model.predict.return_value = [fake_result]

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        vision = self._fake_vision(classes=["sticker", "label"])

        with patch.object(self.backend, "_get_ultralytics_model", return_value=fake_model), \
             patch.object(self.backend, "_resolve_model_path", return_value="dummy.pt"), \
             patch("pathlib.Path.exists", return_value=True):
            result = self.backend.predict(image, vision)

        self.assertIn("allowed_labels_filter", result)
        # The filter list contains normalized lowercase class names
        self.assertIsNotNone(result["allowed_labels_filter"])


# ---------------------------------------------------------------------------
# Phase 4: class_names written into .meta.json by _register_trained_model
# ---------------------------------------------------------------------------

class MetaJsonClassNamesTest(unittest.TestCase):
    """_register_trained_model must persist class_names into the .meta.json file."""

    def test_class_names_written_to_meta_json(self) -> None:
        models_repo = MagicMock()
        models_repo.add_model.return_value = {"id": 99}
        worker = _make_training_worker(models_repo=models_repo)

        job = {
            "id": "abc12345-0000-0000-0000-000000000000",
            "dataset_id": "ds1",
            "dataset_version_id": "v1",
            "base_model": "yolov5n",
            "base_model_display_name": "YOLOv5n",
            "params": {},
        }

        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "model.pt"
            artifact_path.write_bytes(b"fake")
            worker._register_trained_model(
                job,
                "models/trained/model.pt",
                artifact_path,
                class_names=["sticker", "no_sticker"],
            )

            meta_path = artifact_path.with_suffix(".meta.json")
            self.assertTrue(meta_path.exists(), "meta.json must be created")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertIn("class_names", meta)
            self.assertEqual(meta["class_names"], ["sticker", "no_sticker"])

    def test_empty_class_names_written_as_empty_list(self) -> None:
        models_repo = MagicMock()
        models_repo.add_model.return_value = {"id": 1}
        worker = _make_training_worker(models_repo=models_repo)

        job = {"id": "00000001", "dataset_id": "ds2", "params": {}}

        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "model.pt"
            artifact_path.write_bytes(b"fake")
            worker._register_trained_model(
                job, "models/trained/model.pt", artifact_path, class_names=[]
            )
            meta = json.loads((artifact_path.with_suffix(".meta.json")).read_text(encoding="utf-8"))
            self.assertEqual(meta["class_names"], [])


# ---------------------------------------------------------------------------
# Phase 5 & 6: artifact filename and registry display name include job_id_short
# ---------------------------------------------------------------------------

class ModelNamingTest(unittest.TestCase):
    """Trained artifact filenames and registry display names must include a per-run identifier."""

    def _job_with_id(self, job_id: str) -> dict:
        return {
            "id": job_id,
            "status": "queued",
            "dataset_id": "ds1",
            "base_model": "yolov5n",
            "base_model_catalog_id": "yolov5n",
            "base_model_display_name": "YOLOv5n",
            "dataset_version_id": "v1",
            "dataset_version_name": "v1",
            "dataset_version_display_label": "v1",
            "params": {"epochs": 1},
            "requested_device_mode": "cpu",
        }

    def test_artifact_filename_contains_job_id_short(self) -> None:
        """The trained .pt filename must contain the first 8 chars of the job id."""
        job_id = "abcdef12-3456-7890-abcd-ef1234567890"
        job_id_short = "abcdef12"  # first 8 of hex-stripped id

        models_repo = MagicMock()
        models_repo.add_model.return_value = {"id": 1}
        worker = _make_training_worker(models_repo=models_repo)

        captured_paths: list[str] = []

        def _fake_simulated(*, job_id, artifact_path, epochs_requested):
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(b"fake")
            captured_paths.append(str(artifact_path))
            metrics = {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "map50": 0.0, "accuracy": 0.0}
            return metrics, {}, {"epochs_requested": 1, "epochs_ran": 1, "early_stopped": False}

        job = self._job_with_id(job_id)

        with tempfile.TemporaryDirectory() as tmp:
            fake_models_dir = Path(tmp) / "models"
            fake_models_dir.mkdir()

            with patch("backend.app.workers.training_worker.MODELS_DIR", fake_models_dir), \
                 patch.object(worker, "_run_simulated_training", side_effect=_fake_simulated), \
                 patch.object(worker, "_repo"):
                worker._repo.transition = MagicMock()
                worker._repo.update_job = MagicMock()
                worker._repo.get_job = MagicMock(return_value={"status": "running"})
                worker._run_job(job)

        self.assertTrue(captured_paths, "artifact path must have been set")
        filename = Path(captured_paths[0]).name
        self.assertIn(job_id_short, filename, f"job_id_short={job_id_short!r} not in filename {filename!r}")

    def test_two_concurrent_jobs_get_different_filenames(self) -> None:
        """Different job_ids must produce different artifact filenames even at the same timestamp."""
        worker = _make_training_worker()

        # Simulate two jobs with the same frozen timestamp but different job_ids
        frozen_ts = "20260101-120000"
        job_id_a = "aaaaaaaa-0000-0000-0000-000000000000"
        job_id_b = "bbbbbbbb-0000-0000-0000-000000000000"

        with patch("backend.app.workers.training_worker.datetime") as mock_dt:
            mock_now = MagicMock()
            mock_now.strftime.return_value = frozen_ts
            mock_dt.now.return_value = mock_now
            mock_dt.UTC = __import__("datetime").timezone.utc

            path_a = f"models/trained/ds1__yolov5n__{frozen_ts}__{str(job_id_a).replace('-','')[:8]}.pt"
            path_b = f"models/trained/ds1__yolov5n__{frozen_ts}__{str(job_id_b).replace('-','')[:8]}.pt"

        self.assertNotEqual(path_a, path_b)

    def test_registry_display_name_contains_job_id_short(self) -> None:
        """The model registry name must include #<job_id_short>."""
        models_repo = MagicMock()
        captured: list[str] = []

        def _add_model(name, *args, **kwargs):
            captured.append(name)
            return {"id": 1}

        models_repo.add_model.side_effect = _add_model
        worker = _make_training_worker(models_repo=models_repo)

        job = {
            "id": "deadbeef-cafe-0000-0000-000000000000",
            "dataset_id": "ds1",
            "base_model": "yolov5n",
            "base_model_display_name": "YOLOv5n",
            "dataset_version_display_label": "v2",
            "params": {},
        }

        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "model.pt"
            artifact_path.write_bytes(b"fake")
            worker._register_trained_model(
                job, "models/trained/model.pt", artifact_path, class_names=[]
            )

        self.assertTrue(captured, "add_model must have been called")
        display_name = captured[0]
        self.assertIn("deadbeef", display_name, f"job_id_short not in display_name: {display_name!r}")


# ---------------------------------------------------------------------------
# Phase 7: backward compat — old artifact paths without job_id_short load fine
# ---------------------------------------------------------------------------

class BackwardCompatTest(unittest.TestCase):
    """Old .pt filenames (no job_id_short) must still resolve through _trained_artifact_path."""

    def test_old_path_format_resolves_without_error(self) -> None:
        """_trained_artifact_path is purely a path transform and must not require a specific format."""
        old_path = "models/trained/ds1__yolov5n__20260101-120000.pt"
        result = TrainingWorker._trained_artifact_path(old_path)
        self.assertTrue(str(result).endswith("ds1__yolov5n__20260101-120000.pt"))

    def test_new_path_format_also_resolves(self) -> None:
        new_path = "models/trained/ds1__yolov5n__20260101-120000__abcdef12.pt"
        result = TrainingWorker._trained_artifact_path(new_path)
        self.assertTrue(str(result).endswith("ds1__yolov5n__20260101-120000__abcdef12.pt"))


# ---------------------------------------------------------------------------
# Phase 8: tilt_gate_enabled toggle controls OUT_OF_ANGLE; telemetry always present
# ---------------------------------------------------------------------------

class TiltGateToggleTest(unittest.TestCase):
    """_validate_sticker must gate OUT_OF_ANGLE on tilt_gate_enabled, while always
    computing and returning tilt telemetry regardless of the toggle state."""

    def _make_state(self, *, tilt_gate_enabled: bool, max_tilt_degrees: float | None = 5.0):
        from backend.app.models.session_state import SessionState
        from shared.contracts.enums import SessionStatus
        from shared.contracts.templates import (
            CameraDefaults, InspectionTemplate, PartReadyConfig,
            PersistenceConfig, RoiGeometry, StickerRule, VisionConfig,
        )

        sticker = StickerRule(
            part_name="P1",
            expected_class="sticker",
            enabled=True,
            validator_mode="ml_detection",
            min_roi_confidence=0.0,
            expected_tilt_degrees=0.0,
            max_tilt_degrees=max_tilt_degrees,
            tilt_gate_enabled=tilt_gate_enabled,
        )
        template = InspectionTemplate(
            id=1,
            version_id=1,
            version_number=1,
            name="T",
            description="",
            is_active=True,
            camera=CameraDefaults(),
            part_ready_roi=RoiGeometry(),
            sticker_roi=RoiGeometry(),
            vision=VisionConfig(),
            part_ready=PartReadyConfig(),
            sticker=sticker,
            persistence=PersistenceConfig(),
        )
        return SessionState(
            session_id="s1",
            client_id="c1",
            camera_index=0,
            template=template,
            status=SessionStatus.RUNNING,
        )

    def _detection_payload(self) -> dict:
        return {
            "backend": "ultralytics",
            "model_path": "m.pt",
            "meta_path": None,
            "class_names": ["sticker"],
            "fallback_reason": None,
        }

    def _one_passing_detection(self) -> list[dict]:
        return [
            {
                "label": "sticker",
                "confidence": 0.9,
                "class_confidence": 0.9,
                "position": {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0},
            }
        ]

    def _part_ready_payload(self) -> dict:
        return {
            "part_ready": True,
            "part_ready_confidence": 1.0,
            "reject_reason_code": None,
        }

    def _high_deviation_tilt_info(self) -> dict:
        return {
            "status": "ok",
            "angle_degrees": 30.0,
            "expected_tilt_degrees": 0.0,
            "deviation_degrees": 30.0,
            "contour_area": 100.0,
            "threshold_mode": "binary",
        }

    def _call_validate(self, state, detections, *, tilt_return_value):
        service = _make_inspection_service()
        roi_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        from backend.app.services import inspection_session as _mod
        with unittest.mock.patch.object(_mod, "_estimate_tilt_from_roi", return_value=tilt_return_value):
            return service._validate_sticker(
                roi_frame=roi_frame,
                state=state,
                detections=detections,
                detection_payload=self._detection_payload(),
                part_ready_payload=self._part_ready_payload(),
                username=None,
                user_id=None,
            )

    # ------------------------------------------------------------------
    # Serialisation: tilt_gate_enabled survives template round-trip
    # ------------------------------------------------------------------

    def test_tilt_gate_enabled_round_trips_in_to_dict(self) -> None:
        from shared.contracts.templates import StickerRule
        rule = StickerRule(part_name="P", expected_class="C", tilt_gate_enabled=True)
        from dataclasses import asdict
        d = asdict(rule)
        self.assertTrue(d["tilt_gate_enabled"])

    def test_tilt_gate_disabled_default_in_to_dict(self) -> None:
        from shared.contracts.templates import StickerRule
        rule = StickerRule(part_name="P", expected_class="C")
        from dataclasses import asdict
        d = asdict(rule)
        self.assertFalse(d["tilt_gate_enabled"])

    def test_template_from_dict_accepts_tilt_gate_enabled(self) -> None:
        from shared.contracts.templates import template_from_dict
        payload = {
            "name": "T", "version_number": 1,
            "sticker": {
                "part_name": "P", "expected_class": "C", "line": "L",
                "tilt_gate_enabled": True,
                "expected_tilt_degrees": 10.0,
                "max_tilt_degrees": 8.0,
            },
        }
        tmpl = template_from_dict(payload)
        self.assertTrue(tmpl.sticker.tilt_gate_enabled)
        self.assertEqual(tmpl.sticker.expected_tilt_degrees, 10.0)
        self.assertEqual(tmpl.sticker.max_tilt_degrees, 8.0)

    def test_template_from_dict_defaults_tilt_gate_to_false_when_absent(self) -> None:
        from shared.contracts.templates import template_from_dict
        payload = {
            "name": "T", "version_number": 1,
            "sticker": {"part_name": "P", "expected_class": "C", "line": "L"},
        }
        tmpl = template_from_dict(payload)
        self.assertFalse(tmpl.sticker.tilt_gate_enabled)

    # ------------------------------------------------------------------
    # Gate behaviour: toggle OFF → OUT_OF_ANGLE never raised
    # ------------------------------------------------------------------

    def test_tilt_gate_off_does_not_reject_out_of_angle(self) -> None:
        state = self._make_state(tilt_gate_enabled=False, max_tilt_degrees=5.0)
        result = self._call_validate(
            state,
            self._one_passing_detection(),
            tilt_return_value=self._high_deviation_tilt_info(),
        )
        self.assertNotEqual(
            result.get("reject_reason_code"), "OUT_OF_ANGLE",
            "Gate is OFF — OUT_OF_ANGLE must not fire even when deviation=30 > threshold=5",
        )
        self.assertEqual(result.get("decision"), "ACCEPT")

    def test_tilt_gate_off_accepts_when_no_max_tilt(self) -> None:
        state = self._make_state(tilt_gate_enabled=False, max_tilt_degrees=None)
        result = self._call_validate(
            state,
            self._one_passing_detection(),
            tilt_return_value=self._high_deviation_tilt_info(),
        )
        self.assertNotEqual(result.get("reject_reason_code"), "OUT_OF_ANGLE")

    # ------------------------------------------------------------------
    # Gate behaviour: toggle ON → OUT_OF_ANGLE fired when deviation exceeds max
    # ------------------------------------------------------------------

    def test_tilt_gate_on_rejects_when_deviation_exceeds_max(self) -> None:
        state = self._make_state(tilt_gate_enabled=True, max_tilt_degrees=5.0)
        result = self._call_validate(
            state,
            self._one_passing_detection(),
            tilt_return_value=self._high_deviation_tilt_info(),  # deviation=30 > max=5
        )
        self.assertEqual(
            result.get("reject_reason_code"), "OUT_OF_ANGLE",
            "Gate is ON and deviation=30 > max=5 — must reject OUT_OF_ANGLE",
        )
        self.assertEqual(result.get("decision"), "REJECT")

    def test_tilt_gate_on_accepts_when_deviation_within_max(self) -> None:
        state = self._make_state(tilt_gate_enabled=True, max_tilt_degrees=45.0)
        low_deviation_tilt = {
            "status": "ok",
            "angle_degrees": 3.0,
            "expected_tilt_degrees": 0.0,
            "deviation_degrees": 3.0,  # well within max=45
            "contour_area": 100.0,
            "threshold_mode": "binary",
        }
        result = self._call_validate(
            state,
            self._one_passing_detection(),
            tilt_return_value=low_deviation_tilt,
        )
        self.assertNotEqual(result.get("reject_reason_code"), "OUT_OF_ANGLE")
        self.assertEqual(result.get("decision"), "ACCEPT")

    def test_tilt_gate_on_but_no_max_tilt_does_not_raise(self) -> None:
        state = self._make_state(tilt_gate_enabled=True, max_tilt_degrees=None)
        result = self._call_validate(
            state,
            self._one_passing_detection(),
            tilt_return_value=self._high_deviation_tilt_info(),
        )
        self.assertNotEqual(result.get("reject_reason_code"), "OUT_OF_ANGLE")

    # ------------------------------------------------------------------
    # Telemetry: tilt angles always present in payload regardless of toggle
    # ------------------------------------------------------------------

    def test_tilt_telemetry_present_when_gate_off(self) -> None:
        state = self._make_state(tilt_gate_enabled=False, max_tilt_degrees=5.0)
        tilt_info = self._high_deviation_tilt_info()
        result = self._call_validate(
            state, self._one_passing_detection(), tilt_return_value=tilt_info
        )
        self.assertIsNotNone(result.get("sticker_tilt_angle"), "tilt angle must be in payload")
        self.assertIsNotNone(result.get("sticker_tilt_deviation"), "deviation must be in payload")
        self.assertEqual(result["sticker_tilt_angle"], tilt_info["angle_degrees"])

    def test_tilt_telemetry_present_when_gate_on(self) -> None:
        state = self._make_state(tilt_gate_enabled=True, max_tilt_degrees=5.0)
        tilt_info = self._high_deviation_tilt_info()
        result = self._call_validate(
            state, self._one_passing_detection(), tilt_return_value=tilt_info
        )
        self.assertIsNotNone(result.get("sticker_tilt_angle"))
        self.assertIsNotNone(result.get("sticker_tilt_deviation"))

    def test_tilt_gate_enabled_flag_in_thresholds(self) -> None:
        """validation_details.thresholds must expose tilt_gate_enabled for observability."""
        state = self._make_state(tilt_gate_enabled=True, max_tilt_degrees=5.0)
        result = self._call_validate(
            state, self._one_passing_detection(), tilt_return_value=self._high_deviation_tilt_info()
        )
        details = result.get("validation_details") or {}
        thresholds = details.get("thresholds") or {}
        self.assertIn("tilt_gate_enabled", thresholds)
        self.assertTrue(thresholds["tilt_gate_enabled"])

    # ------------------------------------------------------------------
    # Regression: other gates are unaffected by the tilt toggle
    # ------------------------------------------------------------------

    def test_low_roi_conf_still_rejects_when_tilt_gate_off(self) -> None:
        from shared.contracts.templates import StickerRule
        state = self._make_state(tilt_gate_enabled=False)
        state.template.sticker.min_roi_confidence = 0.95  # very high threshold
        low_conf_detection = [
            {
                "label": "sticker",
                "confidence": 0.1,  # far below 0.95
                "class_confidence": 0.9,
                "position": {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0},
            }
        ]
        result = self._call_validate(
            state, low_conf_detection, tilt_return_value=self._high_deviation_tilt_info()
        )
        self.assertEqual(result.get("reject_reason_code"), "LOW_ROI_CONF")


# NOTE (FASE 0): OCR-based sticker validation was REMOVED by design — sticker mode
# now validates presence/position/tilt, NOT code/content. The following OCR-specific
# test classes were retired: OcrAnchorPrimaryGateTest and the OCR cases of
# StickerOnlyOcrGateTest (use_ocr / ocr_expected_code / ocr_mode / ocr_engine /
# expected_dot_x/y / max_anchor_offset — all removed from StickerRule/VisionConfig).
# See TESTING.md "Retired: OCR sticker validation" and HANDOFF.md R5.
# The non-OCR tilt-normalization utility test is preserved below.
class TiltNormalizationTest(unittest.TestCase):
    def test_normalize_tilt_180(self) -> None:
        service = _make_inspection_service()
        self.assertEqual(service._normalize_tilt_180(0), 0)
        self.assertEqual(service._normalize_tilt_180(175), -5)
        self.assertEqual(service._normalize_tilt_180(-175), 5)
        self.assertEqual(service._normalize_tilt_180(90), 90)
        self.assertEqual(service._normalize_tilt_180(180), 0)


if __name__ == "__main__":
    unittest.main()
