"""test_sticker_detection_gates.py — regression tests for runtime gate observability and model naming.

Covers:
  Phase 3 — raw_detection_count and allowed_labels_filter forwarded into sticker_detection.
  (Phases 4-7 covered the training worker, removed 2026-09-18.)
  Phase 8 — inline _validate_sticker confidence/class gates.
"""
from __future__ import annotations

import unittest
import unittest.mock
from unittest.mock import MagicMock, patch

import numpy as np

from backend.app.core.config import AppConfig
from backend.app.services.inspection_session import InspectionSessionService
from backend.app.services.sticker_inference import StickerInferenceService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inspection_service() -> InspectionSessionService:
    return InspectionSessionService(
        template_runtime=MagicMock(),
        results_repo=MagicMock(),
        sticker_inference=MagicMock(),
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
# Phase 8: inline _validate_sticker gates (confidence / class / position)
# ---------------------------------------------------------------------------

class StickerValidateGateTest(unittest.TestCase):
    """_validate_sticker (inline sticker path) must apply the confidence and
    class gates and accept a clean detection. The tilt gate was removed 2026-09-18."""

    def _make_state(self):
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
            min_roi_confidence=0.0,
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

    def _call_validate(self, state, detections):
        service = _make_inspection_service()
        roi_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        return service._validate_sticker(
            roi_frame=roi_frame,
            state=state,
            detections=detections,
            detection_payload=self._detection_payload(),
            part_ready_payload=self._part_ready_payload(),
            username=None,
            user_id=None,
        )

    def test_clean_detection_is_accepted(self) -> None:
        result = self._call_validate(self._make_state(), self._one_passing_detection())
        self.assertEqual(result.get("decision"), "ACCEPT")
        self.assertIsNone(result.get("reject_reason_code"))
        self.assertNotIn("tilt", result["validation_details"])

    def test_wrong_class_rejects_wrong_type(self) -> None:
        state = self._make_state()
        state.template.sticker.expected_class = "other"
        result = self._call_validate(state, self._one_passing_detection())
        self.assertEqual(result.get("reject_reason_code"), "WRONG_TYPE")

    def test_low_roi_conf_rejects(self) -> None:
        state = self._make_state()
        state.template.sticker.min_roi_confidence = 0.95  # very high threshold
        low_conf_detection = [
            {
                "label": "sticker",
                "confidence": 0.1,  # far below 0.95
                "class_confidence": 0.9,
                "position": {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0},
            }
        ]
        result = self._call_validate(state, low_conf_detection)
        self.assertEqual(result.get("reject_reason_code"), "LOW_ROI_CONF")


# NOTE: OCR-based sticker validation was REMOVED by design — sticker mode validates
# presence/position/tilt, NOT code/content. OCR test classes were retired in FASE 0
# and the last OCR helper code (incl. _normalize_tilt_180) was deleted 2026-09-18.
# See TESTING.md "Retired: OCR sticker validation".
if __name__ == "__main__":
    unittest.main()
