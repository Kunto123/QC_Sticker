from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from backend.app.core.config import AppConfig
from backend.app.core.device_runtime import DeviceResolution, DeviceRuntimeResolver
from backend.app.repositories.models_repository import ModelsRepository
from backend.app.services.inference_backend import (
    InferenceBackend,
    ONNXBackend,
    OpenVINOBackend,
    TFLiteBackend,
    UltralyticsBackend,
)
from shared.contracts.templates import VisionConfig


class StickerInferenceService:
    def __init__(
        self,
        app_config: AppConfig,
        models_repo: ModelsRepository,
        device_runtime: DeviceRuntimeResolver | None = None,
    ) -> None:
        self._config = app_config
        self._models_repo = models_repo
        self._device_runtime = device_runtime or DeviceRuntimeResolver(app_config)
        self._runtime_lock = threading.RLock()
        self._loaded_models: dict[str, Any] = {}
        self._meta_cache: dict[str, tuple[dict, float]] = {}
        self._backend_cache: dict[str, InferenceBackend] = {}

    def unload_model(self, model_path: str) -> int:
        """Drop a model (and its meta) from the runtime caches so its files can be
        deleted — OpenVINO keeps the .bin memory-mapped while compiled."""
        import gc

        resolved = str(Path(model_path).resolve())
        folder = str(Path(resolved).parent)
        with self._runtime_lock:
            model_keys = [k for k in self._loaded_models if k == resolved or k.startswith(resolved + "::")]
            for key in model_keys:
                self._loaded_models.pop(key, None)
            for key in [k for k in self._meta_cache if str(Path(k).parent) == folder]:
                self._meta_cache.pop(key, None)
        gc.collect()
        return len(model_keys)

    def _resolve_mode(self) -> str:
        mode = str(self._config.sticker_inference_mode or "auto").strip().lower()
        return mode if mode in {"auto", "ultralytics", "classic", "tflite", "onnx", "openvino"} else "auto"

    def _resolve_model_path(self, vision: VisionConfig) -> str:
        direct = str(vision.model_path or "").strip()
        if direct:
            return direct
        return str(self._config.default_sticker_model_path or "").strip()

    def _resolve_meta_path(self, vision: VisionConfig) -> str:
        model_path = self._resolve_model_path(vision)
        if model_path:
            p = Path(model_path)
            for candidate in (p.parent / (p.stem + ".meta.json"), p.with_suffix(".json")):
                logger.debug("[inference] auto-discover meta: %s (exists=%s)", candidate, candidate.exists())
                if candidate.exists():
                    logger.info("[inference] meta found: %s", candidate)
                    return str(candidate)
        logger.info("[inference] meta not found for model: %s", model_path)
        return ""

    def _load_meta(self, meta_path: str) -> dict[str, Any]:
        if not meta_path:
            return {}
        now = time.monotonic()
        with self._runtime_lock:
            cached = self._meta_cache.get(meta_path)
            if cached is not None:
                payload, loaded_at = cached
                if now - loaded_at < 30.0:  # TTL 30 detik
                    return payload
            path = Path(meta_path)
            if not path.exists():
                logger.warning("[inference] meta file not found: %s", meta_path)
                self._meta_cache.pop(meta_path, None)  # evict stale entry
                return {}
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[inference] meta file parse error %s: %s", meta_path, exc)
                payload = {}
            self._meta_cache[meta_path] = (payload, now)
            return payload

    def _resolve_device(self) -> DeviceResolution:
        return self._device_runtime.resolve()

    @staticmethod
    def _letterbox(
        image: np.ndarray,
        target_size: tuple[int, int],
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> tuple[np.ndarray, float, int, int]:
        """Resize image preserving aspect ratio, pad remainder with gray.
        Returns: (padded_image, scale, pad_left, pad_top)
        scale: factor applied to original image to fit in target_size
        """
        h_orig, w_orig = image.shape[:2]
        h_tgt, w_tgt = target_size
        scale = min(w_tgt / w_orig, h_tgt / h_orig)
        w_new = int(round(w_orig * scale))
        h_new = int(round(h_orig * scale))
        img_scaled = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((h_tgt, w_tgt, 3), color, dtype=np.uint8)
        pad_top = (h_tgt - h_new) // 2
        pad_left = (w_tgt - w_new) // 2
        canvas[pad_top:pad_top + h_new, pad_left:pad_left + w_new] = img_scaled
        return canvas, scale, pad_left, pad_top

    def _get_backend(self, mode: str) -> InferenceBackend:
        """Create and cache backend instances (thread-safe)."""
        with self._runtime_lock:
            if mode in self._backend_cache:
                return self._backend_cache[mode]

            backend: InferenceBackend
            if mode == "tflite":
                backend = TFLiteBackend(
                    config=self._config,
                    loaded_models=self._loaded_models,
                    meta_cache=self._meta_cache,
                    runtime_lock=self._runtime_lock,
                )
            elif mode == "openvino":
                backend = OpenVINOBackend(
                    config=self._config,
                    loaded_models=self._loaded_models,
                    meta_cache=self._meta_cache,
                    runtime_lock=self._runtime_lock,
                )
            elif mode == "onnx":
                backend = ONNXBackend(
                    config=self._config,
                    loaded_models=self._loaded_models,
                    meta_cache=self._meta_cache,
                    runtime_lock=self._runtime_lock,
                )
            elif mode == "ultralytics":
                device_resolution = self._resolve_device()
                backend = UltralyticsBackend(
                    config=self._config,
                    loaded_models=self._loaded_models,
                    meta_cache=self._meta_cache,
                    runtime_lock=self._runtime_lock,
                    device_resolution=device_resolution,
                )
            else:
                raise ValueError(f"Unknown backend mode: {mode}")

            self._backend_cache[mode] = backend
            return backend

    def _normalize_detections(
        self,
        *,
        result,
        names: dict[int, str] | dict[Any, Any],
        allowed_labels: set[str] | None,
        allowed_label_keys: set[str] | None,
    ) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            xyxy = [float(value) for value in box.xyxy[0].tolist()]
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = str(names.get(class_id, class_id))
            if allowed_labels is not None:
                normalized_label = label.strip().lower()
                normalized_key = self._normalize_label_key(label)
                if normalized_label not in allowed_labels and (
                    allowed_label_keys is None or normalized_key not in allowed_label_keys
                ):
                    continue
            detections.append(
                {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "class_confidence": round(confidence, 4),
                    "class_id": class_id,
                    "position": {
                        "x1": xyxy[0],
                        "y1": xyxy[1],
                        "x2": xyxy[2],
                        "y2": xyxy[3],
                    },
                }
            )
        return detections

    @staticmethod
    def _normalize_label_key(value: Any) -> str:
        text = str(value or "").strip().lower()
        return "".join(ch for ch in text if ch.isalnum())

    @staticmethod
    def _round_bbox(position: dict[str, Any] | None) -> dict[str, float] | None:
        if not position:
            return None
        return {
            "x1": round(float(position.get("x1", 0.0)), 2),
            "y1": round(float(position.get("y1", 0.0)), 2),
            "x2": round(float(position.get("x2", 0.0)), 2),
            "y2": round(float(position.get("y2", 0.0)), 2),
        }

    def _predict_classic(self, image, vision: VisionConfig, expected_class: str | None) -> dict[str, Any]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                "backend": "classic",
                "mode": "classic",
                "model_path": self._resolve_model_path(vision) or None,
                "meta_path": self._resolve_meta_path(vision) or None,
                "class_names": list(vision.classes or []),
                "detections": [],
                "fallback_reason": None,
            }
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = float((w * h) / max(1, image.shape[0] * image.shape[1]))
        confidence = max(0.1, min(0.99, area_ratio + 0.2))
        label = str(expected_class or (vision.classes[0] if vision.classes else "sticker")).strip() or "sticker"
        return {
            "backend": "classic",
            "mode": "classic",
            "model_path": self._resolve_model_path(vision) or None,
            "meta_path": self._resolve_meta_path(vision) or None,
            "class_names": list(vision.classes or []),
            "detections": [
                {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "class_confidence": round(confidence, 4),
                    "class_id": None,
                    "position": {"x1": float(x), "y1": float(y), "x2": float(x + w), "y2": float(y + h)},
                }
            ],
            "fallback_reason": None,
        }

    def predict(
        self,
        image,
        vision: VisionConfig,
        *,
        expected_class: str | None = None,
    ) -> dict[str, Any]:
        if image is None or image.size == 0:
            return {
                "backend": "none",
                "mode": self._resolve_mode(),
                "model_path": self._resolve_model_path(vision) or None,
                "meta_path": self._resolve_meta_path(vision) or None,
                "class_names": list(vision.classes or []),
                "detections": [],
                "fallback_reason": "empty_roi",
            }

        mode = self._resolve_mode()
        if mode == "classic":
            return self._predict_classic(image, vision, expected_class)

        # Auto-detect TFLite from file extension when mode is "auto"
        if mode == "auto":
            model_path = self._resolve_model_path(vision)
            if model_path:
                _suffix = Path(model_path).suffix.lower()
                if _suffix == ".tflite":
                    mode = "tflite"
                elif _suffix == ".onnx":
                    mode = "onnx"
                elif _suffix == ".xml":
                    mode = "openvino"
                else:
                    mode = "ultralytics"
            else:
                mode = "ultralytics"

        # TFLite mode: CPU-only, no GPU device resolution needed
        if mode == "tflite":
            try:
                return self._get_backend("tflite").predict(image, vision, expected_class=expected_class)
            except (FileNotFoundError, ValueError, AttributeError) as exc:
                raise
            except Exception as exc:
                logging.warning("TFLite inference failed, fallback to classic: %s", exc)
                payload = self._predict_classic(image, vision, expected_class)
                payload["fallback_reason"] = f"tflite_error: {exc}"
                return payload

        # ONNX mode: CPU-only via onnxruntime
        if mode == "onnx":
            try:
                return self._get_backend("onnx").predict(image, vision, expected_class=expected_class)
            except (FileNotFoundError, ValueError, AttributeError) as exc:
                raise
            except Exception as exc:
                logging.warning("ONNX inference failed, fallback to classic: %s", exc)
                payload = self._predict_classic(image, vision, expected_class)
                payload["fallback_reason"] = f"onnx_error: {exc}"
                return payload

        # OpenVINO mode: Intel CPU optimized
        if mode == "openvino":
            try:
                return self._get_backend("openvino").predict(image, vision, expected_class=expected_class)
            except (FileNotFoundError, ValueError, AttributeError) as exc:
                raise
            except Exception as exc:
                logging.warning("OpenVINO inference failed, fallback to classic: %s", exc)
                payload = self._predict_classic(image, vision, expected_class)
                payload["fallback_reason"] = f"openvino_error: {exc}"
                return payload

        # Ultralytics mode (auto/ultralytics)
        try:
            return self._get_backend("ultralytics").predict(image, vision, expected_class=expected_class)
        except Exception as exc:
            if mode == "ultralytics":
                raise
            logging.warning("Sticker inference fallback to classic mode: %s", exc)
            device_resolution = self._resolve_device()
            payload = self._predict_classic(image, vision, expected_class)
            payload["fallback_reason"] = str(exc)
            payload["device_mode"] = device_resolution.requested_mode
            payload["effective_device"] = device_resolution.effective_device
            payload["device_backend"] = device_resolution.backend
            payload["device_fallback_reason"] = device_resolution.fallback_reason or str(exc)
            payload["gpu_available"] = device_resolution.gpu_available
            return payload
