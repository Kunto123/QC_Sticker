"""Shared inference backend helpers.

Plan #13 — Full Abstraction Backend Inference.

This module provides:
- InferenceBackend abstract base class
- Concrete backend subclasses: TFLiteBackend, OpenVINOBackend, ONNXBackend, UltralyticsBackend
- Shared helper functions for YOLO output parsing, NMS, and label mapping
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from abc import ABC, abstractmethod

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "InferenceBackend",
    "TFLiteBackend",
    "OpenVINOBackend",
    "ONNXBackend",
    "UltralyticsBackend",
    "_parse_yolo_output",
    "_apply_nms",
    "_apply_names_map",
]


# ──────────────────────────────────────────────────────────────────
# Shared helpers (used by all non-Ultralytics backends)
# ──────────────────────────────────────────────────────────────────

def _parse_yolo_output(
    out: np.ndarray,
    conf_threshold: float,
    pad_left: int,
    pad_top: int,
    w_in: int,
    h_in: int,
    has_objectness: bool = False,
) -> list[tuple[float, int, float, float, float, float]]:
    """Parse YOLOv11 output array into raw candidate detections.

    Handles both [N, 6] and [6, N] layouts (auto-transposes if needed).

    Args:
        has_objectness: If True, row format is [cx, cy, w, h, obj_conf, cls0, cls1, ...]
                       (YOLOv5 raw). If False, row format is [cx, cy, w, h, cls0, cls1, ...]
                       (YOLOv8/11). Default False.
    """
    candidates: list[tuple[float, int, float, float, float, float]] = []
    if out is None or out.size == 0:
        return candidates

    if out.ndim == 2:
        if out.shape[0] < out.shape[1]:
            out = out.T
        w_content = w_in - 2 * pad_left
        h_content = h_in - 2 * pad_top
        class_offset = 5 if has_objectness else 4
        for row in out:
            class_scores = row[class_offset:]
            base_conf = float(np.max(class_scores))
            # YOLOv5 raw: final confidence = objectness × class score
            conf = float(row[4]) * base_conf if has_objectness else base_conf
            if conf < conf_threshold:
                continue
            class_id = int(np.argmax(class_scores))
            xc, yc, w_b, h_b = (
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
            )
            x1_padded = (xc - w_b / 2) * w_in
            y1_padded = (yc - h_b / 2) * h_in
            x2_padded = (xc + w_b / 2) * w_in
            y2_padded = (yc + h_b / 2) * h_in
            x1 = max(0.0, (x1_padded - pad_left) / w_content) if w_content > 0 else 0.0
            y1 = max(0.0, (y1_padded - pad_top) / h_content) if h_content > 0 else 0.0
            x2 = min(1.0, (x2_padded - pad_left) / w_content) if w_content > 0 else 1.0
            y2 = min(1.0, (y2_padded - pad_top) / h_content) if h_content > 0 else 1.0
            candidates.append((conf, class_id, x1, y1, x2, y2))
    return candidates


def _apply_nms(
    candidates: list[tuple[float, int, float, float, float, float]],
    conf_threshold: float,
) -> list[dict[str, Any]]:
    """Apply Non-Maximum Suppression to candidate detections.

    Returns list of detection dicts with normalized position coords.
    """
    if not candidates:
        return []
    roi_h, roi_w = 1.0, 1.0  # will be scaled by caller
    boxes_cv = [[c[2], c[3], c[4] - c[2], c[5] - c[3]] for c in candidates]
    scores_cv = [c[0] for c in candidates]
    indices = cv2.dnn.NMSBoxes(boxes_cv, scores_cv, conf_threshold, 0.45)
    detections: list[dict[str, Any]] = []
    if len(indices) > 0:
        for idx in indices.flatten():
            conf, class_id, x1, y1, x2, y2 = candidates[int(idx)]
            detections.append(
                {
                    "label": str(class_id),
                    "confidence": round(conf, 4),
                    "class_confidence": round(conf, 4),
                    "class_id": class_id,
                    "position": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },
                    "bbox": [x1, y1, x2, y2],
                }
            )
    return detections


def _apply_names_map(
    detections: list[dict[str, Any]],
    names_map: dict[int, str],
    allowed_labels: set[str] | None,
) -> list[dict[str, Any]]:
    """Map class_id → label name and filter by allowed_labels.

    When names_map is empty (no meta file), labels stay as string class_ids
    and label-based filtering is skipped (returns all detections).
    """
    filtered: list[dict[str, Any]] = []
    for det in detections:
        label = names_map.get(det["class_id"], str(det["class_id"]))
        det["label"] = label
        # Only filter by label when class mapping is available
        if names_map and allowed_labels is not None:
            if label.strip().lower() not in allowed_labels:
                continue
        filtered.append(det)
    return filtered


# ──────────────────────────────────────────────────────────────────
# Abstract base class
# ──────────────────────────────────────────────────────────────────

class InferenceBackend(ABC):
    """Abstract interface for sticker detection inference backends."""

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        vision: Any,
        expected_class: str | None = None,
    ) -> dict[str, Any]:
        """Run inference on an image and return detection results.

        Returns dict with keys:
            backend, mode, model_path, meta_path, class_names, detections,
            raw_detection_count, allowed_labels_filter, fallback_reason,
            device_mode, effective_device, device_backend,
            device_fallback_reason, gpu_available
        """
        ...


# ──────────────────────────────────────────────────────────────────
# TFLite Backend
# ──────────────────────────────────────────────────────────────────

class TFLiteBackend(InferenceBackend):
    """TFLite inference backend (CPU-friendly, via tflite_runtime / ai_edge_litert)."""

    def __init__(
        self,
        config: Any,
        loaded_models: dict[str, Any],
        meta_cache: dict[str, tuple[dict, float]],
        runtime_lock: threading.RLock,
    ) -> None:
        self._config = config
        self._loaded_models = loaded_models
        self._meta_cache = meta_cache
        self._runtime_lock = runtime_lock

    def _resolve_model_path(self, vision: Any) -> str:
        direct = str(vision.model_path or "").strip()
        if direct:
            return direct
        return str(self._config.default_sticker_model_path or "").strip()

    def _resolve_meta_path(self, vision: Any) -> str:
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

    def _load_tflite_interpreter(self, model_path: str, num_threads: int = 4):
        """Load TFLite model via tflite_runtime (lightweight, no tensorflow needed)."""
        resolved = str(Path(model_path).resolve())
        # Cache key includes thread count so different configs get separate instances
        cache_key = f"{resolved}::t{num_threads}"
        with self._runtime_lock:
            # Check cache
            interp = self._loaded_models.get(cache_key)
            if interp is not None:
                return interp
            # Load model inside lock to prevent duplicate loads
            try:
                from ai_edge_litert.interpreter import Interpreter  # type: ignore
                interpreter = Interpreter(model_path=resolved, num_threads=num_threads)
            except (ImportError, TypeError):
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=".*tf.lite.Interpreter.*",
                            category=UserWarning,
                        )
                        from tflite_runtime.interpreter import Interpreter  # type: ignore
                        try:
                            interpreter = Interpreter(model_path=resolved, num_threads=num_threads)
                        except TypeError:
                            interpreter = Interpreter(model_path=resolved)
                except ImportError:
                    try:
                        import tensorflow as tf  # type: ignore
                        Interpreter = tf.lite.Interpreter
                        try:
                            interpreter = Interpreter(model_path=resolved, num_threads=num_threads)
                        except TypeError:
                            interpreter = Interpreter(model_path=resolved)
                    except ImportError:
                        raise ModuleNotFoundError(
                            "TFLite runtime not found. Install one of: "
                            "pip install ai-edge-litert | pip install tflite-runtime | pip install tensorflow"
                        )
            interpreter.allocate_tensors()
            self._loaded_models[cache_key] = interpreter
            return interpreter

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

    def predict(self, image: np.ndarray, vision: Any, expected_class: str | None = None) -> dict[str, Any]:
        """Run inference via TFLite interpreter (CPU-friendly)."""
        import time as _time
        t0 = _time.perf_counter()

        model_path = self._resolve_model_path(vision)
        if not model_path:
            raise FileNotFoundError("Sticker model path is not configured.")
        resolved_model_path = Path(model_path)
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {resolved_model_path}")

        logger.debug("[tflite] loading model: %s", resolved_model_path)
        _num_threads = getattr(self._config, "inference_num_threads", 4)
        interpreter = self._load_tflite_interpreter(str(resolved_model_path), num_threads=_num_threads)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]["shape"]
        input_dtype = input_details[0]["dtype"]
        logger.debug("[tflite] input_shape=%s dtype=%s", input_shape, input_dtype)

        # Preprocess
        h_in, w_in = int(input_shape[1]), int(input_shape[2])
        img_padded, _scale, _pad_left, _pad_top = self._letterbox(image, (h_in, w_in))
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        if np.issubdtype(input_dtype, np.floating):
            img_normalized = (img_rgb.astype(np.float32) / 255.0).astype(input_dtype)
        else:
            img_normalized = img_rgb.astype(input_dtype)
        input_data = np.expand_dims(img_normalized, axis=0)
        t1 = _time.perf_counter()
        logger.debug("[tflite] preprocess=%.1fms", (t1 - t0) * 1000)

        # Run inference
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
        t2 = _time.perf_counter()
        logger.debug("[tflite] invoke=%.1fms", (t2 - t1) * 1000)

        # Parse output — use shared helper
        candidates = []
        raw_box_count = 0
        roi_h, roi_w = image.shape[:2]
        if output_data is not None and output_data.size > 0:
            raw_out = output_data[0]
            logger.debug("[tflite] output_shape=%s", raw_out.shape)
            if raw_out.ndim == 2:
                if raw_out.shape[0] < raw_out.shape[1]:
                    raw_out = raw_out.T
                raw_box_count = int(raw_out.shape[0])
                candidates = _parse_yolo_output(
                    raw_out,
                    float(vision.conf_threshold),
                    _pad_left,
                    _pad_top,
                    w_in,
                    h_in,
                )

        detections = _apply_nms(candidates, float(vision.conf_threshold))
        # Scale normalized coords to pixel space
        for det in detections:
            px = det["position"]
            px["x1"] = round(px["x1"] * roi_w, 2)
            px["y1"] = round(px["y1"] * roi_h, 2)
            px["x2"] = round(px["x2"] * roi_w, 2)
            px["y2"] = round(px["y2"] * roi_h, 2)
            det["bbox"] = [px["x1"], px["y1"], px["x2"], px["y2"]]

        t3 = _time.perf_counter()
        logger.debug("[tflite] total=%.1fms parse=%.1fms raw=%d filtered=%d",
                     (t3 - t0) * 1000, (t3 - t2) * 1000, len(candidates), len(detections))

        # Load class names and filter — use shared helper
        meta = self._load_meta(self._resolve_meta_path(vision))
        class_names = meta.get("class_names", [])
        names_map = {i: name for i, name in enumerate(class_names)}

        if not names_map:
            logger.warning(
                "[inference] class_names kosong — label akan tampil sebagai angka. "
                "Pastikan file JSON dengan key 'class_names' ada di folder model."
            )

        allowed_label_values = [str(label) for label in (vision.classes or []) if str(label).strip()]
        if expected_class and str(expected_class).strip():
            allowed_label_values.append(str(expected_class).strip())
        if allowed_label_values:
            for label in (getattr(vision, "text_anchor_class", ""), getattr(vision, "center_dot_class", "")):
                if str(label or "").strip():
                    allowed_label_values.append(str(label))
        allowed_labels = {label.strip().lower() for label in allowed_label_values} or None

        filtered = _apply_names_map(detections, names_map, allowed_labels)

        return {
            "backend": "tflite",
            "mode": "tflite",
            "model_path": str(resolved_model_path),
            "meta_path": self._resolve_meta_path(vision) or None,
            "class_names": class_names,
            "detections": filtered,
            "raw_detection_count": raw_box_count,
            "allowed_labels_filter": sorted(allowed_labels) if allowed_labels is not None else None,
            "fallback_reason": None,
            "device_mode": "cpu",
            "effective_device": "cpu",
            "device_backend": "tflite",
            "device_fallback_reason": None,
            "gpu_available": False,
        }


# ──────────────────────────────────────────────────────────────────
# OpenVINO Backend
# ──────────────────────────────────────────────────────────────────

class OpenVINOBackend(InferenceBackend):
    """OpenVINO IR inference backend (optimized for Intel CPU/iGPU)."""

    def __init__(
        self,
        config: Any,
        loaded_models: dict[str, Any],
        meta_cache: dict[str, tuple[dict, float]],
        runtime_lock: threading.RLock,
    ) -> None:
        self._config = config
        self._loaded_models = loaded_models
        self._meta_cache = meta_cache
        self._runtime_lock = runtime_lock

    def _resolve_model_path(self, vision: Any) -> str:
        direct = str(vision.model_path or "").strip()
        if direct:
            return direct
        return str(self._config.default_sticker_model_path or "").strip()

    def _resolve_meta_path(self, vision: Any) -> str:
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

    def _load_openvino_model(self, model_path: str):
        """Load OpenVINO IR model (.xml) — cached after first load."""
        resolved = str(Path(model_path).resolve())
        with self._runtime_lock:
            # Check cache
            model = self._loaded_models.get(resolved)
            if model is not None:
                return model
            # Load model inside lock to prevent duplicate compilations
            try:
                from openvino import Core  # type: ignore  # OpenVINO 2024.x+
            except ImportError:
                try:
                    from openvino.runtime import Core  # type: ignore  # Legacy 2022.x–2023.x
                except ImportError:
                    raise ModuleNotFoundError("openvino required: pip install openvino")
            ie = Core()
            ov_model = ie.read_model(model=resolved)
            _n = getattr(self._config, "inference_num_threads", 4)
            ie.set_property("CPU", {"INFERENCE_NUM_THREADS": str(_n)})
            compiled = ie.compile_model(model=ov_model, device_name="CPU")
            self._loaded_models[resolved] = compiled
            logger.debug("[openvino] model loaded: %s", resolved)
            return compiled

    @staticmethod
    def _letterbox(
        image: np.ndarray,
        target_size: tuple[int, int],
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> tuple[np.ndarray, float, int, int]:
        """Resize image preserving aspect ratio, pad remainder with gray.
        Returns: (padded_image, scale, pad_left, pad_top)
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

    def predict(self, image: np.ndarray, vision: Any, expected_class: str | None = None) -> dict[str, Any]:
        """Run inference via OpenVINO IR (optimized for Intel CPU/iGPU)."""
        import time as _time
        t0 = _time.perf_counter()

        model_path = self._resolve_model_path(vision)
        if not model_path:
            raise FileNotFoundError("Sticker model path is not configured.")
        resolved_model_path = Path(model_path)
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"OpenVINO model not found: {resolved_model_path}")

        compiled = self._load_openvino_model(str(resolved_model_path))
        input_layer = compiled.input(0)
        input_shape = tuple(input_layer.shape)  # e.g. (1, 3, 640, 640) NCHW or (1, 640, 640, 3) NHWC

        # Determine layout: NCHW if dim[1] in {1,3}, else assume NHWC
        if len(input_shape) == 4 and input_shape[1] in (1, 3):
            _, _c, h_in, w_in = input_shape
            nchw = True
        else:
            _, h_in, w_in, _c = input_shape
            nchw = False

        # Letterbox preprocessing (same as TFLite)
        img_padded, _scale, _pad_left, _pad_top = self._letterbox(image, (int(h_in), int(w_in)))
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        if nchw:
            input_data = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]
        else:
            input_data = img_norm[np.newaxis, ...]
        t1 = _time.perf_counter()

        # Inference
        results = compiled([input_data])
        out = list(results.values())[0]
        t2 = _time.perf_counter()

        # Parse output — same YOLO format as TFLite/ONNX [1, 6, 8400] — use shared helper
        candidates = []
        raw_box_count = 0
        roi_h, roi_w = image.shape[:2]
        if out is not None and out.size > 0:
            raw_out = out[0]
            if raw_out.ndim == 2:
                if raw_out.shape[0] < raw_out.shape[1]:
                    raw_out = raw_out.T
                raw_box_count = int(raw_out.shape[0])
                candidates = _parse_yolo_output(
                    raw_out,
                    float(vision.conf_threshold),
                    _pad_left,
                    _pad_top,
                    int(w_in),
                    int(h_in),
                )

        detections = _apply_nms(candidates, float(vision.conf_threshold))
        # Scale normalized coords to pixel space
        for det in detections:
            px = det["position"]
            px["x1"] = round(px["x1"] * roi_w, 2)
            px["y1"] = round(px["y1"] * roi_h, 2)
            px["x2"] = round(px["x2"] * roi_w, 2)
            px["y2"] = round(px["y2"] * roi_h, 2)
            det["bbox"] = [px["x1"], px["y1"], px["x2"], px["y2"]]

        t3 = _time.perf_counter()
        logger.debug("[openvino] total=%.1fms parse=%.1fms raw=%d filtered=%d",
                    (t3 - t0) * 1000, (t3 - t2) * 1000, len(candidates), len(detections))

        # Class name mapping (same as TFLite) — use shared helper
        meta = self._load_meta(self._resolve_meta_path(vision))
        class_names = meta.get("class_names", [])
        names_map = {i: name for i, name in enumerate(class_names)}
        if not names_map:
            logger.warning(
                "[openvino] class_names kosong — label akan tampil sebagai angka. "
                "Pastikan file JSON dengan key 'class_names' ada di folder model."
            )
        allowed_label_values = [str(label) for label in (vision.classes or []) if str(label).strip()]
        if expected_class and str(expected_class).strip():
            allowed_label_values.append(str(expected_class).strip())
        if allowed_label_values:
            for label in (getattr(vision, "text_anchor_class", ""), getattr(vision, "center_dot_class", "")):
                if str(label or "").strip():
                    allowed_label_values.append(str(label))
        allowed_labels = {label.strip().lower() for label in allowed_label_values} or None
        filtered = _apply_names_map(detections, names_map, allowed_labels)

        return {
            "backend": "openvino",
            "mode": "openvino",
            "model_path": str(resolved_model_path),
            "meta_path": self._resolve_meta_path(vision) or None,
            "class_names": class_names,
            "detections": filtered,
            "raw_detection_count": raw_box_count,
            "allowed_labels_filter": sorted(allowed_labels) if allowed_labels is not None else None,
            "fallback_reason": None,
            "device_mode": "cpu",
            "effective_device": "cpu",
            "device_backend": "openvino",
            "device_fallback_reason": None,
            "gpu_available": False,
        }


# ──────────────────────────────────────────────────────────────────
# ONNX Backend
# ──────────────────────────────────────────────────────────────────

class ONNXBackend(InferenceBackend):
    """ONNX Runtime inference backend (CPU-friendly, no GPU needed)."""

    def __init__(
        self,
        config: Any,
        loaded_models: dict[str, Any],
        meta_cache: dict[str, tuple[dict, float]],
        runtime_lock: threading.RLock,
    ) -> None:
        self._config = config
        self._loaded_models = loaded_models
        self._meta_cache = meta_cache
        self._runtime_lock = runtime_lock

    def _resolve_model_path(self, vision: Any) -> str:
        direct = str(vision.model_path or "").strip()
        if direct:
            return direct
        return str(self._config.default_sticker_model_path or "").strip()

    def _resolve_meta_path(self, vision: Any) -> str:
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

    def _load_onnx_session(self, model_path: str):
        """Load ONNX model via onnxruntime (CPU-friendly). Thread-safe."""
        resolved = str(Path(model_path).resolve())
        with self._runtime_lock:
            # Check cache
            sess = self._loaded_models.get(resolved)
            if sess is not None:
                return sess
            # Load model inside lock to prevent duplicate sessions
            try:
                import onnxruntime as ort  # type: ignore
            except ImportError:
                raise ModuleNotFoundError(
                    "onnxruntime is required for ONNX inference. "
                    "Install it with: pip install onnxruntime"
                )
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _n = getattr(self._config, "inference_num_threads", 4)
            opts.intra_op_num_threads = _n   # threads within one op (matmul, conv)
            opts.inter_op_num_threads = 1    # parallel between ops — 1 sufficient for small models
            sess = ort.InferenceSession(resolved, sess_options=opts, providers=["CPUExecutionProvider"])
            self._loaded_models[resolved] = sess
            return sess

    @staticmethod
    def _letterbox(
        image: np.ndarray,
        target_size: tuple[int, int],
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> tuple[np.ndarray, float, int, int]:
        """Resize image preserving aspect ratio, pad remainder with gray.
        Returns: (padded_image, scale, pad_left, pad_top)
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

    def predict(self, image: np.ndarray, vision: Any, expected_class: str | None = None) -> dict[str, Any]:
        """Run inference via ONNX Runtime (CPU-friendly, no GPU needed)."""
        import time as _time
        t0 = _time.perf_counter()

        model_path = self._resolve_model_path(vision)
        if not model_path:
            raise FileNotFoundError("Sticker model path is not configured.")
        resolved_model_path = Path(model_path)
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {resolved_model_path}")

        logger.debug("[onnx] loading model: %s", resolved_model_path)
        sess = self._load_onnx_session(str(resolved_model_path))
        input_name = sess.get_inputs()[0].name
        logger.debug("[onnx] input_name=%s", input_name)

        # Preprocess: letterbox → BGR→RGB → normalize /255 → float32
        # Model expects NHWC [1, 640, 640, 3] (TFLite-origin ONNX, channel-last)
        _onnx_pad, _onnx_scale, _onnx_pad_left, _onnx_pad_top = self._letterbox(image, (640, 640))
        img_rgb = cv2.cvtColor(_onnx_pad, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        input_data = img_norm[np.newaxis, ...]  # [1, 640, 640, 3]
        t1 = _time.perf_counter()
        logger.debug("[onnx] preprocess=%.1fms", (t1 - t0) * 1000)

        # Inference
        out = sess.run(None, {input_name: input_data})[0]  # [1, 6, 8400] or similar
        t2 = _time.perf_counter()
        logger.debug("[onnx] invoke=%.1ffms", (t2 - t1) * 1000)

        # Parse output — YOLOv11 format: rows are [cx, cy, w, h, cls0_score, cls1_score, ...]
        # No separate objectness score; class_probs ARE the confidence.
        # Use shared helper.
        out_raw = out[0]  # → [6, 8400] or [8400, 6]
        if out_raw.ndim == 2 and out_raw.shape[0] < out_raw.shape[1]:
            out_raw = out_raw.T  # → [N, C]

        candidates = _parse_yolo_output(
            out_raw,
            float(vision.conf_threshold),
            _onnx_pad_left,
            _onnx_pad_top,
            640,
            640,
        )

        detections = _apply_nms(candidates, float(vision.conf_threshold))
        # Scale normalized coords to pixel space
        roi_h, roi_w = image.shape[:2]
        for det in detections:
            px = det["position"]
            px["x1"] = round(px["x1"] * roi_w, 2)
            px["y1"] = round(px["y1"] * roi_h, 2)
            px["x2"] = round(px["x2"] * roi_w, 2)
            px["y2"] = round(px["y2"] * roi_h, 2)
            det["bbox"] = [px["x1"], px["y1"], px["x2"], px["y2"]]

        t3 = _time.perf_counter()
        logger.debug("[onnx] total=%.1fms parse=%.1fms raw=%d filtered=%d",
                     (t3 - t0) * 1000, (t3 - t2) * 1000, len(candidates), len(detections))

        # Load class names — use shared helper
        meta = self._load_meta(self._resolve_meta_path(vision))
        class_names = meta.get("class_names", [])
        names_map = {i: name for i, name in enumerate(class_names)}
        if not names_map:
            logger.warning(
                "[onnx] class_names kosong — label akan tampil sebagai angka. "
                "Pastikan file JSON dengan key 'class_names' ada di folder model."
            )

        allowed_label_values = [str(label) for label in (vision.classes or []) if str(label).strip()]
        if expected_class and str(expected_class).strip():
            allowed_label_values.append(str(expected_class).strip())
        if allowed_label_values:
            for label in (getattr(vision, "text_anchor_class", ""), getattr(vision, "center_dot_class", "")):
                if str(label or "").strip():
                    allowed_label_values.append(str(label))
        allowed_labels = {label.strip().lower() for label in allowed_label_values} or None

        filtered = _apply_names_map(detections, names_map, allowed_labels)

        return {
            "backend": "onnx",
            "mode": "onnx",
            "model_path": str(resolved_model_path),
            "meta_path": self._resolve_meta_path(vision) or None,
            "class_names": class_names,
            "detections": filtered,
            "raw_detection_count": len(candidates),
            "allowed_labels_filter": sorted(allowed_labels) if allowed_labels is not None else None,
            "fallback_reason": None,
            "device_mode": "cpu",
            "effective_device": "cpu",
            "device_backend": "onnx",
            "device_fallback_reason": None,
            "gpu_available": False,
        }


# ──────────────────────────────────────────────────────────────────
# Ultralytics Backend
# ──────────────────────────────────────────────────────────────────

class UltralyticsBackend(InferenceBackend):
    """Ultralytics YOLO inference backend (supports GPU via device resolution)."""

    def __init__(
        self,
        config: Any,
        loaded_models: dict[str, Any],
        meta_cache: dict[str, tuple[dict, float]],
        runtime_lock: threading.RLock,
        device_resolution: Any,
    ) -> None:
        self._config = config
        self._loaded_models = loaded_models
        self._meta_cache = meta_cache
        self._runtime_lock = runtime_lock
        self._device_resolution = device_resolution

    def _resolve_model_path(self, vision: Any) -> str:
        direct = str(vision.model_path or "").strip()
        if direct:
            return direct
        return str(self._config.default_sticker_model_path or "").strip()

    def _resolve_meta_path(self, vision: Any) -> str:
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

    def _load_yolo_class(self):
        from ultralytics import YOLO  # type: ignore
        return YOLO

    def _get_ultralytics_model(self, model_path: str):
        resolved = str(Path(model_path).resolve())
        with self._runtime_lock:
            model = self._loaded_models.get(resolved)
            if model is not None:
                return model
            YOLO = self._load_yolo_class()
            model = YOLO(resolved, task="detect")
            self._loaded_models[resolved] = model
            return model

    @staticmethod
    def _normalize_label_key(value: Any) -> str:
        text = str(value or "").strip().lower()
        return "".join(ch for ch in text if ch.isalnum())

    def _normalize_detections(
        self,
        *,
        result: Any,
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
                    "bbox": [xyxy[0], xyxy[1], xyxy[2], xyxy[3]],
                }
            )
        return detections

    def predict(self, image: np.ndarray, vision: Any, expected_class: str | None = None) -> dict[str, Any]:
        model_path = self._resolve_model_path(vision)
        if not model_path:
            raise FileNotFoundError("Sticker model path is not configured.")
        resolved_model_path = Path(model_path)
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"Sticker model not found: {resolved_model_path}")

        model = self._get_ultralytics_model(str(resolved_model_path))
        kwargs: dict[str, Any] = {
            "verbose": False,
            "conf": float(vision.conf_threshold),
            "device": self._device_resolution.effective_device,
        }
        if int(vision.imgsz or 0) > 0:
            kwargs["imgsz"] = int(vision.imgsz)
        result = model.predict(image, **kwargs)[0]
        names = result.names or {}
        raw_box_count = int(len(result.boxes)) if result.boxes is not None else 0
        # Build allowed labels from vision.classes + anchor classes + expected_class
        allowed_label_values = [str(label) for label in (vision.classes or []) if str(label).strip()]
        # Always include expected_class (from template sticker.expected_class)
        if expected_class and str(expected_class).strip():
            allowed_label_values.append(str(expected_class).strip())
        if allowed_label_values:
            for label in (getattr(vision, "text_anchor_class", ""), getattr(vision, "center_dot_class", "")):
                if str(label or "").strip():
                    allowed_label_values.append(str(label))
        # Normalize to lowercase for case-insensitive matching
        allowed_labels = {label.strip().lower() for label in allowed_label_values} or None
        allowed_label_keys = ({self._normalize_label_key(label) for label in allowed_label_values if self._normalize_label_key(label)} or None)
        detections = self._normalize_detections(
            result=result,
            names=names,
            allowed_labels=allowed_labels,
            allowed_label_keys=allowed_label_keys,
        )
        return {
            "backend": "ultralytics",
            "mode": "ultralytics",
            "model_path": str(resolved_model_path),
            "meta_path": self._resolve_meta_path(vision) or None,
            "class_names": list(self._load_meta(self._resolve_meta_path(vision)).get("class_names") or []),
            "detections": detections,
            "raw_detection_count": raw_box_count,
            "allowed_labels_filter": sorted(allowed_labels) if allowed_labels is not None else None,
            "fallback_reason": None,
            "device_mode": self._device_resolution.requested_mode,
            "effective_device": self._device_resolution.effective_device,
            "device_backend": self._device_resolution.backend,
            "device_fallback_reason": self._device_resolution.fallback_reason,
            "gpu_available": self._device_resolution.gpu_available,
        }
