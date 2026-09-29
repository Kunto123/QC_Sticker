"""Helper backend inference bersama.

Plan #13 — Full Abstraction Backend Inference.

Modul ini menyediakan:
- Base class abstrak InferenceBackend
- Subclass backend konkret: TFLiteBackend, OpenVINOBackend, ONNXBackend, UltralyticsBackend
- Fungsi helper bersama untuk parsing output YOLO, NMS, dan pemetaan label
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
# Helper bersama (dipakai semua backend non-Ultralytics)
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
    """Parse array output mentah YOLOv8/11 (atau v5) jadi kandidat deteksi.

    Menangani layout [N, C] maupun [C, N] (auto-transpose kalau perlu) dan
    kedua konvensi koordinat yang dipakai export Ultralytics:
      * head mentah OpenVINO / ONNX / .pt: cx, cy, w, h dalam **piksel input** (0..imgsz)
      * TFLite: cx, cy, w, h **ternormalisasi** ke 0..1
    Terdeteksi per tensor: kalau tidak ada koordinat box yang melebihi 1.5, tensornya dianggap ternormalisasi.

    Returns (conf, class_id, x1, y1, x2, y2) dengan x/y ternormalisasi terhadap
    *content* letterbox (padding sudah dibuang), yaitu relatif terhadap gambar ROI asli.

    Args:
        has_objectness: Kalau True, format baris [cx, cy, w, h, obj_conf, cls0, cls1, ...]
                       (YOLOv5 mentah). Kalau False, format baris [cx, cy, w, h, cls0, cls1, ...]
                       (YOLOv8/11). Default False.
    """
    candidates: list[tuple[float, int, float, float, float, float]] = []
    if out is None or out.size == 0 or out.ndim != 2:
        return candidates
    if out.shape[0] < out.shape[1]:
        out = out.T
    out = np.asarray(out, dtype=np.float32)
    class_offset = 5 if has_objectness else 4
    if out.shape[1] <= class_offset:
        return candidates

    class_scores = out[:, class_offset:]
    conf = class_scores.max(axis=1)
    if has_objectness:
        conf = out[:, 4] * conf  # YOLOv5 mentah: objectness × class score
    keep = np.flatnonzero(conf >= conf_threshold)
    if keep.size == 0:
        return candidates

    boxes = out[keep, :4]
    if float(np.abs(boxes).max()) <= 1.5:  # ternormalisasi (export TFLite)
        boxes = boxes * np.array([w_in, h_in, w_in, h_in], dtype=np.float32)
    w_content = w_in - 2 * pad_left
    h_content = h_in - 2 * pad_top
    for idx, (xc, yc, w_b, h_b) in zip(keep.tolist(), boxes.tolist()):
        x1_padded = xc - w_b / 2
        y1_padded = yc - h_b / 2
        x2_padded = xc + w_b / 2
        y2_padded = yc + h_b / 2
        x1 = min(1.0, max(0.0, (x1_padded - pad_left) / w_content)) if w_content > 0 else 0.0
        y1 = min(1.0, max(0.0, (y1_padded - pad_top) / h_content)) if h_content > 0 else 0.0
        x2 = min(1.0, max(0.0, (x2_padded - pad_left) / w_content)) if w_content > 0 else 1.0
        y2 = min(1.0, max(0.0, (y2_padded - pad_top) / h_content)) if h_content > 0 else 1.0
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append((float(conf[idx]), int(class_scores[idx].argmax()), x1, y1, x2, y2))
    return candidates


def _apply_nms(
    candidates: list[tuple[float, int, float, float, float, float]],
    conf_threshold: float,
) -> list[dict[str, Any]]:
    """Terapkan Non-Maximum Suppression ke kandidat deteksi.

    Returns list dict deteksi dengan koordinat posisi ternormalisasi.
    """
    if not candidates:
        return []
    roi_h, roi_w = 1.0, 1.0  # akan di-scale oleh pemanggil
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
    """Petakan class_id → nama label dan filter dengan allowed_labels.

    Kalau names_map kosong (tidak ada file meta), label tetap berupa
    string class_id dan filter berbasis label dilewati (return semua deteksi).
    """
    filtered: list[dict[str, Any]] = []
    for det in detections:
        label = names_map.get(det["class_id"], str(det["class_id"]))
        det["label"] = label
        # Filter berdasarkan label cuma kalau ada mapping class
        if names_map and allowed_labels is not None:
            if label.strip().lower() not in allowed_labels:
                continue
        filtered.append(det)
    return filtered


# ──────────────────────────────────────────────────────────────────
# Base class abstrak
# ──────────────────────────────────────────────────────────────────

class InferenceBackend(ABC):
    """Interface abstrak untuk backend inference deteksi sticker."""

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        vision: Any,
        expected_class: str | None = None,
    ) -> dict[str, Any]:
        """Jalankan inference pada satu gambar dan return hasil deteksi.

        Returns dict dengan key:
            backend, mode, model_path, meta_path, class_names, detections,
            raw_detection_count, allowed_labels_filter, fallback_reason,
            device_mode, effective_device, device_backend,
            device_fallback_reason, gpu_available
        """
        ...


# ──────────────────────────────────────────────────────────────────
# Backend TFLite
# ──────────────────────────────────────────────────────────────────

class TFLiteBackend(InferenceBackend):
    """Backend inference TFLite (ramah CPU, lewat tflite_runtime / ai_edge_litert)."""

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
        """Muat model TFLite lewat tflite_runtime (ringan, tidak butuh tensorflow)."""
        resolved = str(Path(model_path).resolve())
        # Cache key menyertakan jumlah thread supaya config berbeda dapat instance terpisah
        cache_key = f"{resolved}::t{num_threads}"
        with self._runtime_lock:
            # Cek cache
            interp = self._loaded_models.get(cache_key)
            if interp is not None:
                return interp
            # Muat model di dalam lock supaya tidak double-load
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
                            "TFLite runtime tidak ditemukan. Install salah satu: "
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
        """Resize gambar sambil menjaga aspect ratio, sisanya di-pad abu-abu.
        Returns: (padded_image, scale, pad_left, pad_top)
        scale: faktor yang diterapkan ke gambar asli supaya muat di target_size
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
        """Jalankan inference lewat interpreter TFLite (ramah CPU)."""
        import time as _time
        t0 = _time.perf_counter()

        model_path = self._resolve_model_path(vision)
        if not model_path:
            raise FileNotFoundError("Path model sticker belum dikonfigurasi.")
        resolved_model_path = Path(model_path)
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"Model TFLite tidak ditemukan: {resolved_model_path}")

        logger.debug("[tflite] loading model: %s", resolved_model_path)
        _num_threads = getattr(self._config, "inference_num_threads", 4)
        interpreter = self._load_tflite_interpreter(str(resolved_model_path), num_threads=_num_threads)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]["shape"]
        input_dtype = input_details[0]["dtype"]
        logger.debug("[tflite] input_shape=%s dtype=%s", input_shape, input_dtype)

        # Preprocessing
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

        # Jalankan inference
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
        t2 = _time.perf_counter()
        logger.debug("[tflite] invoke=%.1fms", (t2 - t1) * 1000)

        # Parse output — pakai helper bersama
        candidates = []
        raw_box_count = 0
        roi_h, roi_w = image.shape[:2]
        if output_data is not None and output_data.size > 0:
            raw_out = output_data[0]
            logger.debug("[tflite] output_shape=%s", raw_out.shape)
            if raw_out.ndim == 2:
                if raw_out.shape[0] < raw_out.shape[1]:
                    raw_out = raw_out.T
                candidates = _parse_yolo_output(
                    raw_out,
                    float(vision.conf_threshold),
                    _pad_left,
                    _pad_top,
                    w_in,
                    h_in,
                )
                raw_box_count = len(candidates)  # baris di atas threshold, sebelum NMS

        detections = _apply_nms(candidates, float(vision.conf_threshold))
        # Scale koordinat ternormalisasi ke ruang piksel
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

        # Muat class names dan filter — pakai helper bersama
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
# Backend OpenVINO
# ──────────────────────────────────────────────────────────────────

class OpenVINOBackend(InferenceBackend):
    """Backend inference OpenVINO IR (dioptimasi untuk CPU/iGPU Intel)."""

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
        """Muat model OpenVINO IR (.xml) — di-cache setelah load pertama."""
        resolved = str(Path(model_path).resolve())
        with self._runtime_lock:
            # Cek cache
            model = self._loaded_models.get(resolved)
            if model is not None:
                return model
            # Muat model di dalam lock supaya tidak double-compile
            try:
                from openvino import Core  # type: ignore  # OpenVINO 2024.x+
            except ImportError:
                try:
                    from openvino.runtime import Core  # type: ignore  # Legacy 2022.x–2023.x
                except ImportError:
                    raise ModuleNotFoundError("openvino wajib diinstall: pip install openvino")
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
        """Resize gambar sambil menjaga aspect ratio, sisanya di-pad abu-abu.
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
        """Jalankan inference lewat OpenVINO IR (dioptimasi untuk CPU/iGPU Intel)."""
        import time as _time
        t0 = _time.perf_counter()

        model_path = self._resolve_model_path(vision)
        if not model_path:
            raise FileNotFoundError("Path model sticker belum dikonfigurasi.")
        resolved_model_path = Path(model_path)
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"Model OpenVINO tidak ditemukan: {resolved_model_path}")

        compiled = self._load_openvino_model(str(resolved_model_path))
        input_layer = compiled.input(0)
        input_shape = tuple(input_layer.shape)  # mis. (1, 3, 640, 640) NCHW atau (1, 640, 640, 3) NHWC

        # Tentukan layout: NCHW kalau dim[1] di {1,3}, selain itu anggap NHWC
        if len(input_shape) == 4 and input_shape[1] in (1, 3):
            _, _c, h_in, w_in = input_shape
            nchw = True
        else:
            _, h_in, w_in, _c = input_shape
            nchw = False

        # Preprocessing letterbox (sama seperti TFLite)
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

        # Parse output — format YOLO sama seperti TFLite/ONNX [1, 6, 8400] — pakai helper bersama
        candidates = []
        raw_box_count = 0
        roi_h, roi_w = image.shape[:2]
        if out is not None and out.size > 0:
            raw_out = out[0]
            if raw_out.ndim == 2:
                if raw_out.shape[0] < raw_out.shape[1]:
                    raw_out = raw_out.T
                candidates = _parse_yolo_output(
                    raw_out,
                    float(vision.conf_threshold),
                    _pad_left,
                    _pad_top,
                    int(w_in),
                    int(h_in),
                )
                raw_box_count = len(candidates)  # baris di atas threshold, sebelum NMS

        detections = _apply_nms(candidates, float(vision.conf_threshold))
        # Scale koordinat ternormalisasi ke ruang piksel
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

        # Pemetaan nama class (sama seperti TFLite) — pakai helper bersama
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
# Backend ONNX
# ──────────────────────────────────────────────────────────────────

class ONNXBackend(InferenceBackend):
    """Backend inference ONNX Runtime (ramah CPU, tidak butuh GPU)."""

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
        """Muat model ONNX lewat onnxruntime (ramah CPU). Thread-safe."""
        resolved = str(Path(model_path).resolve())
        with self._runtime_lock:
            # Cek cache
            sess = self._loaded_models.get(resolved)
            if sess is not None:
                return sess
            # Muat model di dalam lock supaya tidak dobel session
            try:
                import onnxruntime as ort  # type: ignore
            except ImportError:
                raise ModuleNotFoundError(
                    "onnxruntime wajib diinstall untuk inference ONNX. "
                    "Install dengan: pip install onnxruntime"
                )
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _n = getattr(self._config, "inference_num_threads", 4)
            opts.intra_op_num_threads = _n   # thread di dalam satu op (matmul, conv)
            opts.inter_op_num_threads = 1    # paralel antar op — 1 sudah cukup untuk model kecil
            sess = ort.InferenceSession(resolved, sess_options=opts, providers=["CPUExecutionProvider"])
            self._loaded_models[resolved] = sess
            return sess

    @staticmethod
    def _letterbox(
        image: np.ndarray,
        target_size: tuple[int, int],
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> tuple[np.ndarray, float, int, int]:
        """Resize gambar sambil menjaga aspect ratio, sisanya di-pad abu-abu.
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
        """Jalankan inference lewat ONNX Runtime (ramah CPU, tidak butuh GPU)."""
        import time as _time
        t0 = _time.perf_counter()

        model_path = self._resolve_model_path(vision)
        if not model_path:
            raise FileNotFoundError("Path model sticker belum dikonfigurasi.")
        resolved_model_path = Path(model_path)
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"Model ONNX tidak ditemukan: {resolved_model_path}")

        logger.debug("[onnx] loading model: %s", resolved_model_path)
        sess = self._load_onnx_session(str(resolved_model_path))
        onnx_input = sess.get_inputs()[0]
        input_name = onnx_input.name
        # Export ONNX Ultralytics itu NCHW [1, 3, 640, 640]; ONNX asal-TFLite itu NHWC.
        # Dim dinamis balik sebagai str/None → fallback ke 640.
        raw_shape = list(onnx_input.shape or [])
        dims = [int(d) if isinstance(d, (int, np.integer)) and int(d) > 0 else None for d in raw_shape]
        nchw = len(dims) == 4 and dims[1] in (1, 3)
        if nchw:
            h_in, w_in = dims[2] or 640, dims[3] or 640
        elif len(dims) == 4:
            h_in, w_in = dims[1] or 640, dims[2] or 640
        else:
            h_in, w_in = 640, 640
        logger.debug("[onnx] input_name=%s shape=%s layout=%s", input_name, raw_shape, "NCHW" if nchw else "NHWC")

        # Preprocessing: letterbox → BGR→RGB → normalize /255 → float32
        _onnx_pad, _onnx_scale, _onnx_pad_left, _onnx_pad_top = self._letterbox(image, (h_in, w_in))
        img_rgb = cv2.cvtColor(_onnx_pad, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        if nchw:
            input_data = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]  # [1, 3, H, W]
        else:
            input_data = img_norm[np.newaxis, ...]  # [1, H, W, 3]
        t1 = _time.perf_counter()
        logger.debug("[onnx] preprocess=%.1fms", (t1 - t0) * 1000)

        # Inference
        out = sess.run(None, {input_name: input_data})[0]  # [1, 6, 8400] atau serupa
        t2 = _time.perf_counter()
        logger.debug("[onnx] invoke=%.1ffms", (t2 - t1) * 1000)

        # Parse output — format YOLOv11: baris [cx, cy, w, h, cls0_score, cls1_score, ...]
        # Tidak ada skor objectness terpisah; class_probs ADALAH confidence-nya.
        # Pakai helper bersama.
        out_raw = out[0]  # → [6, 8400] atau [8400, 6]
        if out_raw.ndim == 2 and out_raw.shape[0] < out_raw.shape[1]:
            out_raw = out_raw.T  # → [N, C]

        candidates = _parse_yolo_output(
            out_raw,
            float(vision.conf_threshold),
            _onnx_pad_left,
            _onnx_pad_top,
            w_in,
            h_in,
        )

        detections = _apply_nms(candidates, float(vision.conf_threshold))
        # Scale koordinat ternormalisasi ke ruang piksel
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

        # Muat class names — pakai helper bersama
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
# Backend Ultralytics
# ──────────────────────────────────────────────────────────────────

class UltralyticsBackend(InferenceBackend):
    """Backend inference Ultralytics YOLO (mendukung GPU lewat resolusi device)."""

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
            raise FileNotFoundError("Path model sticker belum dikonfigurasi.")
        resolved_model_path = Path(model_path)
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"Model sticker tidak ditemukan: {resolved_model_path}")

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
        # Bangun allowed labels dari vision.classes + anchor classes + expected_class
        allowed_label_values = [str(label) for label in (vision.classes or []) if str(label).strip()]
        # Selalu sertakan expected_class (dari template sticker.expected_class)
        if expected_class and str(expected_class).strip():
            allowed_label_values.append(str(expected_class).strip())
        # Normalisasi ke lowercase untuk matching case-insensitive
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
