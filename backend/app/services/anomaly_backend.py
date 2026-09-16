"""Anomaly detection backend for Defect mode.

Provides:
- AnomalyScorer(ABC): abstract interface
- SimpleAnomalyScorer: pixel-based fallback (no external deps)
- AnomalibScorer: wraps anomalib when available
- get_scorer(model_path): factory with caching

FASE 4: Install anomalib for production-grade anomaly detection.
  pip install anomalib
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Cache ───────────────────────────────────────────────────────────────────
_scorer_cache: dict[str, "AnomalyScorer"] = {}


class AnomalyScorer(ABC):
    """Abstract scorer: takes a crop, returns (score, heatmap)."""

    @abstractmethod
    def score(self, crop: np.ndarray) -> tuple[float, np.ndarray]:
        """Score a cropped region.

        Returns:
            score: float in [0, ∞), higher = more anomalous
            heatmap: np.ndarray (H, W) same size as crop, float32
        """
        ...

    @abstractmethod
    def calibrate(self, reference_crops: list[np.ndarray]) -> dict[str, Any]:
        """Calibrate threshold from a set of known-good crops.

        Returns dict with at least:
          - 'threshold': suggested threshold (float)
          - 'mean_score': float
          - 'std_score': float
        """
        ...


class SimpleAnomalyScorer(AnomalyScorer):
    """Simple pixel-intensity based anomaly scorer.

    Uses a reference image (or computes mean/std from calibration frames).
    Score = normalized MSE + gradient difference + color histogram difference.
    No external ML dependencies required.
    """

    def __init__(self, ref_path: str | None = None) -> None:
        self._ref_gray: np.ndarray | None = None
        self._ref_grad: np.ndarray | None = None
        self._ref_hist: np.ndarray | None = None
        self._ref_shape: tuple[int, int] | None = None
        if ref_path:
            self.load_reference(ref_path)

    def load_reference(self, path: str) -> None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot load reference image: {path}")
        self._ref_gray = img.astype(np.float32)
        self._ref_grad = cv2.Laplacian(self._ref_gray, cv2.CV_32F)
        self._ref_hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        self._ref_hist = self._ref_hist / max(self._ref_hist.sum(), 1e-8)
        self._ref_shape = img.shape[:2]

    def set_reference_from_frame(self, crop: np.ndarray) -> None:
        """Set reference from a single good frame crop."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        self._ref_gray = gray.astype(np.float32)
        self._ref_grad = cv2.Laplacian(self._ref_gray, cv2.CV_32F)
        self._ref_hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        self._ref_hist = self._ref_hist / max(self._ref_hist.sum(), 1e-8)
        self._ref_shape = gray.shape[:2]

    def score(self, crop: np.ndarray) -> tuple[float, np.ndarray]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        h, w = gray.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        if self._ref_gray is None:
            # No reference — compute basic stats as fallback
            score = float(np.std(gray) / 128.0)
            heatmap = gray.astype(np.float32) / 255.0
            return score, heatmap

        # Resize if needed
        if (h, w) != self._ref_shape:
            gray_resized = cv2.resize(gray, (self._ref_shape[1], self._ref_shape[0]))
            ref_gray = self._ref_gray
        else:
            gray_resized = gray
            ref_gray = self._ref_gray

        # 1. Pixel MSE
        diff = (gray_resized.astype(np.float32) - ref_gray) / 255.0
        mse = float(np.mean(diff ** 2))

        # 2. Gradient difference
        grad = cv2.Laplacian(gray_resized, cv2.CV_32F)
        grad_diff = np.abs(grad - self._ref_grad) / 255.0
        grad_score = float(np.mean(grad_diff))

        # 3. Histogram difference (color distribution)
        hist = cv2.calcHist([gray_resized], [0], None, [256], [0, 256]).flatten()
        hist = hist / max(hist.sum(), 1e-8)
        hist_diff = float(np.sum(np.abs(hist - self._ref_hist)) / 2.0)

        # Combined score (heuristic weights)
        score = mse * 10.0 + grad_score * 5.0 + hist_diff * 2.0

        # Per-pixel heatmap from combined diff
        heatmap = cv2.resize(
            np.abs(diff).mean(axis=2) if diff.ndim == 3 else np.abs(diff),
            (w, h),
        ) if (h, w) != self._ref_shape else (
            np.abs(diff) if diff.ndim == 2 else np.abs(diff).mean(axis=2)
        )

        return score, heatmap

    def calibrate(self, reference_crops: list[np.ndarray]) -> dict[str, Any]:
        scores = []
        for crop in reference_crops:
            self.set_reference_from_frame(crop)
            s, _ = self.score(crop)
            scores.append(s)

        # Cross-validate: score each against all others
        scores = []
        for i, crop_i in enumerate(reference_crops):
            self.set_reference_from_frame(crop_i)
            for j, crop_j in enumerate(reference_crops):
                if i == j:
                    continue
                s, _ = self.score(crop_j)
                scores.append(s)

        if not scores:
            return {"threshold": 0.5, "mean_score": 0.0, "std_score": 1.0}

        mean_s = float(np.mean(scores))
        std_s = float(np.std(scores)) if len(scores) > 1 else mean_s * 0.5
        # Suggested threshold: mean + 3*std (covers 99.7% of good parts)
        threshold = mean_s + 3.0 * max(std_s, 0.01)

        return {
            "threshold": round(threshold, 4),
            "mean_score": round(mean_s, 4),
            "std_score": round(std_s, 4),
            "num_samples": len(scores),
        }


class AnomalibScorer(AnomalyScorer):
    """Wraps an anomalib PatchCore model.

    Requires: pip install anomalib torch
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: str) -> Any:
        try:
            import torch
        except ImportError:
            raise ImportError("AnomalibScorer requires torch. pip install torch")

        try:
            from anomalib.models import Patchcore
            from anomalib.deploy import export_to_torch

            model = Patchcore.load_from_checkpoint(model_path)
            model.eval()
            return model
        except Exception as exc:
            raise ImportError(
                f"Failed to load anomalib model from {model_path}: {exc}"
            )

    def score(self, crop: np.ndarray) -> tuple[float, np.ndarray]:
        import torch
        # Convert crop to torch tensor
        input_tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            result = self._model(input_tensor)
        anomaly_map = result.get("anomaly_map", np.zeros((crop.shape[0], crop.shape[1])))
        score = float(result.get("pixel_scores", result.get("image_score", 0.0)))
        return score, anomaly_map

    def calibrate(self, reference_crops: list[np.ndarray]) -> dict[str, Any]:
        # anomalib usually doesn't need calibration — use default threshold
        return {"threshold": 0.5, "note": "use model's default threshold"}


def get_scorer(model_path: str | None = None) -> AnomalyScorer:
    """Get or create a cached scorer for the given model path.

    If model_path is None or empty, returns SimpleAnomalyScorer.
    If model_path ends with .ckpt, tries to load AnomalibScorer.
    Otherwise tries SimpleAnomalyScorer with the path as reference image.
    """
    cache_key = model_path or "__simple__"

    if cache_key in _scorer_cache:
        return _scorer_cache[cache_key]

    if not model_path:
        scorer: AnomalyScorer = SimpleAnomalyScorer()
    elif model_path.endswith(".ckpt") or model_path.endswith(".torch"):
        try:
            scorer = AnomalibScorer(model_path)
        except ImportError:
            logger.warning(
                "Anomalib not available for %s — falling back to SimpleAnomalyScorer",
                model_path,
            )
            scorer = SimpleAnomalyScorer()
    else:
        # Try as reference image path
        if Path(model_path).exists():
            scorer = SimpleAnomalyScorer(model_path)
        else:
            logger.warning("Model/reference path not found: %s — using empty scorer", model_path)
            scorer = SimpleAnomalyScorer()

    _scorer_cache[cache_key] = scorer
    return scorer


def clear_cache() -> None:
    _scorer_cache.clear()
