from __future__ import annotations

import base64
import math
from typing import Any

import cv2
import numpy as np


MIN_CALIBRATION_PROFILE_PIXELS = 64


class CalibrationService:
    @staticmethod
    def decode_image(image_b64: str):
        raw = base64.b64decode(image_b64)
        arr = np.frombuffer(raw, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image payload.")
        return image

    @staticmethod
    def apply_roi(image, roi: dict[str, Any] | None):
        if image is None or image.size == 0:
            raise ValueError("Invalid image for ROI crop.")
        if not roi:
            return image
        height, width = image.shape[:2]
        x = max(0, min(width - 1, int(float(roi.get("x", 0.0)) * width)))
        y = max(0, min(height - 1, int(float(roi.get("y", 0.0)) * height)))
        roi_w = max(1, int(float(roi.get("w", 1.0)) * width))
        roi_h = max(1, int(float(roi.get("h", 1.0)) * height))
        x2 = min(width, x + roi_w)
        y2 = min(height, y + roi_h)
        cropped = image[y:y2, x:x2]
        if cropped.size == 0:
            raise ValueError("ROI crop is empty.")
        return cropped

    @staticmethod
    def _convert_pixels(image, colorspace: str) -> tuple[np.ndarray, np.ndarray, tuple[str, str, str]]:
        colorspace = str(colorspace or "LAB").upper()
        if colorspace == "LAB":
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)
            converted[:, :, 0] *= (100.0 / 255.0)
            converted[:, :, 1] -= 128.0
            converted[:, :, 2] -= 128.0
            pixels = converted.reshape(-1, 3)
            mean_bgr = image.mean(axis=(0, 1))
            reference_rgb = np.array([float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0])], dtype=float)
            labels = ("l", "a", "b")
            return pixels, reference_rgb, labels
        pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(float)
        mean_rgb = pixels.mean(axis=0)
        labels = ("r", "g", "b")
        return pixels, mean_rgb, labels

    @staticmethod
    def compute_color_profile(image, colorspace: str = "LAB") -> dict[str, Any]:
        pixels, reference_rgb, labels = CalibrationService._convert_pixels(image, colorspace)
        height, width = image.shape[:2]
        total_pixels = int(height * width)
        if total_pixels < MIN_CALIBRATION_PROFILE_PIXELS:
            raise ValueError(
                "Calibration ROI too small. "
                f"Minimum {MIN_CALIBRATION_PROFILE_PIXELS} pixels required, got {total_pixels}."
            )

        mean_vals = pixels.mean(axis=0)
        std_vals = pixels.std(axis=0)
        pooled_std = math.sqrt(sum(v ** 2 for v in std_vals) / 3.0)

        red, green, blue = float(reference_rgb[0]), float(reference_rgb[1]), float(reference_rgb[2])
        profile_colorspace = str(colorspace or "LAB").upper()
        return {
            "schema_version": 1,
            "method": "color_profile_match",
            "colorspace": profile_colorspace,
            "reference_source": "snippet",
            "reference_color": {
                "hex": "#{:02x}{:02x}{:02x}".format(int(round(red)), int(round(green)), int(round(blue))),
                "rgb": {"r": round(red, 2), "g": round(green, 2), "b": round(blue, 2)},
            },
            "reference_stats": {
                "mean": {label: round(float(value), 4) for label, value in zip(labels, mean_vals)},
                "std": {label: round(float(value), 4) for label, value in zip(labels, std_vals)},
            },
            "tolerance": {"distance_threshold": round(max(8.0, 3.0 * pooled_std), 2)},
            "min_match_ratio": 0.8,
            "sampling_meta": {"width": width, "height": height, "total_pixels": total_pixels},
        }

    @staticmethod
    def evaluate_color_match(
        image,
        profile: dict[str, Any],
        *,
        colorspace: str | None = None,
        distance_threshold: float | None = None,
        min_match_ratio: float | None = None,
    ) -> dict[str, Any]:
        if image is None or image.size == 0:
            raise ValueError("Invalid image for color evaluation.")

        profile_colorspace = str(colorspace or profile.get("colorspace") or "LAB").upper()
        pixels, _, labels = CalibrationService._convert_pixels(image, profile_colorspace)

        if profile_colorspace == "LAB":
            mean = (profile.get("reference_stats") or {}).get("mean") or {}
            reference = np.array(
                [
                    float(mean.get("l", 0.0)),
                    float(mean.get("a", 0.0)),
                    float(mean.get("b", 0.0)),
                ],
                dtype=float,
            )
        else:
            rgb = (profile.get("reference_color") or {}).get("rgb") or {}
            reference = np.array(
                [
                    float(rgb.get("r", 0.0)),
                    float(rgb.get("g", 0.0)),
                    float(rgb.get("b", 0.0)),
                ],
                dtype=float,
            )

        distances = np.sqrt(np.sum((pixels - reference) ** 2, axis=1))
        resolved_distance_threshold = float(
            distance_threshold
            if distance_threshold is not None
            else (profile.get("tolerance") or {}).get("distance_threshold") or 12.0
        )
        resolved_min_match_ratio = float(
            min_match_ratio if min_match_ratio is not None else profile.get("min_match_ratio") or 0.75
        )
        match_ratio = float(np.mean(distances <= resolved_distance_threshold))
        mean_distance = float(distances.mean()) if len(distances) else 0.0
        ready = match_ratio >= resolved_min_match_ratio
        return {
            "colorspace": profile_colorspace,
            "labels": labels,
            "distance_threshold": round(resolved_distance_threshold, 4),
            "min_match_ratio": round(resolved_min_match_ratio, 6),
            "match_ratio": round(match_ratio, 6),
            "mean_distance": round(mean_distance, 6),
            "is_match": ready,
        }
