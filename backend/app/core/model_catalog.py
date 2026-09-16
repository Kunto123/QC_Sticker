from __future__ import annotations

from typing import Any


_FAMILY_LABELS: dict[str, str] = {
    "yolov5": "YOLOv5",
    "yolov11": "YOLOv11",
}

_VARIANT_LABELS: dict[str, str] = {
    "n": "Nano",
    "s": "Small",
    "m": "Medium",
    "l": "Large",
    "x": "X-Large",
}


def _normalize_identifier(value: str | None) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )


def _build_catalog() -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for family in ("yolov5", "yolov11"):
        family_label = _FAMILY_LABELS[family]
        # Ultralytics YOLO11 uses "yolo11" (no 'v') as the weights prefix.
        # The internal catalog ID keeps "yolov11" for backward compatibility with
        # existing job records; only the weights_name changes.
        weights_prefix = "yolo11" if family == "yolov11" else family
        for variant in ("n", "s", "m", "l", "x"):
            variant_label = _VARIANT_LABELS[variant]
            model_id = f"{family}{variant}"
            catalog.append(
                {
                    "id": model_id,
                    "family": family,
                    "family_label": family_label,
                    "variant": variant,
                    "variant_label": variant_label,
                    "display_name": f"{family_label} {variant_label}",
                    "display_label": f"{family_label} {variant_label} ({model_id})",
                    "runtime": "ultralytics",
                    "task": "detection",
                    "weights_name": f"{weights_prefix}{variant}.pt",
                    "source": "catalog",
                    "description": f"Ultralytics {family_label} {variant_label.lower()} detection base model.",
                }
            )
    return catalog


BASE_MODEL_CATALOG: tuple[dict[str, Any], ...] = tuple(_build_catalog())
BASE_MODEL_BY_ID: dict[str, dict[str, Any]] = {item["id"]: item for item in BASE_MODEL_CATALOG}


def list_base_models(family: str | None = None) -> list[dict[str, Any]]:
    family_key = _normalize_identifier(family) if family else ""
    items = [dict(item) for item in BASE_MODEL_CATALOG]
    if family_key:
        if family_key not in _FAMILY_LABELS:
            raise ValueError(f"Unsupported base model family '{family}'. Must be one of: {sorted(_FAMILY_LABELS)}")
        items = [item for item in items if item["family"] == family_key]
    return items


def get_base_model(base_model_id: str | None) -> dict[str, Any] | None:
    model_id = _normalize_identifier(base_model_id)
    if not model_id:
        return None
    item = BASE_MODEL_BY_ID.get(model_id)
    return dict(item) if item is not None else None


def resolve_base_model(
    base_model: str | None = None,
    *,
    family: str | None = None,
    variant: str | None = None,
) -> dict[str, Any] | None:
    base_model_id = _normalize_identifier(base_model)
    if base_model_id:
        item = BASE_MODEL_BY_ID.get(base_model_id)
        if item is not None:
            return dict(item)

    family_key = _normalize_identifier(family) if family else ""
    variant_key = _normalize_identifier(variant) if variant else ""
    if not family_key and not variant_key:
        return None
    if family_key not in _FAMILY_LABELS:
        raise ValueError(f"Unsupported base model family '{family}'. Must be one of: {sorted(_FAMILY_LABELS)}")
    if variant_key not in _VARIANT_LABELS:
        raise ValueError(f"Unsupported base model variant '{variant}'. Must be one of: {sorted(_VARIANT_LABELS)}")

    model_id = f"{family_key}{variant_key}"
    item = BASE_MODEL_BY_ID.get(model_id)
    if item is None:
        raise ValueError(f"Base model '{model_id}' is not available in the catalog.")
    return dict(item)
