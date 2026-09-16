"""shared/contracts/augment.py — transform catalog shared between backend and UI.

Transform categories
--------------------
photometric
    Colour/texture changes only.  Object geometry is unchanged so bounding-box
    annotations can be copied verbatim regardless of the feature flag.
    Transforms: brightness, contrast, blur, noise.

geometric_safe
    Spatial transforms whose label mapping is well-defined and implemented by the
    label geometry engine.  Allowed in version snapshots when
    ``QC_SUITE_GEOMETRIC_AUGMENT_ENABLED=1``.
    Transforms: flip_h, flip_v, rotate.

experimental
    Spatial transforms that are more complex or less stable.  Gated behind a
    separate flag (future work).  Not yet implemented.
    Transforms: shear, perspective.
"""
from __future__ import annotations

from typing import Literal

TransformCategory = Literal["photometric", "geometric_safe", "experimental"]


class TransformInfo:
    """Metadata for a single augmentation transform."""

    __slots__ = ("name", "category", "label_transform", "description")

    def __init__(
        self,
        name: str,
        category: TransformCategory,
        label_transform: str | None,
        description: str,
    ) -> None:
        self.name = name
        self.category = category
        # Name of the label-geometry engine operation, or None if not needed/supported.
        self.label_transform = label_transform
        self.description = description

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "label_transform": self.label_transform,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Canonical catalog — single source of truth for UI and backend
# ---------------------------------------------------------------------------

TRANSFORM_CATALOG: dict[str, TransformInfo] = {
    # ---- photometric -------------------------------------------------------
    "brightness": TransformInfo(
        "brightness", "photometric", None,
        "Random brightness shift ±40.",
    ),
    "contrast": TransformInfo(
        "contrast", "photometric", None,
        "Random contrast scale 0.7–1.3.",
    ),
    "blur": TransformInfo(
        "blur", "photometric", None,
        "Gaussian blur with kernel 3/5/7.",
    ),
    "noise": TransformInfo(
        "noise", "photometric", None,
        "Additive Gaussian noise σ≈15.",
    ),
    # ---- geometric_safe ----------------------------------------------------
    "flip_h": TransformInfo(
        "flip_h", "geometric_safe", "flip_h",
        "Horizontal flip (mirror left–right).",
    ),
    "flip_v": TransformInfo(
        "flip_v", "geometric_safe", "flip_v",
        "Vertical flip (mirror top–bottom).",
    ),
    "rotate": TransformInfo(
        "rotate", "geometric_safe", "rotate",
        "Random rotation ±15°.",
    ),
    # ---- experimental (not yet implemented) --------------------------------
    "shear": TransformInfo(
        "shear", "experimental", None,
        "Affine shear warp (experimental, not yet supported).",
    ),
    "perspective": TransformInfo(
        "perspective", "experimental", None,
        "Perspective warp (experimental, not yet supported).",
    ),
}

# Convenience sets used for fast membership tests.
PHOTOMETRIC_TRANSFORMS: frozenset[str] = frozenset(
    k for k, v in TRANSFORM_CATALOG.items() if v.category == "photometric"
)
GEOMETRIC_SAFE_TRANSFORMS: frozenset[str] = frozenset(
    k for k, v in TRANSFORM_CATALOG.items() if v.category == "geometric_safe"
)
EXPERIMENTAL_TRANSFORMS: frozenset[str] = frozenset(
    k for k, v in TRANSFORM_CATALOG.items() if v.category == "experimental"
)


def build_capabilities(*, geometric_augment_enabled: bool) -> dict:
    """Return the capability contract dict to be served by the API and consumed by the UI."""
    transforms_out = []
    for info in TRANSFORM_CATALOG.values():
        entry = info.to_dict()
        if info.category == "photometric":
            entry["available"] = True
            entry["warning"] = None
        elif info.category == "geometric_safe":
            entry["available"] = geometric_augment_enabled
            entry["warning"] = (
                None if geometric_augment_enabled
                else "Geometric augmentation is disabled. Enable QC_SUITE_GEOMETRIC_AUGMENT_ENABLED=1."
            )
        else:  # experimental
            entry["available"] = False
            entry["warning"] = "Experimental transform — not yet supported."
        transforms_out.append(entry)
    return {
        "geometric_augment_enabled": geometric_augment_enabled,
        "transforms": transforms_out,
    }
