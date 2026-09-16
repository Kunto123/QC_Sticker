from __future__ import annotations

from backend.app.services.evaluators.base import ModeEvaluator
from backend.app.services.evaluators.counter import CounterEvaluator
from backend.app.services.evaluators.defect import DefectEvaluator
# TODO(B5): connect StickerEvaluator once migration from inline _validate_sticker is complete.
# from backend.app.services.evaluators.sticker import StickerEvaluator

# ── Registry ────────────────────────────────────────────────────────────────
# To add a new mode: create an evaluator class, import it, add to _EVALUATORS.
# No other file needs an `if mode == ...` for evaluation dispatch.

_EVALUATORS: dict[str, ModeEvaluator] = {
    e.mode_name: e
    for e in (
        # StickerEvaluator(),  # B5: not yet connected — sticker uses inline path
        CounterEvaluator(),
        DefectEvaluator(),
    )
}


def get_evaluator(mode: str) -> ModeEvaluator:
    """Return the evaluator registered for *mode*.

    Raises KeyError if *mode* is unknown.
    """
    if mode not in _EVALUATORS:
        msg = f"No evaluator registered for mode={mode!r}. Available: {list(_EVALUATORS)}"
        raise KeyError(msg)
    return _EVALUATORS[mode]


def has_evaluator(mode: str) -> bool:
    """Check if a mode has a registered evaluator (without raising)."""
    return mode in _EVALUATORS


def registered_modes() -> list[str]:
    """Return list of registered mode names."""
    return list(_EVALUATORS)
