"""JSON safety helper — convert NumPy types to native Python before jsonify."""
from __future__ import annotations

from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Recursively convert NumPy scalars/arrays to native Python types.

    Safe for: dict, list, tuple, set, np.floating, np.integer, np.bool_,
    np.ndarray, None, str, int, float, bool.
    """
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    # Native Python types: pass through
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Fallback: try to convert to native
    try:
        return float(value) if hasattr(value, "__float__") else str(value)
    except Exception:
        return str(value)


def safe_jsonify(response: Any) -> Any:
    """Convert response to JSON-safe and call Flask jsonify."""
    from flask import jsonify
    return jsonify(to_jsonable(response))
