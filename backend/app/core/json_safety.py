"""Helper keamanan JSON — konversi tipe NumPy ke Python native sebelum jsonify."""
from __future__ import annotations

from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Konversi rekursif skalar/array NumPy ke tipe Python native.

    Aman untuk: dict, list, tuple, set, np.floating, np.integer, np.bool_,
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
    # Tipe Python native: teruskan apa adanya
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Fallback: coba konversi ke native
    try:
        return float(value) if hasattr(value, "__float__") else str(value)
    except Exception:
        return str(value)


def safe_jsonify(response: Any) -> Any:
    """Konversi response ke bentuk JSON-safe lalu panggil Flask jsonify."""
    from flask import jsonify
    return jsonify(to_jsonable(response))
