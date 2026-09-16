"""Single-source-of-truth helpers for mode normalization and display in the client.

Mirrors the backend normalize_mode() in shared/contracts/templates.py
so the client doesn't need to import from backend modules at import time.
"""

# Canonical normalization mapping (must match backend _MODE_ALIASES)
_MODE_ALIASES: dict[str, str] = {
    "component_count": "counter",
    "count": "counter",
    "counter": "counter",
    "ml_detection": "sticker",
    "sticker": "sticker",
    "defect": "defect",
    "": "sticker",
}

# Human-readable labels (canonical -> display string)
_MODE_LABELS: dict[str, str] = {
    "counter": "Component Counter",
    "defect": "Defect Scan",
    "sticker": "QC Sticker",
}

# Radio-button values (canonical -> tkinter radio value)
_MODE_RADIO: dict[str, str] = {
    "counter": "component_count",
    "defect": "defect",
    "sticker": "sticker",
}

# Reverse lookup: radio value -> canonical
_RADIO_TO_MODE: dict[str, str] = {
    "component_count": "counter",
    "defect": "defect",
    "sticker": "sticker",
}

# Legacy validator_mode -> canonical
_LEGACY_TO_MODE: dict[str, str] = {
    "component_count": "counter",
    "defect": "defect",
    "ml_detection": "sticker",
}


def normalize_mode(raw: str | None) -> str:
    """Normalize any mode string to canonical form (one of sticker|counter|defect).

    Handles legacy aliases: component_count -> counter, ml_detection -> sticker, etc.
    Unknown values are returned as-is.
    """
    key = str(raw or "").strip().lower()
    return _MODE_ALIASES.get(key, key)


def mode_label(raw: str | None) -> str:
    """Return human-readable label for a mode value.

    Accepts both canonical and legacy mode strings.
    """
    canonical = normalize_mode(raw)
    return _MODE_LABELS.get(canonical, "QC Sticker")


def mode_to_radio(raw: str | None) -> str:
    """Convert canonical or legacy mode to tkinter radio button value.

    Returns 'sticker', 'component_count', or 'defect'.
    """
    canonical = normalize_mode(raw)
    return _MODE_RADIO.get(canonical, "sticker")


def radio_to_mode(radio_val: str | None) -> str:
    """Convert radio button value back to canonical mode name."""
    key = str(radio_val or "sticker").strip().lower()
    return _RADIO_TO_MODE.get(key, "sticker")


def mode_from_template(detail: dict) -> str:
    """Extract canonical mode name from a template detail dict.

    Reads top-level 'mode' first, falls back to sticker.validator_mode,
    then 'ml_detection' as the ultimate default.
    """
    if not detail:
        return "sticker"
    raw = str(detail.get("mode") or "").strip()
    if raw:
        return normalize_mode(raw)
    sticker = detail.get("sticker") or {}
    vm = str(sticker.get("validator_mode") or "ml_detection").strip().lower()
    return _LEGACY_TO_MODE.get(vm, "sticker")


def is_mode(detail: dict, target: str) -> bool:
    """Check if template detail matches the given canonical mode name.

    Sugar: ``is_mode(detail, \"counter\")`` instead of
    ``mode_from_template(detail) == \"counter\"``.
    """
    return mode_from_template(detail) == normalize_mode(target)


def format_range_widget(min_val, max_val):
    """Format min/max for Tkinter (supports unicode). Never shows None."""
    if min_val is not None and max_val is not None:
        if min_val == max_val:
            return str(min_val)
        return f"{min_val}-{max_val}"
    elif min_val is not None:
        return f"min {min_val}"
    elif max_val is not None:
        return f"max {max_val}"
    return "?"


def format_range_overlay(min_val, max_val):
    """Format min/max for OpenCV overlay (ASCII only). Never shows None."""
    if min_val is not None and max_val is not None:
        if min_val == max_val:
            return str(min_val)
        return f"{min_val}-{max_val}"
    elif min_val is not None:
        return f"{min_val}+"
    elif max_val is not None:
        return str(max_val)
    return "?"


def validator_mode_for_payload(mode_radio: str) -> str:
    """Map radio button value to the validator_mode string saved in sticker.validator_mode.

    ``sticker`` -> ``\"ml_detection\"``
    ``component_count`` -> ``\"component_count\"``
    ``defect`` -> ``\"defect\"``
    """
    canonical = radio_to_mode(mode_radio)
    return {
        "sticker": "ml_detection",
        "counter": "component_count",
        "defect": "defect",
    }.get(canonical, "ml_detection")
