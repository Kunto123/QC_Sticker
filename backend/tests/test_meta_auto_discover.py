"""Test meta auto-discovery and TTL cache.

Plan #9 — Scenarios:
- model path X.xml -> ditemukan X.json (sibling) ✓
- model path X.xml -> ditemukan X.meta.json (sibling) ✓
- model path X.xml -> tidak ada sibling -> return "" ✓
- TTL 30 detik -> reload setelah TTL lewat ✓
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.app.core.config import AppConfig
from backend.app.core.device_runtime import DeviceRuntimeResolver
from backend.app.repositories.models_repository import ModelsRepository
from backend.app.services.sticker_inference import StickerInferenceService
from shared.contracts.templates import VisionConfig


def _make_service(tmp_dir: str) -> StickerInferenceService:
    """Create service with temp directory for meta files."""
    config = AppConfig()
    config.default_sticker_model_path = ""
    config.sticker_inference_mode = "openvino"

    with patch.object(
        DeviceRuntimeResolver, "__init__", lambda self, cfg=None: None
    ):
        device_runtime = DeviceRuntimeResolver.__new__(DeviceRuntimeResolver)

    with patch.object(
        StickerInferenceService, "__init__", lambda self, **kw: None
    ):
        svc = StickerInferenceService.__new__(StickerInferenceService)

    svc._config = config
    svc._models_repo = MagicMock()
    svc._device_runtime = device_runtime
    svc._runtime_lock = __import__("threading").RLock()
    svc._loaded_models = {}
    svc._meta_cache = {}
    return svc


class TestMetaAutoDiscover:
    """Test _resolve_meta_path auto-discovery logic."""

    def test_sibling_json_found(self, tmp_path):
        """model.xml -> ditemukan model.json di folder sama."""
        model_file = tmp_path / "yolov5.xml"
        meta_file = tmp_path / "yolov5.json"
        model_file.write_text("<model/>")
        meta_file.write_text(json.dumps({"class_names": ["A", "B"]}))

        svc = _make_service(str(tmp_path))
        vision = VisionConfig(model_path=str(model_file))
        result = svc._resolve_meta_path(vision)
        assert result == str(meta_file)

    def test_sibling_meta_json_found(self, tmp_path):
        """model.xml -> ditemukan model.meta.json di folder sama."""
        model_file = tmp_path / "yolov5.xml"
        meta_file = tmp_path / "yolov5.meta.json"
        model_file.write_text("<model/>")
        meta_file.write_text(json.dumps({"class_names": ["C"]}))

        svc = _make_service(str(tmp_path))
        vision = VisionConfig(model_path=str(model_file))
        result = svc._resolve_meta_path(vision)
        assert result == str(meta_file)

    def test_no_sibling_returns_empty(self, tmp_path):
        """model.xml -> tidak ada sibling -> return ''."""
        model_file = tmp_path / "yolov5.xml"
        model_file.write_text("<model/>")

        svc = _make_service(str(tmp_path))
        vision = VisionConfig(model_path=str(model_file))
        result = svc._resolve_meta_path(vision)
        assert result == ""

    def test_empty_model_path_returns_empty(self, tmp_path):
        """model path kosong -> return ''."""
        svc = _make_service(str(tmp_path))
        vision = VisionConfig(model_path="")
        result = svc._resolve_meta_path(vision)
        assert result == ""

    def test_meta_json_takes_priority_over_json(self, tmp_path):
        """Jika ada X.meta.json DAN X.json, X.meta.json yang dipilih (dicari dulu)."""
        model_file = tmp_path / "yolov5.xml"
        json_file = tmp_path / "yolov5.json"
        meta_json_file = tmp_path / "yolov5.meta.json"
        model_file.write_text("<model/>")
        json_file.write_text(json.dumps({"class_names": ["from_json"]}))
        meta_json_file.write_text(
            json.dumps({"class_names": ["from_meta_json"]})
        )

        svc = _make_service(str(tmp_path))
        vision = VisionConfig(model_path=str(model_file))
        result = svc._resolve_meta_path(vision)
        # .meta.json dicari dulu, jadi harusnya yang ini yang ditemukan
        assert result == str(meta_json_file)


class TestMetaCacheTTL:
    """Test TTL 30 detik pada _load_meta."""

    def test_cache_hit_within_ttl(self, tmp_path):
        """Cache hit dalam 30 detik -> tidak baca file lagi."""
        meta_file = tmp_path / "model.json"
        meta_file.write_text(json.dumps({"class_names": ["A"]}))

        svc = _make_service(str(tmp_path))

        # First load
        result1 = svc._load_meta(str(meta_file))
        assert result1 == {"class_names": ["A"]}

        # Modify file content
        meta_file.write_text(json.dumps({"class_names": ["B"]}))

        # Second load within TTL -> should return cached
        result2 = svc._load_meta(str(meta_file))
        assert result2 == {"class_names": ["A"]}  # still cached

    def test_cache_expired_after_ttl(self, tmp_path):
        """Setelah 30 detik, cache expired -> baca ulang dari file."""
        meta_file = tmp_path / "model.json"
        meta_file.write_text(json.dumps({"class_names": ["A"]}))

        svc = _make_service(str(tmp_path))

        # First load
        result1 = svc._load_meta(str(meta_file))
        assert result1 == {"class_names": ["A"]}

        # Modify file
        meta_file.write_text(json.dumps({"class_names": ["B"]}))

        # Simulate TTL expiry by manipulating cache timestamp
        with svc._runtime_lock:
            payload, _ = svc._meta_cache[str(meta_file)]
            svc._meta_cache[str(meta_file)] = (payload, time.monotonic() - 31.0)

        # Second load after TTL -> should re-read file
        result2 = svc._load_meta(str(meta_file))
        assert result2 == {"class_names": ["B"]}

    def test_missing_file_not_cached(self, tmp_path):
        """File yang tidak ada tidak di-cache."""
        svc = _make_service(str(tmp_path))
        missing = str(tmp_path / "nonexistent.json")

        result = svc._load_meta(missing)
        assert result == {}
        assert missing not in svc._meta_cache

    def test_empty_meta_path_returns_empty(self, tmp_path):
        """Empty meta path -> return {} tanpa error."""
        svc = _make_service(str(tmp_path))
        result = svc._load_meta("")
        assert result == {}

    def test_parse_error_returns_empty(self, tmp_path):
        """File JSON yang corrupt -> return {}."""
        meta_file = tmp_path / "model.json"
        meta_file.write_text("not valid json {{{")

        svc = _make_service(str(tmp_path))
        result = svc._load_meta(str(meta_file))
        assert result == {}


# Import MagicMock for _make_service
from unittest.mock import MagicMock  # noqa: E402
