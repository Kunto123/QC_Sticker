"""test_training_worker_model_resolution.py — regression matrix for the layered weights resolver.

Scenarios (Phase 7):
  1. Local file available               → source="local_cache", resolved to MODELS_DIR/<file>
  2. Local missing + download ON        → source="download",    resolved to canonical alias (no .pt)
  3. Local missing + download OFF       → FileNotFoundError with actionable message
  4. Absolute path valid                → source="absolute",    resolved to the given path
  5. Absolute path invalid              → FileNotFoundError with absolute-path context
  6. Error message quality              → message contains MODELS_DIR path + how-to-fix hint
  7. Simulated mode non-regression      → _resolve_weights still works; simulated path never calls it

YOLO11 canonical naming (added in fix-yolo11-autodownload):
  8. Legacy yolov11m.pt → canonical download alias yolo11m
  9. Canonical yolo11m.pt → download alias yolo11m unchanged
  10. Legacy yolov11m.pt in MODELS_DIR → found via raw-name fallback (backward compat)
  11. Canonical yolo11m.pt in MODELS_DIR → found via canonical path (preferred)
  12. YOLOv5 path unaffected by YOLO11 normalizer
  13. Offline-strict error for YOLO11 mentions canonical name
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.app.workers.training_worker import TrainingWorker, WeightsResolution


def _make_worker(*, download_allowed: bool = True) -> TrainingWorker:
    """Return a TrainingWorker with a mock repo and minimal config."""
    config = MagicMock()
    config.training_engine_mode = "simulated"
    config.training_weights_download_allowed = download_allowed
    config.training_timeout_minutes = 1
    config.gpu_fail_fast = True
    device_runtime = MagicMock()
    return TrainingWorker(
        training_repo=MagicMock(),
        app_config=config,
        device_runtime=device_runtime,
    )


def _job(weights_name: str) -> dict:
    return {"base_model_weights_name": weights_name}


class WeightsResolverLocalCacheTest(unittest.TestCase):
    """Scenario 1: local file present in MODELS_DIR."""

    def test_resolves_local_cache_when_file_exists(self) -> None:
        worker = _make_worker(download_allowed=True)
        with tempfile.TemporaryDirectory(prefix="qc-models-") as tmp:
            fake_models_dir = Path(tmp)
            weights_file = fake_models_dir / "yolov11m.pt"
            weights_file.write_bytes(b"fake weights")

            with patch("backend.app.workers.training_worker.MODELS_DIR", fake_models_dir):
                result = worker._resolve_weights(_job("yolov11m.pt"))

        self.assertIsInstance(result, WeightsResolution)
        self.assertEqual(result.weights_source, "local_cache")
        self.assertEqual(result.weights_input, "yolov11m.pt")
        self.assertEqual(Path(result.resolved_path), fake_models_dir / "yolov11m.pt")
        self.assertTrue(any("yolov11m.pt" in a for a in result.resolution_attempts))

    def test_local_cache_prefers_local_over_download(self) -> None:
        """When local file exists AND download is allowed, local wins."""
        worker = _make_worker(download_allowed=True)
        with tempfile.TemporaryDirectory(prefix="qc-models-") as tmp:
            fake_models_dir = Path(tmp)
            (fake_models_dir / "yolov11n.pt").write_bytes(b"x")

            with patch("backend.app.workers.training_worker.MODELS_DIR", fake_models_dir):
                result = worker._resolve_weights(_job("yolov11n.pt"))

        self.assertEqual(result.weights_source, "local_cache")


class WeightsResolverDownloadTest(unittest.TestCase):
    """Scenario 2: local missing + download ON."""

    def test_falls_back_to_download_alias_when_local_missing(self) -> None:
        worker = _make_worker(download_allowed=True)
        with tempfile.TemporaryDirectory(prefix="qc-models-") as tmp:
            empty_models_dir = Path(tmp)  # no weights files here

            with patch("backend.app.workers.training_worker.MODELS_DIR", empty_models_dir):
                result = worker._resolve_weights(_job("yolov11m.pt"))

        self.assertEqual(result.weights_source, "download")
        # canonical alias: yolov11m.pt → normalised to yolo11m (Ultralytics naming, no 'v')
        self.assertEqual(result.resolved_path, "yolo11m")
        self.assertTrue(any("download:" in a for a in result.resolution_attempts))

    def test_download_alias_strips_pt_extension(self) -> None:
        worker = _make_worker(download_allowed=True)
        with tempfile.TemporaryDirectory(prefix="qc-models-") as tmp:
            with patch("backend.app.workers.training_worker.MODELS_DIR", Path(tmp)):
                result = worker._resolve_weights(_job("yolov5n.pt"))

        self.assertEqual(result.resolved_path, "yolov5n")


class WeightsResolverOfflineStrictTest(unittest.TestCase):
    """Scenario 3: local missing + download OFF → actionable error."""

    def test_raises_when_local_missing_and_download_disabled(self) -> None:
        worker = _make_worker(download_allowed=False)
        with tempfile.TemporaryDirectory(prefix="qc-models-") as tmp:
            with patch("backend.app.workers.training_worker.MODELS_DIR", Path(tmp)):
                with self.assertRaises(FileNotFoundError):
                    worker._resolve_weights(_job("yolov11m.pt"))

    def test_error_lists_candidates_tried(self) -> None:
        worker = _make_worker(download_allowed=False)
        with tempfile.TemporaryDirectory(prefix="qc-models-") as tmp:
            with patch("backend.app.workers.training_worker.MODELS_DIR", Path(tmp)):
                try:
                    worker._resolve_weights(_job("yolov11m.pt"))
                    self.fail("Expected FileNotFoundError")
                except FileNotFoundError as exc:
                    self.assertIn("yolov11m.pt", str(exc))


class WeightsResolverAbsolutePathTest(unittest.TestCase):
    """Scenario 4: valid absolute path."""

    def test_resolves_absolute_path_when_file_exists(self) -> None:
        worker = _make_worker()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(b"fake")

        try:
            with patch("backend.app.workers.training_worker.MODELS_DIR", Path("/nonexistent")):
                result = worker._resolve_weights(_job(str(tmp_path)))

            self.assertEqual(result.weights_source, "absolute")
            self.assertEqual(result.resolved_path, str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_raises_for_nonexistent_absolute_path(self) -> None:
        """Scenario 5: absolute path given but file doesn't exist."""
        worker = _make_worker()
        nonexistent = Path(tempfile.gettempdir()) / "does_not_exist_at_all.pt"
        assert not nonexistent.exists()

        with self.assertRaises(FileNotFoundError) as ctx:
            worker._resolve_weights(_job(str(nonexistent)))

        self.assertIn(str(nonexistent), str(ctx.exception))


class WeightsResolverErrorMessageQualityTest(unittest.TestCase):
    """Scenario 6: error message must contain actionable guidance."""

    def test_offline_strict_error_contains_fix_hint(self) -> None:
        worker = _make_worker(download_allowed=False)
        with tempfile.TemporaryDirectory(prefix="qc-models-") as tmp:
            with patch("backend.app.workers.training_worker.MODELS_DIR", Path(tmp)):
                try:
                    worker._resolve_weights(_job("yolov11m.pt"))
                    self.fail("Expected FileNotFoundError")
                except FileNotFoundError as exc:
                    msg = str(exc)
                    self.assertIn("QC_SUITE_TRAINING_WEIGHTS_DOWNLOAD_ALLOWED", msg)
                    self.assertIn("MODELS_DIR", msg)

    def test_offline_strict_error_contains_weights_name(self) -> None:
        worker = _make_worker(download_allowed=False)
        with tempfile.TemporaryDirectory(prefix="qc-models-") as tmp:
            with patch("backend.app.workers.training_worker.MODELS_DIR", Path(tmp)):
                try:
                    worker._resolve_weights(_job("yolov5x.pt"))
                    self.fail("Expected FileNotFoundError")
                except FileNotFoundError as exc:
                    self.assertIn("yolov5x.pt", str(exc))


class WeightsResolverSimulatedNonRegressionTest(unittest.TestCase):
    """Scenario 7: simulated training mode still resolves weights correctly."""

    def test_resolve_weights_works_in_simulated_mode(self) -> None:
        """_resolve_weights is mode-agnostic; resolver should still return a result."""
        worker = _make_worker(download_allowed=True)
        self.assertEqual(worker._training_mode, "simulated")

        with tempfile.TemporaryDirectory(prefix="qc-models-") as tmp:
            fake_models_dir = Path(tmp)
            (fake_models_dir / "yolov11n.pt").write_bytes(b"x")

            with patch("backend.app.workers.training_worker.MODELS_DIR", fake_models_dir):
                result = worker._resolve_weights(_job("yolov11n.pt"))

        self.assertEqual(result.weights_source, "local_cache")

    def test_resolve_weights_returns_download_in_simulated_mode_when_local_missing(self) -> None:
        worker = _make_worker(download_allowed=True)
        with tempfile.TemporaryDirectory(prefix="qc-models-") as tmp:
            with patch("backend.app.workers.training_worker.MODELS_DIR", Path(tmp)):
                result = worker._resolve_weights(_job("yolov11n.pt"))

        self.assertEqual(result.weights_source, "download")


class Yolo11CanonicalNamingTest(unittest.TestCase):
    """Scenarios 8–13: YOLO11 legacy-to-canonical normalizer and non-regression for YOLOv5."""

    # ------------------------------------------------------------------
    # _canonicalize_weights_name unit tests
    # ------------------------------------------------------------------

    def test_canonicalize_yolo11_legacy_to_canonical(self) -> None:
        from backend.app.workers.training_worker import TrainingWorker
        for variant in ("n", "s", "m", "l", "x"):
            with self.subTest(variant=variant):
                result = TrainingWorker._canonicalize_weights_name(f"yolov11{variant}.pt")
                self.assertEqual(result, f"yolo11{variant}.pt")

    def test_canonicalize_already_canonical_unchanged(self) -> None:
        from backend.app.workers.training_worker import TrainingWorker
        self.assertEqual(TrainingWorker._canonicalize_weights_name("yolo11m.pt"), "yolo11m.pt")
        self.assertEqual(TrainingWorker._canonicalize_weights_name("yolo11n.pt"), "yolo11n.pt")

    def test_canonicalize_yolov5_unaffected(self) -> None:
        from backend.app.workers.training_worker import TrainingWorker
        for variant in ("n", "s", "m", "l", "x"):
            with self.subTest(variant=variant):
                name = f"yolov5{variant}.pt"
                self.assertEqual(TrainingWorker._canonicalize_weights_name(name), name)

    # ------------------------------------------------------------------
    # Resolver: legacy yolov11 input + empty dir + download ON → yolo11 alias
    # ------------------------------------------------------------------

    def test_legacy_yolov11_download_alias_is_canonical(self) -> None:
        """Scenario 8: yolov11m.pt input + empty MODELS_DIR + download ON → alias yolo11m."""
        worker = _make_worker(download_allowed=True)
        with tempfile.TemporaryDirectory(prefix="qc-models-") as tmp:
            with patch("backend.app.workers.training_worker.MODELS_DIR", Path(tmp)):
                result = worker._resolve_weights(_job("yolov11m.pt"))
        self.assertEqual(result.weights_source, "download")
        self.assertEqual(result.resolved_path, "yolo11m")

    def test_canonical_yolo11_download_alias_unchanged(self) -> None:
        """Scenario 9: canonical yolo11m.pt input → alias yolo11m (no double-stripping)."""
        worker = _make_worker(download_allowed=True)
        with tempfile.TemporaryDirectory(prefix="qc-models-") as tmp:
            with patch("backend.app.workers.training_worker.MODELS_DIR", Path(tmp)):
                result = worker._resolve_weights(_job("yolo11m.pt"))
        self.assertEqual(result.weights_source, "download")
        self.assertEqual(result.resolved_path, "yolo11m")

    # ------------------------------------------------------------------
    # Resolver: local cache — backward-compat raw-name fallback
    # ------------------------------------------------------------------

    def test_legacy_filename_in_models_dir_found_via_raw_fallback(self) -> None:
        """Scenario 10: file saved as yolov11m.pt (legacy) is still found in local cache."""
        worker = _make_worker(download_allowed=True)
        with tempfile.TemporaryDirectory(prefix="qc-models-") as tmp:
            fake_dir = Path(tmp)
            (fake_dir / "yolov11m.pt").write_bytes(b"legacy weights")
            with patch("backend.app.workers.training_worker.MODELS_DIR", fake_dir):
                result = worker._resolve_weights(_job("yolov11m.pt"))
        self.assertEqual(result.weights_source, "local_cache")
        self.assertIn("yolov11m.pt", result.resolved_path)

    def test_canonical_filename_in_models_dir_found_via_canonical_path(self) -> None:
        """Scenario 11: file saved as yolo11m.pt (canonical) is found at the preferred path."""
        worker = _make_worker(download_allowed=True)
        with tempfile.TemporaryDirectory(prefix="qc-models-") as tmp:
            fake_dir = Path(tmp)
            (fake_dir / "yolo11m.pt").write_bytes(b"canonical weights")
            with patch("backend.app.workers.training_worker.MODELS_DIR", fake_dir):
                # Input may be the legacy or canonical name — both should find the canonical file
                for input_name in ("yolov11m.pt", "yolo11m.pt"):
                    with self.subTest(input_name=input_name):
                        result = worker._resolve_weights(_job(input_name))
                        self.assertEqual(result.weights_source, "local_cache")
                        self.assertIn("yolo11m.pt", result.resolved_path)

    # ------------------------------------------------------------------
    # Resolver: YOLOv5 non-regression
    # ------------------------------------------------------------------

    def test_yolov5_download_alias_unaffected_by_normalizer(self) -> None:
        """Scenario 12: yolov5s.pt input → alias yolov5s (no canonical remapping)."""
        worker = _make_worker(download_allowed=True)
        with tempfile.TemporaryDirectory(prefix="qc-models-") as tmp:
            with patch("backend.app.workers.training_worker.MODELS_DIR", Path(tmp)):
                result = worker._resolve_weights(_job("yolov5s.pt"))
        self.assertEqual(result.weights_source, "download")
        self.assertEqual(result.resolved_path, "yolov5s")

    # ------------------------------------------------------------------
    # Offline-strict: error for YOLO11 mentions canonical name
    # ------------------------------------------------------------------

    def test_offline_strict_yolo11_error_mentions_canonical_name(self) -> None:
        """Scenario 13: error for legacy input mentions canonical name for clarity."""
        worker = _make_worker(download_allowed=False)
        with tempfile.TemporaryDirectory(prefix="qc-models-") as tmp:
            with patch("backend.app.workers.training_worker.MODELS_DIR", Path(tmp)):
                try:
                    worker._resolve_weights(_job("yolov11m.pt"))
                    self.fail("Expected FileNotFoundError")
                except FileNotFoundError as exc:
                    msg = str(exc)
                    self.assertIn("yolov11m.pt", msg)   # original input preserved
                    self.assertIn("yolo11m.pt", msg)    # canonical shown for clarity


if __name__ == "__main__":
    unittest.main()
