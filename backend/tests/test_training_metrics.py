"""test_training_metrics.py — focused tests for training metric persistence and epoch safety.

Covers:
  A. _extract_run_metrics parses results.csv correctly (unit).
  B. Completed simulated job persists metrics/evaluation/epoch_summary fields.
  C. Training log contains hparams start line and metric summary line.
  D. Epoch safety assertion fires when epochs_ran > epochs_requested.
  E. Failed/cancelled jobs do not receive metrics fields.
"""
from __future__ import annotations

import atexit
import csv
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: point DATA_ROOT to an isolated temp dir before importing backend.
# ---------------------------------------------------------------------------
_TEST_DATA_ROOT = tempfile.mkdtemp(prefix="qc-suite-metrics-tests-")
atexit.register(lambda: shutil.rmtree(_TEST_DATA_ROOT, ignore_errors=True))
os.environ["QC_SUITE_DATA_ROOT"] = _TEST_DATA_ROOT
os.environ["QC_SUITE_TRAINING_ENGINE_MODE"] = "simulated"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.workers.training_worker import TrainingWorker  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "epoch",
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
]


def _write_results_csv(run_dir: Path, rows: list[dict]) -> Path:
    csv_path = run_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def _sample_row(epoch: int, precision: float = 0.8, recall: float = 0.75,
                map50: float = 0.82, map50_95: float = 0.55) -> dict:
    return {
        "epoch": epoch,
        "train/box_loss": 1.0, "train/cls_loss": 2.0, "train/dfl_loss": 0.4,
        "metrics/precision(B)": precision,
        "metrics/recall(B)": recall,
        "metrics/mAP50(B)": map50,
        "metrics/mAP50-95(B)": map50_95,
        "val/box_loss": 0.9, "val/cls_loss": 1.8, "val/dfl_loss": 0.4,
    }


# ---------------------------------------------------------------------------
# A. Unit tests for _extract_run_metrics
# ---------------------------------------------------------------------------

class ExtractRunMetricsTest(unittest.TestCase):

    def test_extracts_last_epoch_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            _write_results_csv(run_dir, [
                _sample_row(1, precision=0.80, recall=0.75, map50=0.82, map50_95=0.55),
                _sample_row(2, precision=0.90, recall=0.85, map50=0.88, map50_95=0.65),
            ])
            metrics, evaluation, epoch_summary = TrainingWorker._extract_run_metrics(run_dir, epochs_requested=2)

        self.assertAlmostEqual(metrics["precision"], 0.90)
        self.assertAlmostEqual(metrics["accuracy"], 0.90)   # mapped from precision for UI
        self.assertAlmostEqual(metrics["recall"], 0.85)
        self.assertAlmostEqual(metrics["mAP50"], 0.88)
        self.assertAlmostEqual(metrics["map50"], 0.88)       # duplicate key for UI lookup
        self.assertAlmostEqual(metrics["mAP50_95"], 0.65)
        self.assertAlmostEqual(evaluation["val_box_loss"], 0.9)
        self.assertEqual(epoch_summary["epochs_ran"], 2)
        self.assertEqual(epoch_summary["epochs_requested"], 2)
        self.assertFalse(epoch_summary["early_stopped"])

    def test_early_stop_flagged_when_ran_lt_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            _write_results_csv(run_dir, [_sample_row(1)])
            _, _, epoch_summary = TrainingWorker._extract_run_metrics(run_dir, epochs_requested=10)

        self.assertEqual(epoch_summary["epochs_ran"], 1)
        self.assertEqual(epoch_summary["epochs_requested"], 10)
        self.assertTrue(epoch_summary["early_stopped"])

    def test_missing_results_csv_returns_empty_with_zero_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            metrics, evaluation, epoch_summary = TrainingWorker._extract_run_metrics(run_dir, epochs_requested=5)

        self.assertEqual(metrics, {})
        self.assertEqual(evaluation, {})
        self.assertEqual(epoch_summary["epochs_ran"], 0)
        self.assertEqual(epoch_summary["epochs_requested"], 5)

    def test_bom_csv_is_parsed_correctly(self) -> None:
        """Verify utf-8-sig (BOM) encoded CSVs are handled without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            csv_path = run_dir / "results.csv"
            # Write with BOM
            with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerow(_sample_row(1, precision=0.77))

            metrics, _, _ = TrainingWorker._extract_run_metrics(run_dir, epochs_requested=1)

        self.assertAlmostEqual(metrics.get("precision", -1.0), 0.77)


# ---------------------------------------------------------------------------
# D. Epoch safety assertion
# ---------------------------------------------------------------------------

class EpochSafetyTest(unittest.TestCase):

    def test_epoch_summary_flags_overrun_but_does_not_raise(self) -> None:
        """_extract_run_metrics itself just counts rows; the caller raises on overrun."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            _write_results_csv(run_dir, [_sample_row(i) for i in range(1, 11)])  # 10 rows
            _, _, epoch_summary = TrainingWorker._extract_run_metrics(run_dir, epochs_requested=2)

        self.assertEqual(epoch_summary["epochs_ran"], 10)
        self.assertEqual(epoch_summary["epochs_requested"], 2)
        # early_stopped is False when ran > requested (it's an overrun, not early stop)
        self.assertFalse(epoch_summary["early_stopped"])

    def test_safety_assertion_value_error_message(self) -> None:
        """Simulate the guard logic that _run_real_training applies after calling _extract_run_metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            _write_results_csv(run_dir, [_sample_row(i) for i in range(1, 11)])
            _, _, epoch_summary = TrainingWorker._extract_run_metrics(run_dir, epochs_requested=2)

        epochs_ran = epoch_summary["epochs_ran"]
        epochs_requested = epoch_summary["epochs_requested"]

        with self.assertRaises(ValueError) as ctx:
            if epochs_ran > epochs_requested:
                raise ValueError(
                    f"Epoch safety violation: requested {epochs_requested} epoch(s) but "
                    f"results.csv records {epochs_ran} rows. "
                    "A conflicting time= or epochs override may be active."
                )
        self.assertIn("safety violation", str(ctx.exception).lower())
        self.assertIn("10", str(ctx.exception))


# ---------------------------------------------------------------------------
# B & C. Integration tests using simulated training mode
# ---------------------------------------------------------------------------

class SimulatedTrainingMetricsTest(unittest.TestCase):
    """Uses the real TrainingWorker + TrainingRepository in simulated mode."""

    def setUp(self) -> None:
        # Fresh isolated data dir per test to avoid cross-test contamination.
        self._data_dir = tempfile.mkdtemp(prefix="qc-suite-sim-it-")
        os.environ["QC_SUITE_DATA_ROOT"] = self._data_dir

        # Re-import config-dependent objects so they pick up the new DATA_ROOT.
        import importlib
        import backend.app.core.config as _cfg_mod
        importlib.reload(_cfg_mod)
        import backend.app.repositories.base_json as _bj_mod
        importlib.reload(_bj_mod)
        import backend.app.repositories.training_repository as _tr_mod
        importlib.reload(_tr_mod)
        import backend.app.repositories.models_repository as _mr_mod
        importlib.reload(_mr_mod)

        from backend.app.core.config import AppConfig
        from backend.app.repositories.training_repository import TrainingRepository
        from backend.app.repositories.models_repository import ModelsRepository

        self._config = AppConfig()
        self._config.training_engine_mode = "simulated"
        self._training_repo = TrainingRepository()
        self._models_repo = ModelsRepository()
        self._worker = TrainingWorker(
            self._training_repo,
            self._models_repo,
            app_config=self._config,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self._data_dir, ignore_errors=True)

    def _run_until_terminal(self, job: dict, timeout: float = 10.0) -> dict:
        """Process the queue once, then poll until terminal state."""
        self._worker._process_queue()
        deadline = time.time() + timeout
        while time.time() < deadline:
            updated = self._training_repo.get_job(job["id"])
            if updated and updated.get("status") in {"completed", "failed", "cancelled"}:
                return updated
            time.sleep(0.05)
        return self._training_repo.get_job(job["id"]) or {}

    def _create_job(self, epochs: int = 2) -> dict:
        return self._training_repo.create_job(
            dataset_id="test-ds-id",
            base_model="yolov5s",
            params={
                "epochs": epochs,
                "class_names": ["classA", "classB"],
                "base_model_spec": {
                    "id": "yolov5s",
                    "display_name": "YOLOv5 Small",
                    "weights_name": "yolov5s",
                    "runtime": "ultralytics",
                    "task": "detection",
                    "family": "yolov5",
                    "variant": "s",
                },
            },
        )

    # B — metrics fields present on completed job
    def test_completed_job_has_metrics_dict(self) -> None:
        job = self._run_until_terminal(self._create_job())
        self.assertEqual(job.get("status"), "completed", job.get("error"))
        self.assertIsInstance(job.get("metrics"), dict, "metrics field must be a dict")

    def test_completed_job_has_evaluation_dict(self) -> None:
        job = self._run_until_terminal(self._create_job())
        self.assertEqual(job.get("status"), "completed")
        self.assertIsInstance(job.get("evaluation"), dict, "evaluation field must be a dict")

    def test_completed_job_has_epoch_summary(self) -> None:
        job = self._run_until_terminal(self._create_job(epochs=3))
        self.assertEqual(job.get("status"), "completed")
        epoch_summary = job.get("epoch_summary")
        self.assertIsInstance(epoch_summary, dict)
        self.assertEqual(epoch_summary.get("epochs_requested"), 3)
        self.assertIsNotNone(epoch_summary.get("epochs_ran"))

    def test_metrics_keys_match_ui_lookup_names(self) -> None:
        """Ensure at least one of mAP50/map50 is present so UI mAP column is non-empty."""
        job = self._run_until_terminal(self._create_job())
        self.assertEqual(job.get("status"), "completed")
        metrics = job.get("metrics") or {}
        ui_map_keys = {"mAP", "map", "map50", "mAP50", "map_50", "mAP_50", "map_50_95", "mAP_50_95"}
        self.assertTrue(
            ui_map_keys & set(metrics.keys()),
            f"No UI-readable mAP key found in metrics: {list(metrics.keys())}",
        )

    def test_metrics_keys_contain_accuracy_for_ui(self) -> None:
        """Ensure 'accuracy' key is present so UI Accuracy column is non-empty."""
        job = self._run_until_terminal(self._create_job())
        self.assertEqual(job.get("status"), "completed")
        metrics = job.get("metrics") or {}
        self.assertIn("accuracy", metrics, f"metrics dict: {metrics}")

    # C — log content
    def test_log_contains_hparams_start_line(self) -> None:
        job = self._run_until_terminal(self._create_job(epochs=2))
        self.assertEqual(job.get("status"), "completed")
        log: list[str] = job.get("log") or []
        self.assertTrue(
            any("epochs=2" in line for line in log),
            f"No hparams start line found. Log:\n" + "\n".join(log),
        )

    def test_log_contains_metric_summary_on_completion(self) -> None:
        job = self._run_until_terminal(self._create_job())
        self.assertEqual(job.get("status"), "completed")
        log: list[str] = job.get("log") or []
        self.assertTrue(
            any("Summary" in line or "summary" in line for line in log),
            f"No summary line in log:\n" + "\n".join(log),
        )

    def test_log_completion_line_contains_epoch_fraction(self) -> None:
        job = self._run_until_terminal(self._create_job(epochs=4))
        self.assertEqual(job.get("status"), "completed")
        log: list[str] = job.get("log") or []
        # Expect something like "epochs=4/4" in the completion log
        self.assertTrue(
            any("/4" in line for line in log),
            f"No epoch fraction (e.g. 4/4) in log:\n" + "\n".join(log),
        )

    # E — failed/cancelled jobs must not have metrics
    def test_failed_job_does_not_have_metrics(self) -> None:
        job = self._training_repo.create_job(
            dataset_id="fail-ds", base_model="yolov5s", params={"epochs": 1, "class_names": []}
        )
        self._training_repo.transition(job["id"], "running")
        self._training_repo.transition(job["id"], "failed", error="forced failure")
        failed = self._training_repo.get_job(job["id"])
        self.assertIsNone(failed.get("metrics"), "Failed job must not have metrics")
        self.assertIsNone(failed.get("evaluation"))
        self.assertIsNone(failed.get("epoch_summary"))

    def test_cancelled_job_does_not_have_metrics(self) -> None:
        job = self._training_repo.create_job(
            dataset_id="cancel-ds", base_model="yolov5s", params={"epochs": 1, "class_names": []}
        )
        self._training_repo.cancel_job(job["id"])
        cancelled = self._training_repo.get_job(job["id"])
        self.assertIsNone(cancelled.get("metrics"), "Cancelled job must not have metrics")


# ---------------------------------------------------------------------------
# F. Legacy metrics backfill via TrainingService
# ---------------------------------------------------------------------------

class LegacyMetricsBackfillTest(unittest.TestCase):
    """TrainingService._try_backfill_metrics enriches completed jobs on read."""

    def setUp(self) -> None:
        self._data_dir = tempfile.mkdtemp(prefix="qc-suite-backfill-")
        os.environ["QC_SUITE_DATA_ROOT"] = self._data_dir

        import importlib
        import backend.app.core.config as _cfg_mod
        importlib.reload(_cfg_mod)
        import backend.app.repositories.base_json as _bj_mod
        importlib.reload(_bj_mod)
        import backend.app.repositories.training_repository as _tr_mod
        importlib.reload(_tr_mod)

        from backend.app.core.config import AppConfig, MODELS_DIR
        from backend.app.repositories.training_repository import TrainingRepository
        from backend.app.services.training import TrainingService

        self._MODELS_DIR = MODELS_DIR
        self._training_repo = TrainingRepository()

        config = AppConfig()
        config.training_engine_mode = "simulated"
        # Use TrainingService directly; worker started but simulated so no real ops
        self._service = TrainingService(self._training_repo, app_config=config)

    def tearDown(self) -> None:
        shutil.rmtree(self._data_dir, ignore_errors=True)

    def _make_completed_job_without_metrics(self, epochs: int = 2) -> dict:
        """Create a job record that is 'completed' but has no metrics (legacy state)."""
        from backend.app.repositories.training_repository import TrainingRepository
        job = self._training_repo.create_job(
            dataset_id="legacy-ds",
            base_model="yolov5s",
            params={"epochs": epochs, "class_names": ["A", "B"]},
        )
        self._training_repo.transition(job["id"], "running")
        self._training_repo.transition(
            job["id"], "completed",
            trained_model_path="models/trained/legacy.pt",
        )
        completed = self._training_repo.get_job(job["id"])
        # Sanity: no metrics yet
        assert not completed.get("metrics"), "Test setup error: job already has metrics"
        return completed

    def _write_results_csv_for_job(self, job_id: str, rows: list[dict]) -> None:
        run_dir = self._MODELS_DIR / "trained" / "training_runs" / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_results_csv(run_dir, rows)

    def test_backfill_enriches_completed_job_without_metrics(self) -> None:
        job = self._make_completed_job_without_metrics(epochs=2)
        self._write_results_csv_for_job(job["id"], [
            _sample_row(1, precision=0.70),
            _sample_row(2, precision=0.85, recall=0.80, map50=0.82, map50_95=0.60),
        ])

        enriched = self._service.get_job(job["id"])
        self.assertIsInstance(enriched.get("metrics"), dict)
        self.assertAlmostEqual(enriched["metrics"].get("precision", -1), 0.85)
        self.assertAlmostEqual(enriched["metrics"].get("mAP50", -1), 0.82)

    def test_backfill_is_idempotent(self) -> None:
        """Calling get_job twice on a job that already has metrics must not raise or re-write."""
        job = self._make_completed_job_without_metrics()
        self._write_results_csv_for_job(job["id"], [_sample_row(1, precision=0.77)])

        first = self._service.get_job(job["id"])
        second = self._service.get_job(job["id"])
        self.assertEqual(first.get("metrics"), second.get("metrics"))

    def test_backfill_skips_job_without_results_csv(self) -> None:
        """No results.csv → job returned unchanged (no exception)."""
        job = self._make_completed_job_without_metrics()
        result = self._service.get_job(job["id"])
        # metrics should still be absent/falsy since no CSV to parse
        self.assertFalse(result.get("metrics"))

    def test_backfill_does_not_touch_non_completed_jobs(self) -> None:
        job = self._training_repo.create_job(
            dataset_id="active-ds", base_model="yolov5s", params={"epochs": 1}
        )
        self._training_repo.transition(job["id"], "running")
        # Write a CSV anyway; backfill must not run for non-completed jobs
        self._write_results_csv_for_job(job["id"], [_sample_row(1)])
        result = self._service.get_job(job["id"])
        self.assertIsNone(result.get("metrics"), "Non-completed job must not be backfilled")

    def test_list_jobs_enriches_each_completed_legacy_job(self) -> None:
        job_a = self._make_completed_job_without_metrics(epochs=1)
        job_b = self._make_completed_job_without_metrics(epochs=1)
        self._write_results_csv_for_job(job_a["id"], [_sample_row(1, map50=0.91)])
        self._write_results_csv_for_job(job_b["id"], [_sample_row(1, map50=0.72)])

        jobs = self._service.list_jobs()
        by_id = {j["id"]: j for j in jobs}

        enriched_a = by_id.get(job_a["id"], {})
        enriched_b = by_id.get(job_b["id"], {})
        self.assertAlmostEqual((enriched_a.get("metrics") or {}).get("mAP50", -1), 0.91)
        self.assertAlmostEqual((enriched_b.get("metrics") or {}).get("mAP50", -1), 0.72)


# ---------------------------------------------------------------------------
# H. GPU fail-fast — job must fail immediately when device=gpu + CUDA unavailable
# ---------------------------------------------------------------------------

class GpuFailFastTest(unittest.TestCase):
    """Training jobs requesting device=gpu must fail when CUDA is unavailable (fail-fast=True)."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="qc-suite-gpufail-")
        # Use real engine mode to bypass simulated-mode's special fallback reason.
        os.environ["QC_SUITE_TRAINING_ENGINE_MODE"] = "real"
        os.environ["QC_SUITE_GPU_FAIL_FAST"] = "1"
        import importlib
        import backend.app.core.config as _cfg_mod
        importlib.reload(_cfg_mod)
        from backend.app.core.config import AppConfig
        from backend.app.repositories.training_repository import TrainingRepository
        from backend.app.core.device_runtime import DeviceRuntimeResolver
        self._cfg = AppConfig()
        self._repo = TrainingRepository()
        # Build a resolver that reports CUDA unavailable.
        self._resolver = DeviceRuntimeResolver(self._cfg)
        self._resolver._cuda_state = (False, 0, "cuda_unavailable")  # type: ignore[assignment]

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)
        os.environ.pop("QC_SUITE_GPU_FAIL_FAST", None)
        os.environ["QC_SUITE_TRAINING_ENGINE_MODE"] = "simulated"
        import importlib
        import backend.app.core.config as _cfg_mod
        importlib.reload(_cfg_mod)

    def _make_worker(self) -> TrainingWorker:
        return TrainingWorker(
            self._repo,
            device_runtime=self._resolver,
            app_config=self._cfg,
        )

    def test_gpu_job_fails_immediately_when_cuda_unavailable(self) -> None:
        job = self._repo.create_job(
            dataset_id="ds-gpufail",
            base_model="yolov5n",
            params={"epochs": 1, "device_mode": "gpu"},
        )
        worker = self._make_worker()
        worker._training_mode = "real"  # bypass simulated path so fail-fast fires
        worker._run_job(job)

        result = self._repo.get_job(job["id"])
        self.assertEqual(result["status"], "failed")
        error_msg = str(result.get("error") or "").lower()
        self.assertIn("cuda", error_msg)
        # Should NOT contain simulated fallback reason
        self.assertNotIn("simulated", error_msg)

    def test_gpu_job_does_not_fail_when_gpu_fail_fast_disabled(self) -> None:
        os.environ["QC_SUITE_GPU_FAIL_FAST"] = "0"
        import importlib
        import backend.app.core.config as _cfg_mod
        importlib.reload(_cfg_mod)
        from backend.app.core.config import AppConfig
        cfg = AppConfig()
        self.assertFalse(cfg.gpu_fail_fast)

        job = self._repo.create_job(
            dataset_id="ds-gpufail-off",
            base_model="yolov5n",
            params={"epochs": 1, "device_mode": "gpu"},
        )
        worker = TrainingWorker(
            self._repo,
            device_runtime=self._resolver,
            app_config=cfg,
        )
        worker._training_mode = "simulated"  # let it succeed via simulated path
        worker._run_job(job)

        result = self._repo.get_job(job["id"])
        # With fail-fast disabled and simulated mode, it should complete (not fail due to GPU policy)
        self.assertNotEqual(result["status"], "failed")


# ---------------------------------------------------------------------------
# G. Logging toggle — config defaults and quiet mode behaviour
# ---------------------------------------------------------------------------

class LoggingToggleConfigTest(unittest.TestCase):
    """AppConfig log toggle fields default ON (backward-compatible)."""

    def test_access_logs_enabled_defaults_to_true(self) -> None:
        import importlib
        import backend.app.core.config as _cfg_mod
        # Ensure env is not set to 0
        os.environ.pop("QC_SUITE_ACCESS_LOGS_ENABLED", None)
        importlib.reload(_cfg_mod)
        from backend.app.core.config import AppConfig
        self.assertTrue(AppConfig().access_logs_enabled)

    def test_werkzeug_logs_enabled_defaults_to_true(self) -> None:
        import importlib
        import backend.app.core.config as _cfg_mod
        os.environ.pop("QC_SUITE_WERKZEUG_REQUEST_LOGS_ENABLED", None)
        importlib.reload(_cfg_mod)
        from backend.app.core.config import AppConfig
        self.assertTrue(AppConfig().werkzeug_logs_enabled)

    def test_access_logs_disabled_by_env_zero(self) -> None:
        import importlib
        import backend.app.core.config as _cfg_mod
        os.environ["QC_SUITE_ACCESS_LOGS_ENABLED"] = "0"
        importlib.reload(_cfg_mod)
        from backend.app.core.config import AppConfig
        self.assertFalse(AppConfig().access_logs_enabled)
        # Cleanup
        os.environ.pop("QC_SUITE_ACCESS_LOGS_ENABLED", None)

    def test_werkzeug_logs_disabled_by_env_zero(self) -> None:
        import importlib
        import backend.app.core.config as _cfg_mod
        os.environ["QC_SUITE_WERKZEUG_REQUEST_LOGS_ENABLED"] = "0"
        importlib.reload(_cfg_mod)
        from backend.app.core.config import AppConfig
        self.assertFalse(AppConfig().werkzeug_logs_enabled)
        os.environ.pop("QC_SUITE_WERKZEUG_REQUEST_LOGS_ENABLED", None)


class LoggingQuietModeTest(unittest.TestCase):
    """Verify quiet mode suppresses 2xx/3xx and lets >= 400 through."""

    def _make_app(self, *, access_logs_enabled: bool) -> object:
        import importlib
        import backend.app.core.config as _cfg_mod
        importlib.reload(_cfg_mod)
        from backend.app.core.config import AppConfig

        # Build a minimal Flask test app
        from flask import Flask
        from backend.app.core.logging_config import configure_logging

        mini_app = Flask(f"test_quiet_{access_logs_enabled}")
        cfg = AppConfig()
        cfg.access_logs_enabled = access_logs_enabled
        mini_app.config["QC_SUITE"] = cfg
        configure_logging(mini_app)

        @mini_app.get("/ok")
        def _ok():
            from flask import jsonify
            return jsonify({"ok": True}), 200

        @mini_app.get("/err")
        def _err():
            from flask import jsonify
            return jsonify({"error": "bad"}), 404

        return mini_app

    def test_quiet_mode_suppresses_200_log(self) -> None:
        import logging
        app = self._make_app(access_logs_enabled=False)
        with app.test_client() as client:
            with self.assertLogs("qc.access", level=logging.DEBUG) as log_ctx:
                client.get("/err")   # trigger a log so assertLogs doesn't fail
                client.get("/ok")    # this should NOT produce a log
        # Only the /err WARNING should appear; /ok should be absent
        combined = "\n".join(log_ctx.output)
        self.assertIn("/err", combined)
        self.assertNotIn("/ok", combined)

    def test_quiet_mode_keeps_400_log(self) -> None:
        import logging
        app = self._make_app(access_logs_enabled=False)
        with app.test_client() as client:
            with self.assertLogs("qc.access", level=logging.WARNING) as log_ctx:
                client.get("/err")
        combined = "\n".join(log_ctx.output)
        self.assertIn("status=404", combined)
        self.assertIn("WARNING", combined)

    def test_normal_mode_logs_200_as_info(self) -> None:
        import logging
        app = self._make_app(access_logs_enabled=True)
        with app.test_client() as client:
            with self.assertLogs("qc.access", level=logging.INFO) as log_ctx:
                client.get("/ok")
        combined = "\n".join(log_ctx.output)
        self.assertIn("status=200", combined)
        self.assertIn("INFO", combined)


class DatabaseBackendConfigTest(unittest.TestCase):
    """Verify relational backend selection stays explicit and predictable."""

    def _reload_app_config(self):
        import importlib

        import backend.app.core.config as config_module

        importlib.reload(config_module)
        return config_module.AppConfig

    def test_defaults_to_local_without_relational_credentials(self) -> None:
        for key in (
            "QC_SUITE_DATABASE_BACKEND",
            "QC_SUITE_SQL_ENABLED",
            "MSSQL_SERVER",
            "MSSQL_DATABASE",
            "MSSQL_USERNAME",
            "MSSQL_PASSWORD",
            "POSTGRESQL_HOST",
            "POSTGRESQL_DATABASE",
            "POSTGRESQL_USERNAME",
            "POSTGRESQL_PASSWORD",
        ):
            os.environ.pop(key, None)

        AppConfig = self._reload_app_config()
        cfg = AppConfig()
        self.assertEqual(cfg.database_backend, "local")
        self.assertFalse(cfg.sql_enabled)
        self.assertFalse(cfg.postgresql_enabled)

    def test_explicit_postgresql_backend_activates_postgresql(self) -> None:
        os.environ["QC_SUITE_DATABASE_BACKEND"] = "postgresql"
        os.environ["POSTGRESQL_HOST"] = "localhost"
        os.environ["POSTGRESQL_PORT"] = "5432"
        os.environ["POSTGRESQL_DATABASE"] = "qc_suite"
        os.environ["POSTGRESQL_USERNAME"] = "qc_user"
        os.environ["POSTGRESQL_PASSWORD"] = "secret"

        AppConfig = self._reload_app_config()
        cfg = AppConfig()
        self.assertEqual(cfg.database_backend, "postgresql")
        self.assertTrue(cfg.sql_enabled)
        self.assertTrue(cfg.postgresql_enabled)

        for key in (
            "QC_SUITE_DATABASE_BACKEND",
            "POSTGRESQL_HOST",
            "POSTGRESQL_PORT",
            "POSTGRESQL_DATABASE",
            "POSTGRESQL_USERNAME",
            "POSTGRESQL_PASSWORD",
        ):
            os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
