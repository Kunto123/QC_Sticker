"""training_worker.py — background daemon that drives training job execution.

State machine:
    queued → running → completed | failed
    Any non-terminal state → cancelled (on user request via cancel_job())

Training mode is configurable via ``QC_SUITE_TRAINING_ENGINE_MODE``:
    - ``real``: execute Ultralytics YOLO training.
    - ``simulated``: keep legacy placeholder behavior for smoke/local fallback.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.core.config import AppConfig, MODELS_DIR, PROJECT_ROOT
from backend.app.core.device_runtime import DeviceResolution, DeviceRuntimeResolver

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 1  # seconds between queue checks


@dataclass
class WeightsResolution:
    """Result of the layered weights resolver."""
    weights_input: str          # raw filename or path from the job
    weights_source: str         # "absolute" | "local_cache" | "download"
    resolved_path: str          # actual path or alias passed to YOLO()
    resolution_attempts: list[str] = field(default_factory=list)
_SIMULATED_TOTAL_SECONDS = 0.8
_SIMULATED_STEP_SECONDS = 0.1


class _TrainingJobCancelled(RuntimeError):
    """Raised when a training job is cancelled while execution is in progress."""


def _safe_fragment(value: str) -> str:
    fragment = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in str(value or "").strip())
    return fragment.strip("._-") or "model"


class TrainingWorker:
    def __init__(
        self,
        training_repo,
        models_repo=None,
        device_runtime: DeviceRuntimeResolver | None = None,
        app_config: AppConfig | None = None,
    ) -> None:
        self._repo = training_repo
        self._models_repo = models_repo
        self._config = app_config or AppConfig()
        self._device_runtime = device_runtime or DeviceRuntimeResolver(self._config)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        mode = str(self._config.training_engine_mode or "real").strip().lower()
        self._training_mode = mode if mode in {"real", "simulated"} else "real"

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="training-worker", daemon=True)
        self._thread.start()
        logger.info("[training-worker] started")

    def stop(self) -> None:
        self._stop_event.set()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._process_queue()
            except Exception:  # noqa: BLE001
                logger.exception("[training-worker] unhandled error in loop")
            self._stop_event.wait(timeout=_POLL_INTERVAL)

    def _process_queue(self) -> None:
        jobs = self._repo.list_jobs()
        for job in jobs:
            if job["status"] == "queued":
                self._run_job(job)

    def _resolve_job_device(self, job: dict) -> DeviceResolution:
        params = job.get("params") if isinstance(job.get("params"), dict) else {}
        requested_device_mode = str(
            job.get("requested_device_mode")
            or params.get("device_mode")
            or "auto"
        ).strip().lower() or "auto"
        if self._training_mode == "simulated":
            fallback_reason = "simulated_mode_forces_cpu" if requested_device_mode != "cpu" else "simulated_mode"
            return DeviceResolution(
                requested_mode=requested_device_mode,
                effective_device="cpu",
                backend="cpu",
                gpu_available=False,
                cuda_device_id=None,
                fallback_reason=fallback_reason,
            )
        return self._device_runtime.resolve(requested_device_mode)

    @staticmethod
    def _device_log_suffix(resolution: DeviceResolution) -> str:
        suffix = f"requested={resolution.requested_mode}, effective={resolution.effective_device}"
        if resolution.fallback_reason:
            suffix += f", fallback={resolution.fallback_reason}"
        return suffix

    @staticmethod
    def _base_model_label(job: dict) -> str:
        return str(
            job.get("base_model_display_name")
            or job.get("base_model_catalog_id")
            or job.get("base_model")
            or "baseline"
        ).strip()

    @staticmethod
    def _dataset_version_label(job: dict) -> str:
        return str(
            job.get("dataset_version_display_label")
            or job.get("dataset_version_name")
            or job.get("dataset_version_id")
            or ""
        ).strip()

    @staticmethod
    def _trained_artifact_path(trained_model_path: str) -> Path:
        relative = Path(str(trained_model_path or "").strip())
        if relative.parts and relative.parts[0].lower() == "models":
            relative = Path(*relative.parts[1:])
        return MODELS_DIR / relative

    def _job_status(self, job_id: str) -> str:
        job = self._repo.get_job(job_id)
        if not isinstance(job, dict):
            return ""
        return str(job.get("status") or "").strip().lower()

    def _raise_if_cancelled(self, job_id: str) -> None:
        if self._job_status(job_id) == "cancelled":
            raise _TrainingJobCancelled("Job cancelled by user.")

    @staticmethod
    def _cleanup_generated_artifacts(artifact_path: Path) -> None:
        for target in (artifact_path, artifact_path.with_suffix(".meta.json")):
            try:
                if target.exists() and target.is_file():
                    target.unlink()
            except OSError:
                logger.warning("[training-worker] failed to remove artifact: %s", target)

    @staticmethod
    def _clamp_progress(value: int) -> int:
        return min(100, max(0, int(value)))

    def _set_progress(
        self,
        job_id: str,
        *,
        percent: int,
        stage: str,
        message: str,
        append_log: bool = False,
    ) -> None:
        self._raise_if_cancelled(job_id)
        payload = {
            "progress_percent": self._clamp_progress(percent),
            "progress_stage": str(stage or "running").strip() or "running",
            "progress_message": str(message or "").strip(),
        }
        log_line = payload["progress_message"] if append_log and payload["progress_message"] else None
        self._repo.update_job(job_id, log_line=log_line, **payload)

    def _resolve_training_source(self, job: dict) -> Path:
        params = job.get("params") if isinstance(job.get("params"), dict) else {}
        export_root_raw = str(
            job.get("dataset_version_export_root")
            or params.get("dataset_version_export_root")
            or ""
        ).strip()
        if not export_root_raw:
            raise ValueError("dataset_version_id with exported dataset is required for real training.")

        export_root = Path(export_root_raw)
        if not export_root.is_absolute():
            export_root = (PROJECT_ROOT / export_root).resolve()
        if not export_root.exists() or not export_root.is_dir():
            raise ValueError(f"Dataset version export directory not found: {export_root}")

        data_yaml = export_root / "data.yaml"
        if not data_yaml.exists() or not data_yaml.is_file():
            raise ValueError(f"Dataset version export is not ready (missing data.yaml): {export_root}")
        self._normalize_data_yaml(data_yaml_path=data_yaml, export_root=export_root)
        return data_yaml

    @staticmethod
    def _normalize_data_yaml(*, data_yaml_path: Path, export_root: Path) -> None:
        """Ensure exported YOLO data.yaml uses an absolute dataset root path.

        Some Ultralytics versions resolve `path: .` relative to the process CWD
        instead of the YAML file location. Rewriting `path` to an absolute export
        root keeps train/val/test lookups stable across runtime environments.
        """
        try:
            original = data_yaml_path.read_text(encoding="utf-8")
        except OSError:
            return

        expected_path_line = f"path: {export_root.resolve().as_posix()}"
        lines = original.splitlines()
        path_index = next(
            (index for index, line in enumerate(lines) if line.strip().startswith("path:")),
            None,
        )

        changed = False
        if path_index is None:
            lines.insert(0, expected_path_line)
            changed = True
        elif lines[path_index].strip() != expected_path_line:
            lines[path_index] = expected_path_line
            changed = True

        if not changed:
            return

        normalized = "\n".join(lines).rstrip() + "\n"
        try:
            data_yaml_path.write_text(normalized, encoding="utf-8")
        except OSError:
            logger.warning("[training-worker] failed to normalize data.yaml path: %s", data_yaml_path)

    @staticmethod
    def _resolve_weights_name(job: dict) -> str:
        raw_value = str(
            job.get("base_model_weights_name")
            or job.get("base_model_catalog_id")
            or job.get("base_model")
            or "yolov5n"
        ).strip()
        if not raw_value:
            raw_value = "yolov5n"
        if not Path(raw_value).suffix:
            raw_value = f"{raw_value}.pt"
        return raw_value

    @staticmethod
    def _canonicalize_weights_name(weights_name: str) -> str:
        """Map legacy ``yolov11*.pt`` names to the canonical Ultralytics ``yolo11*.pt`` format.

        Ultralytics YOLO11 weights use the prefix ``yolo11`` (no ``v``).
        Legacy job records or catalog entries that still carry ``yolov11`` are
        transparently remapped so that download and local-cache lookups use the
        correct filename.  YOLOv5 names are unaffected.
        """
        if re.match(r"^yolov11[a-z0-9]", weights_name, re.IGNORECASE):
            # Replace leading "yolov11" with "yolo11"
            return "yolo11" + weights_name[len("yolov11"):]
        return weights_name

    def _resolve_weights(self, job: dict) -> WeightsResolution:
        """Form an ordered candidate list and return the first viable resolution.

        Resolution order:
        1. Absolute path (if weights_name is absolute and the file exists).
        2. Local cache — canonical name first (``MODELS_DIR/yolo11m.pt``), then
           raw/legacy name (``MODELS_DIR/yolov11m.pt``) when they differ.
        3. Auto-download via canonical alias (``yolo11m``) — only when
           ``training_weights_download_allowed`` is True.

        Raises ``FileNotFoundError`` with an actionable message when all
        candidates are exhausted.
        """
        weights_name = self._resolve_weights_name(job)
        canonical_name = self._canonicalize_weights_name(weights_name)
        attempts: list[str] = []

        # Candidate 1: absolute path supplied explicitly
        candidate = Path(weights_name)
        if candidate.is_absolute():
            attempts.append(str(candidate))
            if candidate.exists() and candidate.is_file():
                return WeightsResolution(
                    weights_input=weights_name,
                    weights_source="absolute",
                    resolved_path=str(candidate),
                    resolution_attempts=attempts,
                )
            raise FileNotFoundError(
                f"Weights file not found at absolute path: {candidate}. "
                "Verify the path is correct and the file exists on this host."
            )

        # Candidate 2a: local cache — canonical name (preferred)
        canonical_local = MODELS_DIR / canonical_name
        attempts.append(str(canonical_local))
        if canonical_local.exists() and canonical_local.is_file():
            return WeightsResolution(
                weights_input=weights_name,
                weights_source="local_cache",
                resolved_path=str(canonical_local),
                resolution_attempts=attempts,
            )

        # Candidate 2b: local cache — raw/legacy name (backward compat)
        if canonical_name != weights_name:
            raw_local = MODELS_DIR / weights_name
            attempts.append(str(raw_local))
            if raw_local.exists() and raw_local.is_file():
                return WeightsResolution(
                    weights_input=weights_name,
                    weights_source="local_cache",
                    resolved_path=str(raw_local),
                    resolution_attempts=attempts,
                )

        # Candidate 3: auto-download via canonical alias (strip .pt)
        canonical_alias = Path(canonical_name).stem  # e.g. "yolo11m"
        if self._config.training_weights_download_allowed:
            attempts.append(f"download:{canonical_alias}")
            return WeightsResolution(
                weights_input=weights_name,
                weights_source="download",
                resolved_path=canonical_alias,
                resolution_attempts=attempts,
            )

        # All candidates exhausted, downloads disabled
        canonical_suffix = f" (canonical: '{canonical_name}')" if canonical_name != weights_name else ""
        raise FileNotFoundError(
            f"Weights file '{weights_name}'{canonical_suffix} not found locally "
            f"and download is disabled (QC_SUITE_TRAINING_WEIGHTS_DOWNLOAD_ALLOWED=0). "
            f"Checked: {', '.join(attempts)}. "
            "Fix: copy the weights file to MODELS_DIR or set "
            "QC_SUITE_TRAINING_WEIGHTS_DOWNLOAD_ALLOWED=1."
        )

    def _resolve_train_hparams(self, job: dict) -> dict[str, Any]:
        params = job.get("params") if isinstance(job.get("params"), dict) else {}

        def _int_value(key: str, default: int, minimum: int, maximum: int) -> int:
            try:
                value = int(params.get(key, default))
            except (TypeError, ValueError):
                value = default
            return min(maximum, max(minimum, value))

        cache_raw = params.get("cache", False)
        if isinstance(cache_raw, bool):
            cache_value = cache_raw
        else:
            cache_value = str(cache_raw or "").strip().lower() in {"1", "true", "yes", "on"}

        return {
            "epochs": _int_value("epochs", self._config.training_default_epochs, 1, 1000),
            "imgsz": _int_value("imgsz", self._config.training_default_imgsz, 64, 2048),
            "batch": _int_value("batch", self._config.training_default_batch, 1, 256),
            "patience": _int_value("patience", self._config.training_default_patience, 1, 500),
            "workers": _int_value("workers", 0, 0, 32),
            "cache": cache_value,
        }

    @staticmethod
    def _extract_class_names_from_model(model) -> list[str]:
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            values: list[str] = []
            for index in sorted(names):
                values.append(str(names[index]))
            return values
        if isinstance(names, list):
            return [str(item) for item in names]
        return []

    @staticmethod
    def _extract_run_metrics(run_dir: Path, epochs_requested: int) -> tuple[dict, dict, dict]:
        """Parse results.csv and return (metrics, evaluation, epoch_summary).

        Column names are stripped of BOM and whitespace for YOLO compatibility.
        Returns empty dicts (with epochs_ran=0) when results.csv is absent or unreadable.
        """
        import csv as _csv

        results_csv = run_dir / "results.csv"
        metrics: dict = {}
        evaluation: dict = {}
        epoch_summary: dict = {
            "epochs_requested": epochs_requested,
            "epochs_ran": 0,
            "early_stopped": False,
        }

        if not results_csv.exists():
            return metrics, evaluation, epoch_summary

        try:
            rows: list[dict] = []
            with results_csv.open(encoding="utf-8-sig") as fh:
                reader = _csv.DictReader(fh)
                for row in reader:
                    rows.append({k.strip(): v.strip() for k, v in row.items()})

            if not rows:
                return metrics, evaluation, epoch_summary

            epochs_ran = len(rows)
            epoch_summary["epochs_ran"] = epochs_ran
            epoch_summary["early_stopped"] = epochs_ran < epochs_requested

            last = rows[-1]

            def _sf(val: object) -> float | None:
                try:
                    return float(val)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return None

            precision = _sf(last.get("metrics/precision(B)"))
            recall    = _sf(last.get("metrics/recall(B)"))
            map50     = _sf(last.get("metrics/mAP50(B)"))
            map50_95  = _sf(last.get("metrics/mAP50-95(B)"))

            if precision is not None:
                metrics["precision"] = precision
                metrics["accuracy"] = precision  # UI "Accuracy" column reads this key
            if recall is not None:
                metrics["recall"] = recall
            if map50 is not None:
                metrics["mAP50"] = map50
                metrics["map50"] = map50
            if map50_95 is not None:
                metrics["mAP50_95"] = map50_95
                metrics["map50_95"] = map50_95

            for csv_col, eval_key in (
                ("val/box_loss", "val_box_loss"),
                ("val/cls_loss", "val_cls_loss"),
                ("val/dfl_loss", "val_dfl_loss"),
            ):
                val = _sf(last.get(csv_col))
                if val is not None:
                    evaluation[eval_key] = val

        except Exception:  # noqa: BLE001
            logger.warning("[training-worker] failed to parse results.csv from: %s", run_dir)

        return metrics, evaluation, epoch_summary

    def _resolve_output_weights_path(self, *, model, run_project: Path, run_name: str) -> Path:
        candidates: list[Path] = []

        trainer = getattr(model, "trainer", None)
        if trainer is not None:
            for attr_name in ("best", "last"):
                raw_path = getattr(trainer, attr_name, None)
                if raw_path:
                    candidates.append(Path(str(raw_path)))

        candidates.extend(
            [
                run_project / run_name / "weights" / "best.pt",
                run_project / run_name / "weights" / "last.pt",
            ]
        )

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        raise FileNotFoundError("Training completed but no YOLO weights file was produced.")

    def _run_real_training(
        self,
        *,
        job_id: str,
        job: dict,
        artifact_path: Path,
        resolution: DeviceResolution,
        timeout_seconds: int,  # noqa: ARG002 — kept for API compatibility; hard stop handled by caller
    ) -> tuple[list[str], dict, dict, dict]:
        from ultralytics import YOLO  # type: ignore

        self._raise_if_cancelled(job_id)
        self._set_progress(job_id, percent=20, stage="preparing", message="Preparing YOLO training inputs.")
        data_yaml_path = self._resolve_training_source(job)
        weights_resolution = self._resolve_weights(job)
        hparams = self._resolve_train_hparams(job)
        run_project = artifact_path.parent / "training_runs"
        run_name = str(job.get("id") or datetime.now(UTC).strftime("%Y%m%d%H%M%S"))

        run_project.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[training-worker] weights resolved: input=%s, source=%s, path=%s",
            weights_resolution.weights_input,
            weights_resolution.weights_source,
            weights_resolution.resolved_path,
        )
        self._repo.update_job(
            job_id,
            resolved_weights_input=weights_resolution.weights_input,
            resolved_weights_source=weights_resolution.weights_source,
            resolution_attempts=weights_resolution.resolution_attempts,
        )
        model = YOLO(weights_resolution.resolved_path, task="detect")

        # Log hparams before training starts so log reflects what was actually requested.
        self._repo.update_job(
            job_id,
            log_line=(
                f"Training started: epochs={hparams['epochs']}, imgsz={hparams['imgsz']}, "
                f"batch={hparams['batch']}, patience={hparams['patience']}, "
                f"workers={hparams['workers']}, weights={weights_resolution.weights_input} "
                f"(source={weights_resolution.weights_source}), "
                f"device={resolution.effective_device}"
            ),
        )

        self._set_progress(job_id, percent=35, stage="training", message="YOLO training is running.")
        # NOTE: `time=` (hours limit) is intentionally omitted — it overrides `epochs` in
        # Ultralytics and caused training to run far more epochs than requested. The elapsed
        # wall-clock guard in _run_job handles the timeout after training returns.
        model.train(
            data=str(data_yaml_path),
            epochs=hparams["epochs"],
            imgsz=hparams["imgsz"],
            batch=hparams["batch"],
            patience=hparams["patience"],
            workers=hparams["workers"],
            cache=hparams["cache"],
            device=resolution.effective_device,
            project=str(run_project),
            name=run_name,
            exist_ok=True,
            verbose=False,
        )
        self._raise_if_cancelled(job_id)
        self._set_progress(job_id, percent=75, stage="validating", message="Validating generated weights artifact.")

        # Extract metrics from results.csv before copying weights.
        run_dir = run_project / run_name
        metrics, evaluation, epoch_summary = self._extract_run_metrics(run_dir, hparams["epochs"])

        # Safety assertion: epochs actually run must not exceed what was requested.
        # Early stopping (epochs_ran < epochs_requested) is acceptable.
        epochs_ran = epoch_summary.get("epochs_ran", 0)
        if epochs_ran > hparams["epochs"]:
            raise ValueError(
                f"Epoch safety violation: requested {hparams['epochs']} epoch(s) but "
                f"results.csv records {epochs_ran} rows. "
                "A conflicting time= or epochs override may be active."
            )

        source_weights = self._resolve_output_weights_path(model=model, run_project=run_project, run_name=run_name)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_weights, artifact_path)
        self._raise_if_cancelled(job_id)

        if not artifact_path.exists() or artifact_path.stat().st_size <= 0:
            raise ValueError(f"Training produced invalid artifact: {artifact_path}")

        # Quality gate: ensure generated model can be loaded before marking job completed.
        validated_model = YOLO(str(artifact_path))
        self._set_progress(job_id, percent=85, stage="validating", message="Extracting class metadata from trained model.")
        class_names = self._extract_class_names_from_model(validated_model)
        if not class_names:
            params = job.get("params") if isinstance(job.get("params"), dict) else {}
            class_names = list(params.get("class_names") or params.get("classes") or [])

        return class_names, metrics, evaluation, epoch_summary

    def _run_simulated_training(
        self, *, job_id: str, artifact_path: Path, epochs_requested: int
    ) -> tuple[dict, dict, dict]:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        self._repo.update_job(
            job_id,
            log_line=f"Training started (simulated): epochs={epochs_requested}",
        )

        # Simulate small work chunks so cancellation can be observed promptly.
        remaining = _SIMULATED_TOTAL_SECONDS
        completed_time = 0.0
        self._set_progress(job_id, percent=20, stage="training", message="Simulated training started.")
        while remaining > 0:
            self._raise_if_cancelled(job_id)
            step = min(_SIMULATED_STEP_SECONDS, remaining)
            time.sleep(step)
            remaining -= step
            completed_time += step
            ratio = min(1.0, completed_time / _SIMULATED_TOTAL_SECONDS)
            progress = 20 + int(round(ratio * 55))
            self._set_progress(job_id, percent=progress, stage="training", message="Simulated training is running.")

        artifact_path.write_bytes(b"QC Suite simulated trained model artifact\n")
        self._set_progress(job_id, percent=80, stage="validating", message="Validating simulated artifact.")
        self._raise_if_cancelled(job_id)

        # Return stub metrics so the job record and UI have non-empty fields.
        metrics = {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "map50": 0.0, "accuracy": 0.0}
        evaluation: dict = {}
        epoch_summary = {
            "epochs_requested": epochs_requested,
            "epochs_ran": epochs_requested,
            "early_stopped": False,
        }
        return metrics, evaluation, epoch_summary

    def _register_trained_model(
        self,
        job: dict,
        trained_model_path: str,
        artifact_path: Path,
        *,
        class_names: list[str],
    ) -> dict | None:
        if self._models_repo is None:
            return None

        meta_path = artifact_path.with_suffix(".meta.json")
        params = job.get("params") if isinstance(job.get("params"), dict) else {}
        meta_path.write_text(
            json.dumps(
                {
                    "training_job_id": job.get("id"),
                    "dataset_id": job.get("dataset_id"),
                    "dataset_version_id": job.get("dataset_version_id"),
                    "base_model": job.get("base_model"),
                    "base_model_display_name": job.get("base_model_display_name"),
                    "trained_model_path": trained_model_path,
                    "training_engine_mode": self._training_mode,
                    "class_names": list(class_names),
                    "training_params": {
                        "epochs": params.get("epochs"),
                        "imgsz": params.get("imgsz"),
                        "batch": params.get("batch"),
                        "patience": params.get("patience"),
                        "workers": params.get("workers"),
                        "cache": params.get("cache"),
                    },
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        name = str(job.get("base_model_display_name") or job.get("base_model") or "trained model").strip()
        suffix = str(job.get("dataset_version_display_label") or job.get("dataset_version_name") or job.get("dataset_id") or "dataset").strip()
        job_id_short = str(job.get("id") or "").replace("-", "")[:8]
        display_name = f"{name} [{suffix}]" if not job_id_short else f"{name} [{suffix}] #{job_id_short}"
        return self._models_repo.add_model(
            display_name,
            str(artifact_path),
            source="training",
            meta_path=str(meta_path),
            runtime=str(job.get("base_model_runtime") or "ultralytics").strip() or "ultralytics",
            task=str(job.get("base_model_task") or "detection").strip() or "detection",
            class_names=list(class_names),
            architecture_family=str(job.get("base_model_family") or "").strip() or None,
            architecture_variant=str(job.get("base_model_variant") or "").strip() or None,
            source_dataset_id=str(job.get("dataset_id") or "").strip() or None,
            training_job_id=str(job.get("id") or "").strip() or None,
        )

    def _run_job(self, job: dict) -> None:
        job_id = job["id"]
        logger.info("[training-worker] starting job %s", job_id)
        resolution = self._resolve_job_device(job)
        base_model_label = self._base_model_label(job)
        base_model_fragment = _safe_fragment(job.get("base_model_catalog_id") or job.get("base_model") or base_model_label)
        dataset_version_label = self._dataset_version_label(job)
        dataset_version_fragment = _safe_fragment(dataset_version_label) if dataset_version_label else ""
        device_fields = {
            "requested_device_mode": resolution.requested_mode,
            "effective_device": resolution.effective_device,
            "device_backend": resolution.backend,
            "device_fallback_reason": resolution.fallback_reason,
        }
        try:
            self._repo.transition(
                job_id,
                "running",
                log_line=(
                    f"Job started at {datetime.now(UTC).isoformat()} ({self._device_log_suffix(resolution)}, "
                    f"base_model={base_model_label}"
                    f"{', dataset_version=' + dataset_version_label if dataset_version_label else ''})"
                ),
                **device_fields,
            )
            self._set_progress(job_id, percent=10, stage="initializing", message="Resolving training runtime and resources.")
        except ValueError:
            return  # already transitioned (race or cancelled)

        artifact_path: Path | None = None
        try:
            # GPU fail-fast: fail the job immediately if GPU was explicitly requested
            # but CUDA is unavailable, rather than silently falling back to CPU.
            _CUDA_FAIL_FAST_REASONS = {"torch_not_installed", "cuda_unavailable", "cuda_device_count_zero"}
            if (
                resolution.requested_mode == "gpu"
                and getattr(self._config, "gpu_fail_fast", True)
                and resolution.fallback_reason in _CUDA_FAIL_FAST_REASONS
            ):
                raise RuntimeError(
                    f"GPU requested (device=gpu) but CUDA is unavailable "
                    f"({resolution.fallback_reason}). "
                    "Set device_mode=auto or device_mode=cpu to allow CPU fallback, "
                    "or verify the server GPU driver/CUDA setup."
                )

            start_time = time.monotonic()
            timeout_seconds = max(60, int(self._config.training_timeout_minutes) * 60)
            path_prefix = f"{job['dataset_id']}__{base_model_fragment}"
            if dataset_version_fragment:
                path_prefix = f"{job['dataset_id']}__{dataset_version_fragment}__{base_model_fragment}"
            job_id_short = str(job_id).replace("-", "")[:8]
            trained_path = f"models/trained/{path_prefix}__{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}__{job_id_short}.pt"
            artifact_path = self._trained_artifact_path(trained_path)
            self._raise_if_cancelled(job_id)

            params = job.get("params") if isinstance(job.get("params"), dict) else {}
            epochs_requested = int(params.get("epochs") or 1)

            if self._training_mode == "simulated":
                metrics, evaluation, epoch_summary = self._run_simulated_training(
                    job_id=job_id, artifact_path=artifact_path, epochs_requested=epochs_requested
                )
                raw_class_names = params.get("class_names") or params.get("classes") or []
                class_names = [str(item) for item in raw_class_names] if isinstance(raw_class_names, list) else []
            else:
                class_names, metrics, evaluation, epoch_summary = self._run_real_training(
                    job_id=job_id,
                    job=job,
                    artifact_path=artifact_path,
                    resolution=resolution,
                    timeout_seconds=timeout_seconds,
                )

            elapsed_seconds = round(time.monotonic() - start_time, 3)
            if elapsed_seconds > timeout_seconds:
                raise TimeoutError(
                    f"Training exceeded timeout ({elapsed_seconds:.2f}s > {timeout_seconds}s)."
                )
            self._raise_if_cancelled(job_id)

            self._set_progress(job_id, percent=90, stage="registering", message="Registering trained model metadata.")
            registered_model = self._register_trained_model(
                job,
                trained_path,
                artifact_path,
                class_names=list(class_names or []),
            )
            self._raise_if_cancelled(job_id)

            # Build a compact metric summary for the log line.
            _summary_parts: list[str] = [
                f"epochs={epoch_summary.get('epochs_ran', '?')}/{epoch_summary.get('epochs_requested', '?')}",
            ]
            if epoch_summary.get("early_stopped"):
                _summary_parts.append("early_stopped=true")
            for _mkey, _mlabel in (("mAP50", "mAP50"), ("precision", "precision"), ("recall", "recall")):
                _mval = metrics.get(_mkey)
                if _mval is not None:
                    _summary_parts.append(f"{_mlabel}={_mval:.4f}")
            _metric_summary = ", ".join(_summary_parts)

            self._repo.transition(
                job_id,
                "completed",
                trained_model_path=trained_path,
                registered_model_id=(registered_model or {}).get("id"),
                metrics=metrics,
                evaluation=evaluation,
                epoch_summary=epoch_summary,
                log_line=(
                    f"Training completed. Model saved to: {trained_path} "
                    f"({self._device_log_suffix(resolution)}, base_model={base_model_label}"
                    f"{', dataset_version=' + dataset_version_label if dataset_version_label else ''}, "
                    f"mode={self._training_mode}, elapsed={elapsed_seconds}s) "
                    f"[Summary: {_metric_summary}]"
                ),
                progress_percent=100,
                progress_stage="completed",
                progress_message="Training completed successfully.",
                **device_fields,
            )
            logger.info("[training-worker] job %s completed", job_id)
        except _TrainingJobCancelled as exc:
            if artifact_path is not None:
                self._cleanup_generated_artifacts(artifact_path)
            try:
                self._repo.update_job(
                    job_id,
                    progress_stage="cancelled",
                    progress_message="Job cancelled while training was running.",
                )
            except ValueError:
                pass
            logger.info("[training-worker] job %s cancelled during execution: %s", job_id, exc)
        except ValueError:
            pass  # job was cancelled mid-run
        except Exception as exc:  # noqa: BLE001
            logger.exception("[training-worker] job %s failed: %s", job_id, exc)
            if artifact_path is not None:
                self._cleanup_generated_artifacts(artifact_path)
            try:
                self._repo.transition(
                    job_id,
                    "failed",
                    error=str(exc),
                    log_line=(
                        f"Training failed: {exc} ({self._device_log_suffix(resolution)}, base_model={base_model_label}"
                        f"{', dataset_version=' + dataset_version_label if dataset_version_label else ''})"
                    ),
                    progress_stage="failed",
                    progress_message=f"Training failed: {exc}",
                    **device_fields,
                )
            except ValueError:
                pass
