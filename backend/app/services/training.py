from __future__ import annotations

import logging

from backend.app.core.config import AppConfig
from backend.app.core.device_runtime import DeviceRuntimeResolver
from backend.app.repositories.training_repository import TrainingRepository
from backend.app.workers.training_worker import TrainingWorker

logger = logging.getLogger(__name__)


class TrainingService:
    def __init__(
        self,
        training_repo: TrainingRepository,
        models_repo=None,
        device_runtime: DeviceRuntimeResolver | None = None,
        app_config: AppConfig | None = None,
    ) -> None:
        self._training_repo = training_repo
        self._worker = TrainingWorker(
            training_repo,
            models_repo=models_repo,
            device_runtime=device_runtime,
            app_config=app_config,
        )
        self._worker.start()

    def list_jobs(self) -> list[dict]:
        return [self._try_backfill_metrics(job) for job in self._training_repo.list_jobs()]

    def get_job(self, job_id: str) -> dict | None:
        job = self._training_repo.get_job(job_id)
        if job is None:
            return None
        return self._try_backfill_metrics(job)

    def create_job(self, dataset_id: str, base_model: str, params: dict) -> dict:
        return self._training_repo.create_job(dataset_id, base_model, params)

    def cancel_job(self, job_id: str) -> dict:
        return self._training_repo.cancel_job(job_id)

    def delete_job(self, job_id: str) -> dict:
        return self._training_repo.delete_job(job_id)

    def _try_backfill_metrics(self, job: dict) -> dict:
        """Backfill metrics for completed legacy jobs that predate metric persistence.

        Safe to call repeatedly: short-circuits when metrics are already present or
        when results.csv cannot be found/parsed. Never mutates job status.
        """
        if str(job.get("status") or "").strip().lower() != "completed":
            return job
        if job.get("metrics"):  # non-empty dict → already enriched
            return job

        try:
            from backend.app.core.config import MODELS_DIR  # late-bound so tests can reload config
            run_dir = MODELS_DIR / "trained" / "training_runs" / str(job.get("id") or "")
            params = job.get("params") if isinstance(job.get("params"), dict) else {}
            epochs_requested = int(params.get("epochs") or 1)

            metrics, evaluation, epoch_summary = TrainingWorker._extract_run_metrics(
                run_dir, epochs_requested
            )
            if not metrics and not evaluation:
                return job  # nothing to persist

            updated = self._training_repo.update_job(
                job["id"],
                metrics=metrics,
                evaluation=evaluation,
                epoch_summary=epoch_summary,
            )
            logger.info("[training-service] backfilled metrics for legacy job %s", job["id"])
            return updated
        except Exception:  # noqa: BLE001
            # Never break the list/get endpoint due to backfill failure.
            return job

