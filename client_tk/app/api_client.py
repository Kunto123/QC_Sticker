from __future__ import annotations

import base64
from functools import lru_cache
import mimetypes
from pathlib import Path
from urllib.parse import quote

import requests


@lru_cache(maxsize=1)
def _local_backend_app():
    from backend.app.main import app as backend_app

    return backend_app


class ApiClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token: str | None = None
        self.session = requests.Session()
        self._local_mode = self._is_local_transport(self.base_url)
        self._local_client = None

    @staticmethod
    def _is_local_transport(base_url: str) -> bool:
        normalized = str(base_url or "").strip().lower()
        return not normalized or normalized.startswith("local://")

    def _get_local_client(self):
        if self._local_client is None:
            self._local_client = _local_backend_app().test_client()
        return self._local_client

    def set_token(self, token: str | None) -> None:
        self.token = token

    def _headers(self, *, json_content_type: bool = True) -> dict[str, str]:
        headers = {"Content-Type": "application/json"} if json_content_type else {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _local_request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        payload: dict | None = None,
        json_content_type: bool = True,
    ):
        client = self._get_local_client()
        response = client.open(
            path,
            method=method,
            query_string=params or {},
            json=payload if json_content_type else None,
            data=payload if not json_content_type else None,
            headers=self._headers(json_content_type=json_content_type),
        )
        return response

    def _request_json(self, method: str, path: str, *, params: dict | None = None, payload: dict | None = None, timeout: int = 20):
        if self._local_mode:
            response = self._local_request(method, path, params=params, payload=payload)
            if not (200 <= response.status_code < 400):
                detail = response.get_data(as_text=True)
                try:
                    parsed = response.get_json(silent=True)
                    if isinstance(parsed, dict):
                        detail = parsed.get("error") or detail
                except Exception:  # noqa: BLE001
                    pass
                raise RuntimeError(f"{response.status_code}: {detail}")
            return response.get_json()

        response = self.session.request(
            method=method,
            url=f"{self.base_url}{path}",
            params=params,
            json=payload,
            headers=self._headers(),
            timeout=timeout,
        )
        if not response.ok:
            detail = response.text
            try:
                detail = response.json().get("error") or detail
            except ValueError:
                pass
            raise RuntimeError(f"{response.status_code}: {detail}")
        return response.json()

    def _request_bytes(self, method: str, path: str, *, params: dict | None = None, payload: dict | None = None, timeout: int = 20) -> bytes:
        if self._local_mode:
            response = self._local_request(method, path, params=params, payload=payload)
            if not (200 <= response.status_code < 400):
                detail = response.get_data(as_text=True)
                try:
                    parsed = response.get_json(silent=True)
                    if isinstance(parsed, dict):
                        detail = parsed.get("error") or detail
                except Exception:  # noqa: BLE001
                    pass
                raise RuntimeError(f"{response.status_code}: {detail}")
            return bytes(response.data)

        response = self.session.request(
            method=method,
            url=f"{self.base_url}{path}",
            params=params,
            json=payload,
            headers=self._headers(),
            timeout=timeout,
        )
        if not response.ok:
            detail = response.text
            try:
                detail = response.json().get("error") or detail
            except ValueError:
                pass
            raise RuntimeError(f"{response.status_code}: {detail}")
        return response.content

    def _get(self, path: str, params: dict | None = None):
        return self._request_json("GET", path, params=params, timeout=15)

    def _post(self, path: str, payload: dict):
        return self._request_json("POST", path, payload=payload, timeout=20)

    def _put(self, path: str, payload: dict, params: dict | None = None):
        return self._request_json("PUT", path, payload=payload, timeout=20, params=params)

    def _patch(self, path: str, payload: dict):
        return self._request_json("PATCH", path, payload=payload, timeout=20)

    def _delete(self, path: str):
        return self._request_json("DELETE", path, timeout=20)

    def login(self, username: str, password: str) -> dict:
        return self._post("/auth/login", {"username": username, "password": password})

    def login_rfid(self, rfid_uid: str) -> dict:
        return self._post("/auth/login", {"rfid_uid": rfid_uid})

    def me(self) -> dict:
        return self._get("/auth/me")

    def logout(self) -> dict:
        return self._post("/auth/logout", {})

    def logout_all(self) -> dict:
        return self._post("/auth/logout-all", {})

    def list_sessions(self) -> list[dict]:
        return self._get("/auth/sessions")

    def list_templates(self) -> list[dict]:
        return self._get("/templates")

    def get_template(self, template_id: int) -> dict:
        return self._get(f"/templates/{template_id}")

    def get_template_version(self, version_id: int) -> dict:
        return self._get(f"/templates/versions/{version_id}")

    def get_runtime_template(self, version_id: int) -> dict:
        return self._get(f"/templates/versions/{version_id}/runtime-template")

    def create_template(self, payload: dict) -> dict:
        return self._post("/templates", payload)

    def update_template(self, template_id: int, payload: dict, update_current_version: bool = False) -> dict:
        return self._put(f"/templates/{template_id}", payload, params={"update_current": str(update_current_version).lower()})

    def delete_template(self, template_id: int) -> dict:
        return self._delete(f"/templates/{template_id}")

    def list_deployments(self) -> list[dict]:
        return self._get("/deployments")

    def deploy_template(self, payload: dict) -> dict:
        return self._post("/deployments", payload)

    def get_active_deployment(self) -> dict:
        return self._get("/deployments/active")

    def update_deployment(self, deployment_id: int, payload: dict) -> dict:
        return self._put(f"/deployments/{deployment_id}", payload)

    def deactivate_deployment(self, deployment_id: int) -> dict:
        return self._delete(f"/deployments/{deployment_id}")

    def create_session(self, payload: dict) -> dict:
        return self._post("/inspection/sessions/start", payload)

    def capture_part_ready_ref(self, template_id: int, frame_b64: str, roi: dict) -> dict:
        return self._post(
            f"/templates/{template_id}/part-ready-ref/capture",
            {"frame_b64": frame_b64, "roi": roi},
        )

    def upload_part_ready_ref(self, template_id: int, file_path: str) -> dict:
        """Upload reference patch image."""
        import mimetypes
        from pathlib import Path
        path = Path(file_path)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with path.open("rb") as fh:
            if self._local_mode:
                # Local mode: read file and send as payload
                file_bytes = fh.read()
                import base64
                return self._post(
                    f"/templates/{template_id}/part-ready-ref/upload",
                    {"file_b64": base64.b64encode(file_bytes).decode("ascii")},
                )
            response = self.session.request(
                method="POST",
                url=f"{self.base_url}/templates/{template_id}/part-ready-ref/upload",
                files={"file": (path.name, fh, content_type)},
                headers=self._headers(json_content_type=False),
                timeout=30,
            )
            if not response.ok:
                detail = response.text
                try:
                    detail = response.json().get("error") or detail
                except ValueError:
                    pass
                raise RuntimeError(f"{response.status_code}: {detail}")
            return response.json()

    def update_roi(self, session_id: str, payload: dict) -> dict:
        return self._post(f"/inspection/sessions/{session_id}/roi", payload)

    def update_rois(
        self,
        session_id: str,
        *,
        part_ready_roi: dict | None = None,
        sticker_roi: dict | None = None,
    ) -> dict:
        payload: dict[str, dict] = {}
        if part_ready_roi is not None:
            payload["part_ready_roi"] = part_ready_roi
        if sticker_roi is not None:
            payload["sticker_roi"] = sticker_roi
        return self._post(f"/inspection/sessions/{session_id}/roi", payload)

    def get_latest_preview(self) -> dict:
        return self._get("/inspection/latest-preview")

    def stop_session(self, session_id: str) -> dict:
        return self._post(f"/inspection/sessions/{session_id}/stop", {})

    def manual_release(self, session_id: str) -> dict:
        return self._post(f"/inspection/sessions/{session_id}/release", {})

    def push_frame(self, session_id: str, image_b64: str, *, response_mode: str | None = None) -> dict:
        payload: dict[str, object] = {"image_b64": image_b64}
        if response_mode:
            payload["response_mode"] = str(response_mode).strip()
        return self._post(f"/inspection/sessions/{session_id}/frame", payload)

    def push_frame_local(
        self,
        session_id: str,
        frame,
        *,
        username: str | None = None,
        user_id: int | None = None,
        response_mode: str | None = None,
    ) -> dict:
        """Local-mode fast path: call process_frame_decoded directly, bypassing encode/HTTP.

        Only valid when _local_mode is True. Falls back to push_frame (with re-encode)
        if something goes wrong importing the backend service.
        """
        from backend.app.core.container import inspection_session_service

        return inspection_session_service.process_frame_decoded(
            session_id,
            frame=frame,
            response_mode=response_mode,
            username=username,
            user_id=user_id,
        )

    def list_inspections(self, params: dict | None = None) -> list[dict]:
        return self._get("/inspections", params)

    def export_inspections_csv(self, params: dict | None = None) -> str:
        """Return raw CSV text from the server export endpoint."""
        if self._local_mode:
            response = self._local_request("GET", "/inspections/export", params=params or {})
            if not (200 <= response.status_code < 400):
                detail = response.get_data(as_text=True)
                raise RuntimeError(f"{response.status_code}: {detail}")
            return response.get_data(as_text=True)

        response = self.session.request(
            method="GET",
            url=f"{self.base_url}/inspections/export",
            params=params or {},
            headers=self._headers(),
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(f"{response.status_code}: {response.text}")
        return response.text

    def get_inspection(self, result_id: int) -> dict:
        return self._get(f"/inspections/{result_id}")

    def update_inspection(self, result_id: int, payload: dict) -> dict:
        return self._patch(f"/inspections/{result_id}", payload)

    def delete_inspection(self, result_id: int) -> dict:
        return self._delete(f"/inspections/{result_id}")

    def retry_inspection_push(self, result_id: int) -> dict:
        return self._post(f"/inspections/{result_id}/retry-push", {})

    def retry_failed_inspection_pushes(self, result_ids: list[int] | None = None, limit: int = 100) -> dict:
        payload = {"limit": limit}
        if result_ids:
            payload["result_ids"] = result_ids
        return self._post("/inspections/retry-push", payload)

    def dashboard_summary(self, params: dict | None = None) -> dict:
        return self._get("/dashboard/summary", params)

    def dashboard_buckets(self, params: dict | None = None) -> list[dict]:
        return self._get("/dashboard/buckets", params)

    def list_users(self) -> list[dict]:
        return self._get("/auth/users")

    def create_user(self, payload: dict) -> dict:
        return self._post("/auth/users", payload)

    def set_user_active(self, user_id: int, is_active: bool) -> dict:
        return self._put(f"/auth/users/{user_id}", {"is_active": is_active})

    def revoke_user_sessions(self, user_id: int) -> dict:
        return self._post(f"/auth/users/{user_id}/revoke-sessions", {})

    def delete_user(self, user_id: int) -> dict:
        return self._delete(f"/auth/users/{user_id}")

    def bind_user_rfid(self, user_id: int, rfid_uid: str) -> dict:
        return self._post(f"/auth/users/{user_id}/rfid", {"rfid_uid": rfid_uid})

    def clear_user_rfid(self, user_id: int) -> dict:
        return self._delete(f"/auth/users/{user_id}/rfid")

    def list_profiles(self) -> list[dict]:
        return self._get("/calibration/profiles")

    def compute_color_profile(self, payload: dict) -> dict:
        return self._post("/calibration/color-profile", payload)

    def save_profile(self, payload: dict) -> dict:
        return self._post("/calibration/profiles", payload)

    def update_profile(self, profile_id: int, payload: dict) -> dict:
        return self._put(f"/calibration/profiles/{profile_id}", payload)

    def delete_profile(self, profile_id: int) -> dict:
        return self._delete(f"/calibration/profiles/{profile_id}")

    def compute_mean_std_threshold(self, empty_b64: str, part_b64: str, sticker_b64: str) -> dict:
        """Send 3 calibration images to compute MEAN_MAX and STD_MAX thresholds."""
        return self._post("/calibration/mean-std-threshold", {
            "empty": empty_b64,
            "part": part_b64,
            "sticker": sticker_b64,
        })

    def list_datasets(self) -> list[dict]:
        return self._get("/datasets")

    def create_dataset(self, payload: dict) -> dict:
        return self._post("/datasets", payload)

    def update_dataset(self, dataset_id: str, payload: dict) -> dict:
        return self._patch(f"/datasets/{dataset_id}", payload)

    def delete_dataset(self, dataset_id: str) -> dict:
        return self._delete(f"/datasets/{dataset_id}")

    def list_dataset_files(self, dataset_id: str, target: str = "images") -> list[dict]:
        return self._get(f"/datasets/{dataset_id}/files", {"target": target})

    def list_dataset_versions(self, dataset_id: str) -> list[dict]:
        return self._get(f"/datasets/{dataset_id}/versions")

    def create_dataset_version(self, dataset_id: str, payload: dict) -> dict:
        return self._post(f"/datasets/{dataset_id}/versions", payload)

    def get_dataset_version(self, dataset_id: str, version_id: str) -> dict:
        return self._get(f"/datasets/{dataset_id}/versions/{version_id}")

    def update_dataset_version(self, dataset_id: str, version_id: str, payload: dict) -> dict:
        return self._put(f"/datasets/{dataset_id}/versions/{version_id}", payload)

    def export_dataset_version(self, dataset_id: str, version_id: str) -> dict:
        return self._post(f"/datasets/{dataset_id}/versions/{version_id}/export", {})

    def upload_dataset_file(self, dataset_id: str, payload: dict) -> dict:
        return self._post(f"/datasets/{dataset_id}/upload", payload)

    def upload_dataset_files(self, dataset_id: str, file_paths: list[str], target: str = "images") -> dict:
        if self._local_mode:
            from backend.app.core.container import datasets_repo

            if not file_paths:
                raise RuntimeError("400: At least one file is required")
            batch = [(Path(file_path).name, Path(file_path).read_bytes()) for file_path in file_paths]
            saved = datasets_repo.save_files(dataset_id, target, batch)
            return {"target": target, "count": len(saved), "items": saved}

        multipart_files = []
        handles = []
        response = None
        try:
            for file_path in file_paths:
                path = Path(file_path)
                handle = path.open("rb")
                handles.append(handle)
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                multipart_files.append(("files", (path.name, handle, content_type)))

            response = self.session.request(
                method="POST",
                url=f"{self.base_url}/datasets/{dataset_id}/upload",
                data={"target": target},
                files=multipart_files,
                headers=self._headers(json_content_type=False),
                timeout=60,
            )
        finally:
            for handle in handles:
                try:
                    handle.close()
                except Exception:
                    pass

        if response is None:
            raise RuntimeError("Upload request did not start")
        if not response.ok:
            detail = response.text
            try:
                detail = response.json().get("error") or detail
            except ValueError:
                pass
            raise RuntimeError(f"{response.status_code}: {detail}")
        return response.json()

    def get_annotation(self, dataset_id: str, image_name: str) -> dict:
        return self._get(f"/datasets/{dataset_id}/annotations/{image_name}")

    def save_annotation(self, dataset_id: str, image_name: str, labels: list[dict]) -> dict:
        return self._post(f"/datasets/{dataset_id}/annotations/{image_name}", {"labels": labels})

    def download_dataset_file(self, dataset_id: str, target: str, file_name: str) -> bytes:
        safe_name = quote(str(Path(file_name).name), safe="")
        return self._request_bytes("GET", f"/datasets/{dataset_id}/files/{target}/{safe_name}", timeout=20)

    def download_dataset_image(self, dataset_id: str, image_name: str) -> bytes:
        return self.download_dataset_file(dataset_id, "images", image_name)

    def get_augment_capabilities(self) -> dict:
        return self._get("/augment/capabilities")

    def list_augment_jobs(self) -> list[dict]:
        return self._get("/augment/jobs")

    def create_augment_job(self, payload: dict) -> dict:
        return self._post("/augment/jobs", payload)

    def delete_augment_job(self, job_id: str) -> dict:
        return self._delete(f"/augment/jobs/{job_id}")

    def list_training_jobs(self) -> list[dict]:
        return self._get("/train/jobs")

    def list_base_models(self, family: str | None = None) -> list[dict]:
        params = {"family": family} if family else None
        return self._get("/train/base-models", params)

    def create_training_job(self, payload: dict) -> dict:
        return self._post("/train/jobs", payload)

    def cancel_training_job(self, job_id: str) -> dict:
        return self._post(f"/train/jobs/{job_id}/cancel", {})

    def delete_training_job(self, job_id: str) -> dict:
        return self._delete(f"/train/jobs/{job_id}")

    def list_models(self) -> list[dict]:
        return self._get("/models")

    def get_model_export_manifest(self, model_id: int) -> dict:
        return self._get(f"/models/{model_id}/export-manifest")

    def export_model_archive(self, model_id: int) -> bytes:
        return self._request_bytes("POST", f"/models/{model_id}/export", timeout=60)

    def import_model_archive(self, archive_path: str, *, target_lifecycle: str = "draft", skip_validation: bool = False, force_rename: bool = False) -> dict:
        archive_bytes = Path(archive_path).read_bytes()
        payload = {
            "content_b64": base64.b64encode(archive_bytes).decode("ascii"),
            "target_lifecycle": target_lifecycle,
            "skip_validation": "1" if skip_validation else "0",
            "force_rename": "1" if force_rename else "0",
        }
        return self._post("/models/import", payload)

    def create_model(self, payload: dict) -> dict:
        return self._post("/models", payload)

    def upload_model_file(self, payload: dict) -> dict:
        return self._post("/models/upload", payload)

    def update_model(self, model_id: int, payload: dict) -> dict:
        return self._patch(f"/models/{model_id}", payload)

    def delete_model(self, model_id: int, *, purge_files: bool = False) -> dict:
        params = {"purge_files": "1"} if purge_files else None
        return self._request_json("DELETE", f"/models/{model_id}", params=params, timeout=20)

    # ------------------------------------------------------------------
    # Template lifecycle
    # ------------------------------------------------------------------

    def list_template_versions(self, template_id: int) -> list[dict]:
        return self._get(f"/templates/{template_id}/versions")

    def transition_template_lifecycle(self, template_id: int, status: str, change_note: str = "") -> dict:
        return self._post(f"/templates/{template_id}/transition", {"status": status, "change_note": change_note})

    def rollback_template_version(self, template_id: int, version_id: int) -> dict:
        return self._post(f"/templates/{template_id}/rollback", {"version_id": version_id})

    def rollback_deployment(self, deployment_id: int) -> dict:
        return self._post(f"/deployments/{deployment_id}/rollback", {})

    # ------------------------------------------------------------------
    # User management extras
    # ------------------------------------------------------------------

    def change_user_role(self, user_id: int, role: str) -> dict:
        return self._post(f"/auth/users/{user_id}/role", {"role": role})

    def reset_user_password(self, user_id: int, password: str) -> dict:
        return self._post(f"/auth/users/{user_id}/reset-password", {"password": password})

    def get_audit_log(self, limit: int = 100, user_id: int | None = None) -> list[dict]:
        params: dict = {"limit": limit}
        if user_id is not None:
            params["user_id"] = user_id
        return self._get("/auth/audit-log", params)

    # ------------------------------------------------------------------
    # Workstation heartbeat
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # PLC / Remote-IO control
    # ------------------------------------------------------------------

    def plc_status(self) -> dict:
        return self._get("/inspection/plc/status")

    def heartbeat(self, machine_id: str, *, client_version: str | None = None,
                  line_id: str | None = None, station_id: str | None = None) -> dict:
        payload: dict = {"machine_id": machine_id}
        if client_version is not None:
            payload["client_version"] = client_version
        if line_id is not None:
            payload["line_id"] = line_id
        if station_id is not None:
            payload["station_id"] = station_id
        return self._post("/workstations/heartbeat", payload)

    def list_workstations(self) -> list[dict]:
        return self._get("/workstations")

    def delete_workstation(self, machine_id: str) -> dict:
        safe_id = quote(machine_id, safe="")
        return self._delete(f"/workstations/{safe_id}")

    # ------------------------------------------------------------------
    # Machine / PLC Settings
    # ------------------------------------------------------------------

    def get_machine_settings(self) -> dict:
        return self._get("/machine-settings")

    def update_machine_settings(self, payload: dict) -> dict:
        return self._put("/machine-settings", payload)

    def seed_machine_settings(self, force: bool = True) -> dict:
        # Pass force as a query param (not embedded in the path) — embedding "?force=1"
        # in the path while the transport also sets query_string raises
        # "Query string is defined in the path and as an argument".
        return self._request_json(
            "POST", "/machine-settings/seed",
            params={"force": "1" if force else "0"}, payload={},
        )

    def get_plc_diagnostics(self) -> dict:
        return self._get("/machine-settings/plc/diagnostics")

    def test_plc_coil(self, address: int, duration_ms: int, confirm: bool = True) -> dict:
        return self._post(
            "/machine-settings/plc/test-coil",
            {"address": address, "duration_ms": duration_ms, "confirm": "yes" if confirm else "no"},
        )

    def plc_all_off(self) -> dict:
        return self._post("/machine-settings/plc/all-off", {})
