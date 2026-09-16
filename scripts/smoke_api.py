from __future__ import annotations

import atexit
import base64
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_DATA_ROOT = Path(tempfile.mkdtemp(prefix="qc-suite-smoke-"))
atexit.register(lambda: shutil.rmtree(TEMP_DATA_ROOT, ignore_errors=True))
os.environ["QC_SUITE_DATA_ROOT"] = str(TEMP_DATA_ROOT)
os.environ["QC_SUITE_STICKER_INFERENCE_MODE"] = "classic"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.factory import create_app


def _login(client, username: str, password: str) -> str:
    response = client.post("/auth/login", json={"username": username, "password": password})
    response.raise_for_status = lambda: None
    payload = response.get_json()
    if response.status_code != 200 or "token" not in payload:
        raise RuntimeError(f"Login failed for {username}: {payload}")
    return str(payload["token"])


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _sample_image_b64() -> str:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (270, 190), (370, 290), (255, 255, 255), thickness=-1)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Failed to encode sample image.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _blank_image_b64() -> str:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Failed to encode sample image.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def main() -> None:
    app = create_app()
    app.testing = True
    client = app.test_client()

    admin_token = _login(client, "admin", "admin123")
    operator_token = _login(client, "operator", "operator123")
    engineer_token = _login(client, "engineer", "engineer123")

    templates_response = client.get("/templates", headers=_headers(operator_token))
    templates = templates_response.get_json()
    if templates_response.status_code != 200 or not templates:
        raise RuntimeError("Template seed is missing.")

    deploy_response = client.post(
        "/deployments",
        json={
            "template_id": 1,
            "template_version_id": 1,
            "line_id": "LINE-A",
            "station_id": "ST-01",
        },
        headers=_headers(admin_token),
    )
    if deploy_response.status_code != 201:
        raise RuntimeError(f"Deployment failed: {deploy_response.get_json()}")

    active_response = client.get(
        "/deployments/active?line_id=LINE-A&station_id=ST-01",
        headers=_headers(operator_token),
    )
    active_payload = active_response.get_json()
    if active_response.status_code != 200 or not active_payload.get("deployment"):
        raise RuntimeError(f"Active deployment lookup failed: {active_payload}")

    session_response = client.post(
        "/inspection/sessions/start",
        json={
            "client_id": "smoke-operator",
            "camera_index": 0,
            "template_version_id": 1,
            "line_id": "LINE-A",
            "station_id": "ST-01",
        },
        headers=_headers(operator_token),
    )
    session_payload = session_response.get_json()
    if session_response.status_code != 201:
        raise RuntimeError(f"Session start failed: {session_payload}")

    frame_response = client.post(
        f"/inspection/sessions/{session_payload['session_id']}/frame",
        json={"image_b64": _sample_image_b64()},
        headers=_headers(operator_token),
    )
    frame_payload = frame_response.get_json()
    if frame_response.status_code != 200:
        raise RuntimeError(f"Frame processing failed: {frame_payload}")
    if frame_payload.get("validation", {}).get("decision") != "ACCEPT":
        raise RuntimeError(f"Unexpected decision: {frame_payload.get('validation')}")
    if not frame_payload.get("count_committed"):
        raise RuntimeError(f"Expected committed count on first part: {frame_payload}")
    if not frame_payload.get("sticker_detection", {}).get("model_path"):
        raise RuntimeError(f"Expected sticker model path metadata in payload: {frame_payload.get('sticker_detection')}")
    validation = frame_payload.get("validation", {})
    for field in ("ocr_status", "ocr_engine", "anchor_offset", "pose_angle"):
        if field not in validation:
            raise RuntimeError(f"Expected OCR/anchor observability field `{field}` in validation: {validation}")

    duplicate_response = client.post(
        f"/inspection/sessions/{session_payload['session_id']}/frame",
        json={"image_b64": _sample_image_b64()},
        headers=_headers(operator_token),
    )
    duplicate_payload = duplicate_response.get_json()
    if duplicate_response.status_code != 200 or duplicate_payload.get("count_committed"):
        raise RuntimeError(f"Duplicate frame should not be recounted: {duplicate_payload}")

    reset_response = client.post(
        f"/inspection/sessions/{session_payload['session_id']}/frame",
        json={"image_b64": _blank_image_b64()},
        headers=_headers(operator_token),
    )
    if reset_response.status_code != 200:
        raise RuntimeError(f"Reset frame failed: {reset_response.get_json()}")

    dataset_response = client.post(
        "/datasets",
        json={"name": "smoke-dataset", "description": "Temporary smoke dataset"},
        headers=_headers(engineer_token),
    )
    if dataset_response.status_code != 201:
        raise RuntimeError(f"Dataset create failed: {dataset_response.get_json()}")

    summary_response = client.get("/dashboard/summary", headers=_headers(admin_token))
    if summary_response.status_code != 200:
        raise RuntimeError(f"Dashboard failed: {summary_response.get_json()}")
    summary_payload = summary_response.get_json()
    if "backend_classic" not in summary_payload or "total_part_ready" not in summary_payload:
        raise RuntimeError(f"Dashboard summary missing Phase 5 fields: {summary_payload}")

    print("Smoke check passed.")
    print(f"Temporary data root: {TEMP_DATA_ROOT}")
    print(f"Dashboard summary: {summary_payload}")


if __name__ == "__main__":
    main()
