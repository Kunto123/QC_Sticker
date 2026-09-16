from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import cv2
from tkinter import ttk

try:
    import tkinter as tk
except Exception:  # noqa: BLE001
    tk = None

from client_tk.app.screens.admin.view import AdminScreen
from client_tk.app.components.live_view import LiveView
from client_tk.app.components.annotation_canvas import AnnotationCanvas
from client_tk.app.components.template_forms import TemplateEditorForm
from client_tk.app.screens.operator.view import OperatorScreen
from client_tk.app.components.scrollable_frame import ScrollableFrame, _dispatch_mousewheel
from client_tk.app.services.session_state import SessionState


class _StubApi:
    def set_token(self, token: str | None):
        return None

    def list_templates(self):
        return [{"id": 1, "name": "QC Line A", "version_id": 1, "version_number": 1}]

    def get_template(self, template_id: int):
        return {
            "id": template_id,
            "version_id": 1,
            "version_number": 1,
            "name": "QC Line A",
            "description": "stub",
            "is_active": True,
            "camera": {"camera_index": 0, "width": 640, "height": 480, "fps": 15},
            "part_ready_roi": {"x": 0.2, "y": 0.2, "w": 0.25, "h": 0.25},
            "sticker_roi": {"x": 0.2, "y": 0.2, "w": 0.6, "h": 0.6},
            "vision": {
                "model_path": "models/dummy.pt",
                "model_meta_path": None,
                "runtime": "ultralytics",
                "conf_threshold": 0.25,
                "stream_fps": 10,
                "inference_fps": 4,
                "imgsz": 640,
                "classes": ["K0W-HB0"],
            },
            "part_ready": {
                "enabled": True,
                "color_profile_id": None,
                "colorspace": "LAB",
                "distance_threshold": None,
                "min_match_ratio": 0.75,
            },
            "sticker": {
                "part_name": "Sample Part",
                "expected_class": "K0W-HB0",
                "line": "LINE-A",
                "enabled": True,
                "validator_mode": "ml_detection",
                "min_roi_confidence": 0.0,
                "min_class_confidence": None,
                "max_offset_x": 80,
                "max_offset_y": 80,
            },
            "persistence": {"write_to_db": True},
            "metadata": {},
        }

    def get_template_version(self, version_id: int):
        payload = self.get_template(1)
        payload["version_id"] = int(version_id)
        payload["version_number"] = int(version_id)
        payload["camera"]["camera_index"] = 1
        payload["sticker"]["part_name"] = "TEST_PART"
        return payload

    def get_active_deployment(self):
        return {
            "deployment": {
                "id": 1,
                "template_id": 1,
                "template_name": "QC Line A",
                "template_version_id": 2,
                "is_active": True,
            }
        }

    def create_session(self, payload: dict):
        return {
            "session_id": "sess-ui-smoke",
            "template_name": "QC Line A",
            "template_version_id": int(payload.get("template_version_id") or 0),
        }

    def update_rois(self, _session_id: str, *, part_ready_roi=None, sticker_roi=None):
        return {"ok": True, "part_ready_roi": part_ready_roi, "sticker_roi": sticker_roi}

    def stop_session(self, _session_id: str):
        return {"ok": True}

    def push_frame(self, _session_id: str, _image_b64: str, *, response_mode: str | None = None):
        return {
            "event_state": "idle",
            "part_ready": {"part_ready": False, "match_ratio": 0.0},
            "validation": {"decision": "REJECT", "detected_class": None},
            "sticker_detection": {"backend": "skipped"},
            "db_write": {"written": False, "reason": "not_committed"},
            "response_mode": response_mode or "full",
            "recent_events": [],
            "counters": {"session_total": 0, "session_accept": 0, "session_reject": 0},
        }

    def heartbeat(self, _machine_id: str, *, client_version=None, line_id=None, station_id=None):
        return {
            "ok": True,
            "client_version": client_version,
            "line_id": line_id,
            "station_id": station_id,
        }

    def list_models(self):
        return []

    def get_model_export_manifest(self, model_id: int):
        return {"model_id": model_id}

    def export_model_archive(self, model_id: int):
        return b"stub-model-archive"

    def import_model_archive(self, archive_path: str, *, target_lifecycle: str = "draft", skip_validation: bool = False, force_rename: bool = False):
        return {
            "status": "success",
            "model_id": 101,
            "imported_name": Path(archive_path).stem,
            "warnings": [],
            "lifecycle_status": target_lifecycle,
        }

    def get_runtime_template(self, version_id: int):
        return {"schema_version": 1, "template_id": f"template-v{version_id}", "name": "QC Line A"}

    def list_dataset_versions(self, dataset_id: str):
        return []

    def create_dataset_version(self, dataset_id: str, payload: dict):
        return {}

    def update_dataset_version(self, dataset_id: str, version_id: str, payload: dict):
        return {"id": version_id, "dataset_id": dataset_id, **payload}

    def export_dataset_version(self, dataset_id: str, version_id: str):
        return {}

    def list_profiles(self):
        return []

    def list_deployments(self):
        return []

    def plc_status(self):
        return {"enabled": False}

    def create_template(self, payload: dict):
        result = dict(payload)
        result["id"] = int(result.get("id") or 11)
        result["version_id"] = 12
        result["version_number"] = 1
        return result

    def update_template(self, template_id: int, payload: dict):
        result = dict(payload)
        result["id"] = template_id
        result["version_id"] = 22
        result["version_number"] = int(result.get("version_number") or 1) + 1
        return result

    def delete_template(self, template_id: int):
        return {"deleted": True, "id": template_id}

    def deploy_template(self, payload: dict):
        return {"id": 31, "is_active": True, "template_name": "QC Line A", **payload}

    def update_deployment(self, deployment_id: int, payload: dict):
        return {"id": deployment_id, **payload}

    def deactivate_deployment(self, deployment_id: int):
        return {"id": deployment_id, "is_active": False}

    def list_users(self):
        return []

    def create_user(self, payload: dict):
        return {"id": 41, "is_active": True, "rfid_bound": False, **payload}

    def bind_user_rfid(self, user_id: int, rfid_uid: str):
        return {"id": user_id, "rfid_bound": True, "rfid_uid_last4": rfid_uid[-4:]}

    def set_user_active(self, user_id: int, is_active: bool):
        return {"id": user_id, "is_active": is_active}

    def revoke_user_sessions(self, user_id: int):
        return {"user_id": user_id, "revoked": 1}

    def clear_user_rfid(self, user_id: int):
        return {"user": {"id": user_id, "rfid_bound": False}, "sessions_revoked": 1}

    def change_user_role(self, user_id: int, role: str):
        return {"id": user_id, "role": role}

    def list_inspections(self, params=None):
        return []

    def export_inspections_csv(self, params=None):
        return "id,decision\n1,ACCEPT\n"
    
    def get_inspection(self, result_id: int):
        return {"id": result_id, "decision": "ACCEPT", "retry_count": 0, "push_status": "sent"}
    
    def update_inspection(self, result_id: int, payload: dict):
        return {"id": result_id, **payload}
    
    def delete_inspection(self, result_id: int):
        return {"deleted": True, "id": result_id}

    def retry_inspection_push(self, result_id: int):
        return {"ok": True, "result": {"id": result_id, "push_status": "sent"}}

    def retry_failed_inspection_pushes(self, result_ids=None, limit: int = 100):
        return {"attempted": 0, "succeeded": 0, "failed": 0, "items": []}

    def dashboard_summary(self, params=None):
        return {}

    def dashboard_buckets(self, params=None):
        return []
    
    def update_profile(self, profile_id: int, payload: dict):
        return {"id": profile_id, **payload}
    
    def delete_profile(self, profile_id: int):
        return {"deleted": True, "id": profile_id}

    def list_datasets(self):
        return []

    def list_dataset_files(self, dataset_id: str, target: str = "images"):
        return []

    def get_annotation(self, dataset_id: str, image_name: str):
        return {"labels": []}

    def save_annotation(self, dataset_id: str, image_name: str, labels: list[dict]):
        return {"labels": labels}

    def delete_dataset(self, dataset_id: str):
        return {"deleted": True, "id": dataset_id}

    def download_dataset_image(self, dataset_id: str, image_name: str):
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        if not ok:
            return b""
        return buffer.tobytes()

    def list_augment_jobs(self):
        return []

    def create_augment_job(self, payload: dict):
        return {"id": "aug-stub", **payload}
    
    def delete_augment_job(self, job_id: str):
        return {"deleted": True, "id": job_id}

    def list_training_jobs(self):
        return []

    def create_training_job(self, payload: dict):
        return {"id": "train-stub", **payload}
    
    def delete_training_job(self, job_id: str):
        return {"deleted": True, "id": job_id}
    
    def update_model(self, model_id: int, payload: dict):
        return {"id": model_id, **payload}

    def delete_model(self, model_id: int, *, purge_files: bool = False):
        return {"deleted": True, "id": model_id, "purge_files": purge_files}
    
    def list_workstations(self):
        return []
    
    def delete_workstation(self, machine_id: str):
        return {"deleted": True, "machine_id": machine_id}

    def list_base_models(self):
        return [
            {"id": "yolov5s", "display_label": "YOLOv5 Small (yolov5s)", "display_name": "YOLOv5 Small", "family": "yolov5", "family_label": "YOLOv5", "variant": "s", "variant_label": "Small", "runtime": "ultralytics", "weights_name": "yolov5s.pt"},
            {"id": "yolov11m", "display_label": "YOLOv11 Medium (yolov11m)", "display_name": "YOLOv11 Medium", "family": "yolov11", "family_label": "YOLOv11", "variant": "m", "variant_label": "Medium", "runtime": "ultralytics", "weights_name": "yolov11m.pt"},
        ]


@unittest.skipIf(tk is None, "Tkinter is not available in this environment")
class UiSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        if self._is_retired_engineer_test():
            self.skipTest("Engineer screen was retired; setup features moved to Admin.")
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk root unavailable: {exc}")
        self.root.withdraw()
        self.api = _StubApi()
        self.state = SessionState(base_url="http://127.0.0.1:8100")
        self.state.user = {"id": 1, "username": "tester"}
        self._async_patchers = [
            mock.patch("client_tk.app.screens.admin.view.run_async", new=self._run_async_sync),
            mock.patch("client_tk.app.screens.admin.view.ApiClient", new=self._engineer_api_client),
        ]
        for patcher in self._async_patchers:
            patcher.start()

    def _is_retired_engineer_test(self) -> bool:
        method_name = getattr(self, "_testMethodName", "")
        if "engineer" in method_name.lower():
            return True
        method = getattr(type(self), method_name, None)
        code = getattr(method, "__code__", None)
        if code is None:
            return False
        constants = [item for item in code.co_consts if isinstance(item, str)]
        return "EngineerScreen" in code.co_names or any("client_tk.app.screens.engineer" in item for item in constants)

    def _engineer_api_client(self, _base_url: str):
        return self.api

    def _run_async_sync(self, widget, func, *, callback=None, args=(), kwargs=None):
        try:
            result = func(*args, **(kwargs or {}))
        except Exception as exc:  # noqa: BLE001
            if callback is not None:
                callback(None, exc)
            return None
        if callback is not None:
            callback(result, None)
        return None

    def _wait_for_engineer_annotation_load(self, screen, *, attempts: int = 6) -> None:
        for _ in range(attempts):
            screen.update_idletasks()
            screen.update()
            if getattr(screen, "annotation_canvas", None) is None:
                continue
            if screen.annotation_canvas._source_frame is not None and screen.annotation_canvas._photo is not None:
                return

    def tearDown(self) -> None:
        for patcher in getattr(self, "_async_patchers", []):
            patcher.stop()
        if getattr(self, "root", None) is not None:
            self.root.update_idletasks()
            self.root.destroy()

    def test_operator_screen_initializes(self) -> None:
        screen = OperatorScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        self.assertTrue(screen.winfo_exists())
        self.assertEqual(str(screen.template_selector["state"]), "readonly")
        self.assertTrue(screen.template_context.get().startswith("Template: QC Line A"))
        self.assertEqual(screen.template_choice.get(), "QC Line A | v1 | version_id=1")
        screen.destroy()

    def test_operator_layout_switches_to_compact(self) -> None:
        screen = OperatorScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        with mock.patch.object(screen, "winfo_width", return_value=1000), mock.patch.object(
            screen.winfo_toplevel(),
            "winfo_width",
            return_value=1000,
        ):
            screen._apply_responsive_layout()
        self.assertTrue(screen._is_compact_layout)
        self.assertEqual(int(screen.action_buttons[0].grid_info()["row"]), 0)
        self.assertEqual(int(screen.action_buttons[1].grid_info()["row"]), 0)
        self.assertEqual(int(screen.template_box.grid_info()["row"]), 1)
        screen.destroy()

    def test_operator_local_roi_view_refresh(self) -> None:
        screen = OperatorScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        frame[:, :] = (20, 90, 180)
        screen.part_ready_roi_x_value.set("0.1")
        screen.part_ready_roi_y_value.set("0.1")
        screen.part_ready_roi_w_value.set("0.2")
        screen.part_ready_roi_h_value.set("0.3")
        screen.sticker_roi_x_value.set("0.4")
        screen.sticker_roi_y_value.set("0.2")
        screen.sticker_roi_w_value.set("0.3")
        screen.sticker_roi_h_value.set("0.4")
        screen._update_local_roi_previews(frame)
        screen.update_idletasks()
        self.assertIsNotNone(screen.main_view._photo)
        self.assertIn("ROIs", screen.display_source.get())
        screen.destroy()

    def test_operator_draws_roi_overlays_on_frame(self) -> None:
        screen = OperatorScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        payload = {
            "part_ready_roi_meta": {"x": 20, "y": 20, "width": 60, "height": 40},
            "sticker_roi_meta": {"x": 100, "y": 10, "width": 50, "height": 50},
            "client_timings": {"frame_width": 200, "frame_height": 100},
            "validation": {"decision": "REJECT", "detected_class": None},
            "part_ready": {"part_ready": False, "match_ratio": 0.0},
            "sticker_detection": {"backend": "skipped", "raw_detection_count": 0},
            "detections": [],
            "event_state": "idle",
        }

        overlay = screen._build_local_detection_overlay(frame, payload)

        self.assertIsNotNone(overlay)
        self.assertGreater(int(np.count_nonzero(overlay[18:24, 18:24])), 0)
        self.assertGreater(int(np.count_nonzero(overlay[8:14, 98:104])), 0)
        screen.destroy()

    def test_operator_load_deployment_keeps_deployment_version(self) -> None:
        screen = OperatorScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        screen.line_value.set("LINE-A")
        screen.station_value.set("ST-01")

        screen._load_deployment()

        self.assertEqual(screen.template_version_value.get(), "2")
        self.assertEqual(screen.line_value.get(), "LINE-A")
        self.assertEqual(screen.station_value.get(), "ST-01")
        screen.destroy()

    def test_operator_poll_ui_survives_tclerror(self) -> None:
        screen = OperatorScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        frame = np.zeros((180, 320, 3), dtype=np.uint8)

        with mock.patch.object(screen, "_update_roi_overlay_preview", side_effect=tk.TclError("image 'pyimage33' doesn't exist")):
            with mock.patch.object(screen.capture, "get_latest_frame", return_value=frame):
                screen._poll_ui()

        self.assertIn("pyimage33", str(screen.state.latest_error or ""))
        self.assertIn("UI render warning", screen.info_var.get())
        screen.destroy()

    def test_operator_poll_ui_uses_second_frame_fetch_for_local_preview(self) -> None:
        screen = OperatorScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        payload = {
            "event_state": "idle",
            "part_ready": {"part_ready": False, "match_ratio": 0.0},
            "validation": {"decision": "REJECT", "detected_class": None},
            "sticker_detection": {"backend": "skipped", "raw_detection_count": 0},
            "db_write": {"written": False, "reason": "not_committed"},
            "recent_events": [],
            "response_mode": "compact",
            "overlay_image_b64": None,
        }

        with screen._lock:
            screen._latest_payload = payload
            screen._latest_error = None

        with mock.patch.object(screen.capture, "get_latest_frame", side_effect=[None, frame, frame, frame]):
            screen._poll_ui()

        self.assertIsNotNone(screen.main_view._photo)
        screen.destroy()

    def test_admin_screen_initializes(self) -> None:
        screen = AdminScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        self.assertTrue(screen.winfo_exists())
        self.assertEqual(screen._notebook.tabs(), ["Templates", "Data", "Training", "Models", "Calibration", "Operators", "Monitor"])
        self.assertNotIn("Raw JSON", screen._notebook.tabs())
        self.assertNotIn("Engineer", screen._notebook.tabs())
        self.assertFalse(hasattr(screen, "template_form"))
        self.assertFalse(hasattr(screen, "workstation_tools_screen"))
        screen.destroy()

    def test_admin_annotation_flow_loads_image_saves_and_updates_coverage(self) -> None:
        datasets = [{"id": "ds-1", "name": "Sticker Dataset", "image_count": 2, "annotated_image_count": 0}]
        files = [{"name": "one.png", "path": ""}, {"name": "two.png", "path": ""}]
        saved: list[tuple[str, str, list[dict]]] = []

        def save_annotation(dataset_id: str, image_name: str, labels: list[dict]):
            saved.append((dataset_id, image_name, list(labels)))
            return {"labels": labels}

        with mock.patch.object(self.api, "list_datasets", return_value=datasets), mock.patch.object(
            self.api,
            "list_dataset_files",
            return_value=files,
        ), mock.patch.object(self.api, "save_annotation", side_effect=save_annotation):
            screen = AdminScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            screen._notebook.set("Data")
            screen._admin_dataset_list.selection_set(0)
            screen._on_admin_dataset_selected()
            self.assertEqual(screen._admin_annot_image_listbox.size(), 2)

            screen._admin_annot_image_listbox.selection_set(0)
            screen._on_admin_annot_image_list_selected()
            self.assertEqual(screen._admin_annot_image_var.get(), "one.png")

            label = {"type": "bbox", "class_name": "object", "bbox": [0.1, 0.1, 0.5, 0.5]}
            screen._admin_annotation_canvas.set_labels([label])
            screen._admin_save_current_annotation_interactive()

            self.assertEqual(saved[-1], ("ds-1", "one.png", [label]))
            self.assertIn("Coverage: 1/2", screen._admin_annot_coverage_var.get())
            screen.destroy()

    def test_admin_augment_estimator_tracks_dataset_and_multiplier(self) -> None:
        datasets = [{"id": "ds-1", "name": "Sticker Dataset", "image_count": 5, "annotated_image_count": 5, "annotation_coverage": 1.0}]
        with mock.patch.object(self.api, "list_datasets", return_value=datasets):
            screen = AdminScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            display = screen._dataset_id_to_display["ds-1"]
            screen._admin_train_dataset_var.set(display)
            screen._on_admin_train_dataset_selected()
            self.assertIn("Source: 5 imgs", screen._admin_aug_estimator_var.get())

            screen._admin_aug_multiplier_var.set("3")
            self.assertIn("multiplier=3", screen._admin_aug_estimator_var.get())
            self.assertIn("Total aug: 30", screen._admin_aug_estimator_var.get())
            screen.destroy()

    def test_admin_preset_selection_loads_wizard(self) -> None:
        deployments = [
            {
                "id": 7,
                "template_id": 1,
                "template_name": "QC Line A",
                "template_version_id": 2,
                "line_id": "LINE-A",
                "station_id": "ST-01",
                "is_active": True,
            }
        ]
        with mock.patch.object(self.api, "list_deployments", return_value=deployments):
            screen = AdminScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            preset_iid = screen.preset_table.get_children()[0]
            screen.preset_table.selection_set(preset_iid)
            screen.preset_table.focus(preset_iid)
            screen.preset_table.event_generate("<<TreeviewSelect>>")

            self.assertEqual(screen.current_template_id, 1)
            self.assertEqual(screen.current_template_version_id, 2)
            self.assertEqual(screen.preset_name_var.get(), "QC Line A")
            self.assertEqual(screen.preset_line_var.get(), "LINE-A")
            self.assertEqual(screen.preset_station_var.get(), "ST-01")
            self.assertEqual(screen.preset_conf_threshold_var.get(), "0.25")
            screen.destroy()

    def test_admin_preset_library_shows_inactive_templates(self) -> None:
        deployments = [
            {
                "id": 7,
                "template_id": 1,
                "template_name": "QC Line A",
                "template_version_id": 2,
                "line_id": "LINE-A",
                "station_id": "ST-01",
                "is_active": True,
            }
        ]
        templates = [
            {"id": 1, "name": "QC Line A", "version_id": 2, "lifecycle_status": "published"},
            {"id": 2, "name": "QC Line B", "version_id": 5, "lifecycle_status": "published"},
        ]
        with mock.patch.object(self.api, "list_deployments", return_value=deployments), mock.patch.object(
            self.api,
            "list_templates",
            return_value=templates,
        ):
            screen = AdminScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            rows = screen.preset_table.get_children()
            self.assertEqual(rows, ("dep:7", "tpl:2"))
            self.assertEqual(screen.preset_table.item("dep:7", "values")[5], "ACTIVE")
            self.assertEqual(screen.preset_table.item("tpl:2", "values")[3], "QC Line B")
            screen.destroy()

    def test_admin_preset_delete_allows_inactive_template_rows(self) -> None:
        templates = [
            {"id": 2, "name": "QC Line B", "version_id": 5, "lifecycle_status": "draft"},
        ]
        deleted_ids: list[int] = []

        def delete_template(template_id: int):
            deleted_ids.append(template_id)
            return {"deleted": True, "id": template_id}

        with mock.patch.object(self.api, "list_deployments", return_value=[]), mock.patch.object(
            self.api,
            "list_templates",
            return_value=templates,
        ), mock.patch.object(
            self.api,
            "delete_template",
            side_effect=delete_template,
        ), mock.patch(
            "client_tk.app.screens.admin.view.messagebox.askyesno",
            return_value=True,
        ):
            screen = AdminScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            preset_iid = screen.preset_table.get_children()[0]
            self.assertEqual(preset_iid, "tpl:2")
            screen.preset_table.selection_set(preset_iid)
            screen.preset_table.focus(preset_iid)
            screen.preset_table.event_generate("<<TreeviewSelect>>")
            self.assertEqual(screen.current_template_id, 2)

            with mock.patch.object(screen, "refresh_presets") as refresh_mock:
                screen.deactivate_selected_preset()

            self.assertEqual(deleted_ids, [2])
            refresh_mock.assert_called_once()
            self.assertEqual(screen.preset_table.selection(), ())
            self.assertIsNone(screen.current_template_id)
            screen.destroy()

    def test_admin_new_preset_clears_selection_and_creates_template(self) -> None:
        deployments = [
            {
                "id": 7,
                "template_id": 1,
                "template_name": "QC Line A",
                "template_version_id": 2,
                "line_id": "LINE-A",
                "station_id": "ST-01",
                "is_active": True,
            }
        ]
        created_payloads: list[dict] = []
        updated_payloads: list[dict] = []

        def create_template(payload: dict):
            created_payloads.append(dict(payload))
            result = dict(payload)
            result["id"] = 99
            result["version_id"] = 100
            return result

        def update_template(template_id: int, payload: dict):
            updated_payloads.append({"template_id": template_id, **payload})
            result = dict(payload)
            result["id"] = template_id
            result["version_id"] = 101
            return result

        with mock.patch.object(self.api, "list_deployments", return_value=deployments), mock.patch.object(
            self.api,
            "create_template",
            side_effect=create_template,
        ), mock.patch.object(
            self.api,
            "update_template",
            side_effect=update_template,
        ), mock.patch("client_tk.app.screens.admin.view.messagebox.showinfo", return_value=None):
            screen = AdminScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            preset_iid = screen.preset_table.get_children()[0]
            screen.preset_table.selection_set(preset_iid)
            screen.preset_table.focus(preset_iid)
            screen.preset_table.event_generate("<<TreeviewSelect>>")
            self.assertEqual(screen.current_template_id, 1)

            screen.reset_preset_wizard()
            self.assertEqual(screen.current_template_id, None)
            self.assertEqual(screen.preset_table.selection(), ())

            screen.preset_name_var.set("Preset Baru")
            screen.preset_line_var.set("LINE-A")
            screen.preset_station_var.set("ST-01")
            screen.preset_model_path_var.set("data/models/sticker.pt")
            screen.preset_expected_class_var.set("K0W-HB0")

            screen.save_and_deploy_preset()

            self.assertTrue(created_payloads)
            self.assertFalse(updated_payloads)
            self.assertEqual(screen.current_template_id, 99)
            screen.destroy()

    def test_template_form_model_path_dropdown_autofills_meta_path(self) -> None:
        form = TemplateEditorForm(self.root)
        form.pack()
        form.update_idletasks()

        form.set_model_options(
            [
                {
                    "id": 9,
                    "name": "Model A",
                    "runtime": "ultralytics",
                    "path": "data\\models\\akh.pt",
                    "meta_path": "data\\models\\akh.meta.json",
                    "class_names": ["K0W-HB0"],
                }
            ]
        )

        available_paths = [str(item) for item in form.model_path_selector.cget("values")]
        self.assertIn("data\\models\\akh.pt", available_paths)

        form.model_path_var.set("data\\models\\akh.pt")
        form._on_model_path_selected()
        self.assertEqual(form.model_meta_path_var.get(), "data\\models\\akh.meta.json")
        self.assertEqual(form.model_runtime_var.get(), "ultralytics")
        form.destroy()

    def test_admin_layout_switches_to_compact(self) -> None:
        screen = AdminScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        with mock.patch.object(screen, "winfo_width", return_value=1000), mock.patch.object(
            screen.winfo_toplevel(),
            "winfo_width",
            return_value=1000,
        ):
            screen._apply_responsive_layout()
        self.assertTrue(screen._layout_compact)
        self.assertEqual(int(screen.presets_left.grid_info()["row"]), 0)
        self.assertEqual(int(screen.presets_right.grid_info()["row"]), 1)
        self.assertEqual(int(screen.operators_left.grid_info()["row"]), 0)
        self.assertEqual(int(screen.operators_right.grid_info()["row"]), 1)
        self.assertEqual(int(screen.admin_cards["presets"].grid_info()["row"]), 0)
        self.assertEqual(int(screen.admin_cards["accept"].grid_info()["row"]), 1)
        screen.destroy()

    def test_admin_overview_cards_hide_on_short_height(self) -> None:
        screen = AdminScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        with mock.patch.object(screen, "winfo_height", return_value=700):
            screen._apply_responsive_layout()
        self.assertFalse(screen._overview_cards_visible)
        screen.destroy()

    def test_admin_save_and_deploy_preset_without_manual_version_id(self) -> None:
        created_payloads: list[dict] = []
        deployed_payloads: list[dict] = []

        def create_template(payload: dict):
            created_payloads.append(dict(payload))
            result = dict(payload)
            result["id"] = 99
            result["version_id"] = 100
            return result

        def deploy_template(payload: dict):
            deployed_payloads.append(dict(payload))
            return {"id": 101, "is_active": True, **payload}

        with mock.patch.object(self.api, "create_template", side_effect=create_template), mock.patch.object(
            self.api,
            "deploy_template",
            side_effect=deploy_template,
        ), mock.patch("client_tk.app.screens.admin.view.messagebox.showinfo", return_value=None):
            screen = AdminScreen(self.root, self.api, self.state)
            screen.preset_name_var.set("Preset A")
            screen.preset_line_var.set("LINE-A")
            screen.preset_station_var.set("ST-01")
            screen.preset_model_path_var.set("data/models/sticker.pt")
            screen.preset_conf_threshold_var.set("0.42")
            screen.preset_expected_class_var.set("K0W-HB0")

            screen.save_and_deploy_preset()

            self.assertTrue(created_payloads)
            self.assertTrue(deployed_payloads)
            self.assertEqual(created_payloads[-1]["vision"]["runtime"], "ultralytics")
            self.assertEqual(created_payloads[-1]["vision"]["conf_threshold"], 0.42)
            self.assertEqual(created_payloads[-1]["sticker"]["ocr_expected_text"], "K0W-HB0")
            self.assertEqual(deployed_payloads[-1]["template_id"], 99)
            self.assertEqual(deployed_payloads[-1]["template_version_id"], 100)
            self.assertEqual(deployed_payloads[-1]["line_id"], "LINE-A")
            screen.destroy()

    def test_admin_visual_roi_picker_updates_preset_payload(self) -> None:
        screen = AdminScreen(self.root, self.api, self.state)
        screen.preset_roi_choice_var.set("Sticker ROI")
        screen._on_preset_roi_selected()
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        screen.preset_roi_picker.load_image(frame)
        self.root.update_idletasks()

        canvas = screen.preset_roi_picker._canvas
        canvas.update_idletasks()
        screen.preset_roi_picker._on_press(SimpleNamespace(x=170, y=120))
        screen.preset_roi_picker._on_drag(SimpleNamespace(x=210, y=145))
        screen.preset_roi_picker._on_release(SimpleNamespace(x=210, y=145))

        self.assertNotEqual(screen.sticker_roi_x_var.get(), "0.2")
        self.assertNotEqual(screen.sticker_roi_y_var.get(), "0.2")

        screen._reset_preset_roi()
        self.assertEqual(screen.sticker_roi_x_var.get(), "0.2")
        self.assertEqual(screen.sticker_roi_y_var.get(), "0.2")
        self.assertEqual(screen.sticker_roi_w_var.get(), "0.6")
        self.assertEqual(screen.sticker_roi_h_var.get(), "0.6")
        screen.destroy()

    def test_operator_in2_cycles_template_dropdown(self) -> None:
        templates = [
            {"id": 1, "name": "QC Line A", "version_id": 1, "version_number": 1, "lifecycle_status": "published"},
            {"id": 2, "name": "QC Line B", "version_id": 2, "version_number": 2, "lifecycle_status": "published"},
        ]

        def get_template(template_id: int):
            payload = _StubApi.get_template(self.api, template_id)
            payload["id"] = template_id
            payload["name"] = f"QC Line {'A' if template_id == 1 else 'B'}"
            payload["version_id"] = template_id
            payload["version_number"] = template_id
            payload["sticker"]["line"] = f"LINE-{'A' if template_id == 1 else 'B'}"
            payload["sticker"]["station"] = f"ST-0{template_id}"
            return payload

        with mock.patch.object(self.api, "list_templates", return_value=templates), mock.patch.object(
            self.api,
            "get_template",
            side_effect=get_template,
        ):
            screen = OperatorScreen(self.root, self.api, self.state)
            first_label = screen.template_selector.cget("values")[0]
            screen.template_choice.set(first_label)
            screen.state.active_deployment = None
            screen.template_version_value.set("1")
            screen.line_value.set("LINE-A")
            screen.station_value.set("ST-01")
            with mock.patch.object(screen, "_restart_camera_for_template_change", return_value=True), mock.patch.object(
                screen,
                "_restart_session_after_template_change",
                return_value=None,
            ):
                screen._last_plc_template_cycle_event_id = 1
                screen._handle_plc_template_cycle_event({"template_cycle_event_id": 2})

            self.assertIsNone(screen.state.active_deployment)
            self.assertEqual(screen.template_choice.get(), screen.template_selector.cget("values")[1])
            self.assertEqual(screen.template_version_value.get(), "2")
            self.assertEqual(screen.line_value.get(), "LINE-B")
            self.assertEqual(screen.station_value.get(), "ST-02")
            screen.destroy()

    def test_admin_quick_add_operator_creates_user_and_binds_rfid(self) -> None:
        created_payloads: list[dict] = []
        bound_calls: list[tuple[int, str]] = []

        def create_user(payload: dict):
            created_payloads.append(dict(payload))
            return {"id": 77, **payload}

        def bind_user_rfid(user_id: int, rfid_uid: str):
            bound_calls.append((user_id, rfid_uid))
            return {"id": user_id, "rfid_bound": True}

        with mock.patch.object(self.api, "create_user", side_effect=create_user), mock.patch.object(
            self.api,
            "bind_user_rfid",
            side_effect=bind_user_rfid,
        ):
            screen = AdminScreen(self.root, self.api, self.state)

            # Step 1: create user (RFID binding is now separate)
            screen.operator_username_var.set("operator-a")
            with mock.patch.object(screen, "refresh_operators"):
                screen._on_save_user()

            # After save, bind target is auto-set to the new user
            self.assertEqual(screen.bind_target_user_id, 77)

            # Step 2: scan RFID and bind via unified bind section
            screen.unified_rfid_var.set("RFID-1234")
            with mock.patch.object(screen, "refresh_operators"):
                screen._on_bind_rfid()

            self.assertEqual(created_payloads[-1]["username"], "operator-a")
            self.assertEqual(created_payloads[-1]["role"], "operator")
            self.assertTrue(created_payloads[-1]["password"])
            self.assertEqual(bound_calls, [(77, "RFID-1234")])
            screen.destroy()

    def test_admin_monitor_retries_visible_failed_pushes(self) -> None:
        results = [
            {
                "id": 33,
                "inspected_at": "2026-04-10T10:00:00+00:00",
                "decision": "REJECT",
                "part_name": "Part-B",
                "line_id": "LINE-B",
                "station_id": "ST-02",
                "push_status": "failed",
                "retry_count": 0,
                "reject_reason_code": "OFFSET",
            }
        ]
        retry_calls: list[list[int]] = []

        def retry_failed_inspection_pushes(result_ids=None, limit: int = 100):
            retry_calls.append(list(result_ids or []))
            return {"attempted": len(result_ids or []), "succeeded": len(result_ids or []), "failed": 0}

        with mock.patch.object(self.api, "list_inspections", return_value=results), mock.patch.object(
            self.api,
            "retry_failed_inspection_pushes",
            side_effect=retry_failed_inspection_pushes,
        ):
            screen = AdminScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            screen.retry_visible_failed_pushes()

            self.assertEqual(retry_calls, [[33]])
            screen.destroy()

    def test_engineer_support_tab_loads_user_and_inspection_tools(self) -> None:
        users = [
            {"id": 5, "username": "operator-a", "role": "operator", "is_active": True, "rfid_bound": True, "rfid_uid_last4": "1234"},
        ]
        inspections = [
            {
                "id": 77,
                "inspected_at": "2026-04-10T10:00:00+00:00",
                "decision": "REJECT",
                "decision_code": "REJECT",
                "part_name": "Part-X",
                "line_id": "LINE-A",
                "station_id": "ST-01",
                "push_status": "failed",
                "reject_reason_code": "OFFSET",
                "part_ready_match_ratio": 0.42,
                "sticker_confidence": 0.73,
                "detected_class": "BAD",
                "expected_class": "GOOD",
            }
        ]
        created_payloads: list[dict] = []
        bind_calls: list[tuple[int, str]] = []
        role_calls: list[tuple[int, str]] = []
        active_calls: list[tuple[int, bool]] = []
        revoke_calls: list[int] = []
        clear_calls: list[int] = []
        update_calls: list[tuple[int, dict]] = []
        retry_calls: list[int] = []
        retry_failed_calls: list[list[int]] = []
        delete_calls: list[int] = []

        def create_user(payload: dict):
            created_payloads.append(dict(payload))
            return {"id": 88, **payload}

        def bind_user_rfid(user_id: int, rfid_uid: str):
            bind_calls.append((user_id, rfid_uid))
            return {"id": user_id, "rfid_bound": True}

        def change_user_role(user_id: int, role: str):
            role_calls.append((user_id, role))
            return {"id": user_id, "role": role}

        def set_user_active(user_id: int, is_active: bool):
            active_calls.append((user_id, is_active))
            return {"id": user_id, "is_active": is_active}

        def revoke_user_sessions(user_id: int):
            revoke_calls.append(user_id)
            return {"user_id": user_id, "revoked": 1}

        def clear_user_rfid(user_id: int):
            clear_calls.append(user_id)
            return {"user": {"id": user_id, "rfid_bound": False}, "sessions_revoked": 1}

        def update_inspection(result_id: int, payload: dict):
            update_calls.append((result_id, dict(payload)))
            return {"id": result_id, **payload}

        def retry_inspection_push(result_id: int):
            retry_calls.append(result_id)
            return {"ok": True, "result": {"id": result_id, "push_status": "sent"}}

        def retry_failed_inspection_pushes(result_ids=None, limit: int = 100):
            retry_failed_calls.append(list(result_ids or []))
            return {"attempted": len(result_ids or []), "succeeded": len(result_ids or []), "failed": 0, "items": []}

        def delete_inspection(result_id: int):
            delete_calls.append(result_id)
            return {"deleted": True, "id": result_id}

        with mock.patch.object(self.api, "list_users", return_value=users), mock.patch.object(
            self.api,
            "list_inspections",
            return_value=inspections,
        ), mock.patch.object(self.api, "create_user", side_effect=create_user), mock.patch.object(
            self.api,
            "bind_user_rfid",
            side_effect=bind_user_rfid,
        ), mock.patch.object(self.api, "change_user_role", side_effect=change_user_role), mock.patch.object(
            self.api,
            "set_user_active",
            side_effect=set_user_active,
        ), mock.patch.object(self.api, "revoke_user_sessions", side_effect=revoke_user_sessions), mock.patch.object(
            self.api,
            "clear_user_rfid",
            side_effect=clear_user_rfid,
        ), mock.patch.object(self.api, "update_inspection", side_effect=update_inspection), mock.patch.object(
            self.api,
            "retry_inspection_push",
            side_effect=retry_inspection_push,
        ), mock.patch.object(self.api, "retry_failed_inspection_pushes", side_effect=retry_failed_inspection_pushes), mock.patch.object(
            self.api,
            "delete_inspection",
            side_effect=delete_inspection,
        ), mock.patch("client_tk.app.screens.engineer.view.messagebox.askyesno", return_value=True):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            self.assertEqual(screen._notebook.tabs(), ["Data", "Training", "Models", "Calibration", "Support"])
            screen.select_tab("Support")
            screen.update_idletasks()

            self.assertTrue(screen.support_users_list.size() >= 1)
            self.assertTrue(screen.support_results_list.size() >= 1)

            screen.refresh_support_users = lambda: None
            screen.refresh_support_results = lambda: None

            screen.support_users_list.selection_set(0)
            screen.on_support_user_selected()
            screen.support_user_role_var.set("admin")
            screen.change_selected_support_user_role()
            screen.toggle_selected_support_user_active()
            screen.revoke_selected_support_user_sessions()
            screen.support_user_rfid_var.set("RFID-1234")
            screen.bind_selected_support_user_rfid()
            screen.clear_selected_support_user_rfid()

            screen.support_user_username_var.set("operator-b")
            screen.support_user_password_var.set("")
            screen.support_user_role_var.set("operator")
            screen.support_user_rfid_var.set("RFID-5678")
            screen.create_support_user()

            screen.support_results_list.selection_set(0)
            screen.open_support_result()
            screen.support_result_patch.set_payload(
                {
                    "decision_code": "ACCEPT",
                    "reject_reason_code": "",
                    "part_ready_match_ratio": 0.88,
                    "sticker_confidence": 0.91,
                    "detected_class": "GOOD",
                    "expected_class": "GOOD",
                    "note": "Corrected in support",
                }
            )
            screen.apply_support_result_correction()
            screen.retry_selected_support_result()
            screen.retry_visible_support_pushes()
            screen.delete_selected_support_result()

            self.assertEqual(role_calls, [(5, "admin")])
            self.assertEqual(active_calls, [(5, False)])
            self.assertEqual(revoke_calls, [5])
            self.assertEqual(clear_calls, [5])
            self.assertEqual(created_payloads[-1]["role"], "operator")
            self.assertTrue(created_payloads[-1]["password"])
            self.assertEqual(bind_calls[-1], (88, "RFID-5678"))
            self.assertEqual(update_calls[-1][0], 77)
            self.assertEqual(retry_calls, [77])
            self.assertEqual(retry_failed_calls[-1], [77])
            self.assertEqual(delete_calls, [77])
            screen.destroy()

    def test_engineer_screen_initializes(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        self.assertTrue(screen.winfo_exists())
        self.assertIsNotNone(screen.training_summary)
        self.assertIsNotNone(screen.annotation_canvas)
        self.assertEqual(str(screen.annot_dataset["state"]), "readonly")
        self.assertEqual(screen.train_device.get(), "auto")
        self.assertEqual(str(screen.train_device["state"]), "readonly")
        self.assertTrue(screen.train_base_model.get())
        self.assertEqual(str(screen.train_base_model["state"]), "readonly")
        self.assertEqual(str(screen.train_dataset_version["state"]), "readonly")
        self.assertEqual(str(screen.annot_shape["state"]), "readonly")
        self.assertEqual(str(screen.annot_class["state"]), "normal")
        self.assertEqual(str(screen.annot_apply_class_button["state"]), "disabled")
        screen.destroy()

    def test_engineer_create_augment_job_uses_checklist_transforms(self) -> None:
        payloads: list[dict] = []

        def create_augment_job(payload: dict):
            payloads.append(dict(payload))
            return {"id": "aug-1", **payload}

        with mock.patch.object(self.api, "create_augment_job", side_effect=create_augment_job):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            screen.select_tab("Training")
            screen.update_idletasks()

            screen.augment_dataset.delete(0, "end")
            screen.augment_dataset.insert(0, "ds-1")
            for var in screen._augment_transform_vars.values():
                var.set(False)
            screen._augment_transform_vars["rotate"].set(True)
            screen._augment_transform_vars["noise"].set(True)
            screen._on_augment_transform_selection_changed()

            screen.create_augment_job()

            self.assertEqual(len(payloads), 1)
            self.assertEqual(payloads[0]["dataset_id"], "ds-1")
            # Transforms are returned in catalog order (photometric before geometric).
            self.assertEqual(sorted(payloads[0]["transforms"]), sorted(["rotate", "noise"]))
            self.assertEqual(payloads[0]["multiplier"], 2)
            screen.destroy()

    def test_engineer_job_listboxes_disable_exportselection(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        screen.select_tab("Training")
        screen.update_idletasks()

        self.assertEqual(str(screen.augment_jobs.cget("exportselection")), "0")
        self.assertEqual(str(screen.train_jobs.cget("exportselection")), "0")
        screen.destroy()

    def test_engineer_delete_selected_augment_job_calls_api(self) -> None:
        jobs = [{"id": "aug-1", "dataset_id": "ds-1", "status": "queued", "transforms": ["flip_h"], "multiplier": 2}]
        deleted_ids: list[str] = []

        def delete_augment_job(job_id: str):
            deleted_ids.append(job_id)
            return {"deleted": True, "id": job_id}

        with mock.patch.object(self.api, "list_augment_jobs", return_value=jobs), mock.patch.object(
            self.api,
            "delete_augment_job",
            side_effect=delete_augment_job,
        ), mock.patch("client_tk.app.screens.engineer.view.messagebox.askyesno", return_value=True):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            screen.select_tab("Training")
            screen.update_idletasks()
            screen.augment_jobs.selection_set(0)

            screen.delete_selected_augment_job()

            self.assertEqual(deleted_ids, ["aug-1"])
            screen.destroy()

    def test_engineer_delete_selected_training_job_calls_api(self) -> None:
        jobs = [{"id": "train-2", "dataset_id": "ds-1", "status": "queued"}]
        deleted_ids: list[str] = []

        def delete_training_job(job_id: str):
            deleted_ids.append(job_id)
            return {"deleted": True, "id": job_id}

        with mock.patch.object(self.api, "list_training_jobs", return_value=jobs), mock.patch.object(
            self.api,
            "delete_training_job",
            side_effect=delete_training_job,
        ), mock.patch("client_tk.app.screens.engineer.view.messagebox.askyesno", return_value=True):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            screen.select_tab("Training")
            screen.update_idletasks()
            screen.train_jobs.selection_set(0)

            screen.delete_selected_training_job()

            self.assertEqual(deleted_ids, ["train-2"])
            screen.destroy()

    def test_engineer_delete_selected_model_calls_api(self) -> None:
        models = [{"id": 9, "name": "Model 9", "path": "models/m9.pt"}]
        calls: list[tuple[int, bool]] = []

        def delete_model(model_id: int, *, purge_files: bool = False):
            calls.append((model_id, purge_files))
            return {"deleted": True, "id": model_id, "purge_files": purge_files}

        with mock.patch.object(self.api, "list_models", return_value=models), mock.patch.object(
            self.api,
            "delete_model",
            side_effect=delete_model,
        ), mock.patch("client_tk.app.screens.engineer.view.messagebox.askyesno", return_value=True):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            screen.select_tab("Models")
            screen.models_list.selection_set(0)

            screen.delete_selected_model()

            self.assertEqual(calls, [(9, True)])
            screen.destroy()

    def test_engineer_update_selected_profile_calls_api(self) -> None:
        profiles = [{"id": 4, "name": "Profile Old", "profile": {"colorspace": "LAB"}}]
        calls: list[tuple[int, dict]] = []

        def update_profile(profile_id: int, payload: dict):
            calls.append((profile_id, dict(payload)))
            return {"id": profile_id, **payload}

        with mock.patch.object(self.api, "list_profiles", return_value=profiles), mock.patch.object(
            self.api,
            "update_profile",
            side_effect=update_profile,
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            screen.select_tab("Calibration")
            screen.profiles_list.selection_set(0)
            screen.on_profile_selected()
            screen.profile_name.delete(0, "end")
            screen.profile_name.insert(0, "Profile New")

            screen.update_selected_profile()

            self.assertTrue(calls)
            profile_id, payload = calls[-1]
            self.assertEqual(profile_id, 4)
            self.assertEqual(payload["name"], "Profile New")
            self.assertEqual(payload["profile"], {"colorspace": "LAB"})
            screen.destroy()

    def test_engineer_deregister_selected_workstation_calls_api(self) -> None:
        workstations = [{"machine_id": "WS-01", "line_id": "LINE-A", "station_id": "ST-01", "last_seen_at": "2026-04-10T10:00:00+00:00"}]
        deleted_ids: list[str] = []

        def delete_workstation(machine_id: str):
            deleted_ids.append(machine_id)
            return {"deleted": True, "machine_id": machine_id}

        with mock.patch.object(self.api, "list_workstations", return_value=workstations), mock.patch.object(
            self.api,
            "delete_workstation",
            side_effect=delete_workstation,
        ), mock.patch("client_tk.app.screens.engineer.view.messagebox.askyesno", return_value=True):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            screen.workstations.selection_set(0)

            screen.deregister_selected_workstation()

            self.assertEqual(deleted_ids, ["WS-01"])
            screen.destroy()

    def test_engineer_layout_hysteresis_prevents_threshold_flapping(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()

        screen._layout_compact = False
        self.assertFalse(screen._should_use_compact_layout(1350))
        self.assertFalse(screen._should_use_compact_layout(1330))
        self.assertTrue(screen._should_use_compact_layout(1310))

        screen._layout_compact = True
        self.assertTrue(screen._should_use_compact_layout(1380))
        self.assertFalse(screen._should_use_compact_layout(1410))

        screen.destroy()

    def test_engineer_training_tab_refreshes_base_models_once_on_empty_cache(self) -> None:
        base_model_calls = 0

        def list_base_models(*_args, **_kwargs):
            nonlocal base_model_calls
            base_model_calls += 1
            return [
                {
                    "id": "yolov5s",
                    "display_label": "YOLOv5 Small (yolov5s)",
                    "display_name": "YOLOv5 Small",
                    "family": "yolov5",
                    "family_label": "YOLOv5",
                    "variant": "s",
                    "variant_label": "Small",
                    "runtime": "ultralytics",
                    "weights_name": "yolov5s.pt",
                }
            ]

        with mock.patch.object(self.api, "list_base_models", side_effect=list_base_models):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            base_model_calls = 0
            screen._base_model_cache = []
            screen.train_base_model.configure(values=())
            screen.train_base_model.set("")

            screen.select_tab("Training")

            self.assertEqual(base_model_calls, 1)
            self.assertTrue(screen.train_base_model.get())

        screen.destroy()

    def test_engineer_models_tab_refreshes_once_on_first_open(self) -> None:
        model_calls = 0

        def list_models(*_args, **_kwargs):
            nonlocal model_calls
            model_calls += 1
            return [{"id": "m1", "name": "Model 1", "path": "models/m1.pt"}]

        with mock.patch.object(self.api, "list_models", side_effect=list_models):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            screen.select_tab("Models")

            self.assertEqual(model_calls, 1)
            self.assertEqual(len(screen._model_cache), 1)

        screen.destroy()

    def test_engineer_calibration_tab_refreshes_once_on_first_open(self) -> None:
        profile_calls = 0

        def list_profiles(*_args, **_kwargs):
            nonlocal profile_calls
            profile_calls += 1
            return [{"id": "p1", "name": "Profile 1", "profile": {"colorspace": "LAB"}}]

        with mock.patch.object(self.api, "list_profiles", side_effect=list_profiles):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            screen.select_tab("Calibration")

            self.assertEqual(profile_calls, 1)
            self.assertEqual(len(screen._profile_cache), 1)

        screen.destroy()

    def test_engineer_layout_split_shell_clears_old_weights_on_compact_switch(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()

        screen._layout_split_shell(screen.training_lower, screen.training_jobs_panel, screen.training_detail_panel, compact=False, left_weight=2, right_weight=3)
        self.assertEqual(int(screen.training_lower.columnconfigure(1)["weight"]), 3)

        screen._layout_split_shell(screen.training_lower, screen.training_jobs_panel, screen.training_detail_panel, compact=True, left_weight=2, right_weight=3)
        self.assertEqual(int(screen.training_lower.columnconfigure(0)["weight"]), 1)
        self.assertEqual(int(screen.training_lower.columnconfigure(1)["weight"]), 0)
        self.assertEqual(int(screen.training_lower.rowconfigure(0)["weight"]), 1)
        self.assertEqual(int(screen.training_lower.rowconfigure(1)["weight"]), 1)

        screen.destroy()

    def test_engineer_training_summary_shows_key_metrics(self) -> None:
        jobs = [
            {
                "id": "train-001",
                "dataset_id": "ds-train",
                "status": "completed",
                "base_model": "yolov11m",
                "base_model_display_name": "YOLOv11 Medium",
                "metrics": {
                    "accuracy": 0.9234,
                    "map": 0.8123,
                },
                "evaluation": {
                    "r2": 0.7845,
                    "rmse": 0.0456,
                },
                "trained_model_path": "models/trained/train-001.pt",
            }
        ]

        with mock.patch.object(self.api, "list_training_jobs", return_value=jobs):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            screen.refresh_training_jobs()

            screen.train_jobs.selection_clear(0, "end")
            screen.train_jobs.selection_set(0)
            screen.on_training_selected()

            self.assertEqual(screen.training_summary._labels["base_model"].cget("text"), "YOLOv11 Medium")
            self.assertEqual(screen.training_summary._labels["status"].cget("text"), "completed")
            self.assertEqual(screen.training_summary._labels["accuracy"].cget("text"), "92.34%")
            self.assertEqual(screen.training_summary._labels["map_score"].cget("text"), "81.23%")
            self.assertEqual(screen.training_summary._labels["r2_score"].cget("text"), "0.785")
            self.assertEqual(screen.training_summary._labels["error"].cget("text"), "RMSE 0.0456")

            screen.destroy()

    def test_engineer_training_progress_widgets_show_stage_and_message(self) -> None:
        jobs = [
            {
                "id": "train-002",
                "dataset_id": "ds-train",
                "status": "running",
                "base_model": "yolov5s",
                "base_model_display_name": "YOLOv5 Small",
                "requested_device_mode": "auto",
                "effective_device": "cpu",
                "progress_percent": 42,
                "progress_stage": "training",
                "progress_message": "YOLO training is running.",
                "params": {"device_mode": "auto"},
            }
        ]

        with mock.patch.object(self.api, "list_training_jobs", return_value=jobs):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            screen.select_tab("Training")
            screen.update_idletasks()

            screen.train_jobs.selection_clear(0, "end")
            screen.train_jobs.selection_set(0)
            screen.on_training_selected()

            self.assertEqual(screen.training_progress_percent_var.get(), "42%")
            self.assertEqual(screen.training_progress_stage_var.get(), "Stage: training")
            self.assertEqual(screen.training_progress_message_var.get(), "YOLO training is running.")
            self.assertEqual(int(float(screen.training_progress_bar.cget("value"))), 42)
            self.assertIn("42%", screen.train_jobs.get(0))
            self.assertIsNotNone(screen._training_auto_refresh_job)

            screen.select_tab("Data")
            self.assertIsNone(screen._training_auto_refresh_job)

            screen.destroy()

    def test_engineer_training_filter_modes(self) -> None:
        jobs = [
            {
                "id": "train-q",
                "dataset_id": "ds-train",
                "status": "queued",
                "base_model": "yolov5s",
                "base_model_display_name": "YOLOv5 Small",
                "requested_device_mode": "auto",
                "effective_device": "pending",
                "progress_percent": 5,
                "progress_stage": "queued",
                "params": {"device_mode": "auto"},
            },
            {
                "id": "train-r",
                "dataset_id": "ds-train",
                "status": "running",
                "base_model": "yolov5s",
                "base_model_display_name": "YOLOv5 Small",
                "requested_device_mode": "auto",
                "effective_device": "cpu",
                "progress_percent": 42,
                "progress_stage": "training",
                "params": {"device_mode": "auto"},
            },
            {
                "id": "train-f",
                "dataset_id": "ds-train",
                "status": "failed",
                "base_model": "yolov11m",
                "base_model_display_name": "YOLOv11 Medium",
                "requested_device_mode": "gpu",
                "effective_device": "cpu",
                "progress_percent": 95,
                "progress_stage": "failed",
                "params": {"device_mode": "gpu"},
            },
            {
                "id": "train-c",
                "dataset_id": "ds-train",
                "status": "completed",
                "base_model": "yolov11m",
                "base_model_display_name": "YOLOv11 Medium",
                "requested_device_mode": "cpu",
                "effective_device": "cpu",
                "progress_percent": 100,
                "progress_stage": "completed",
                "params": {"device_mode": "cpu"},
            },
            {
                "id": "train-x",
                "dataset_id": "ds-train",
                "status": "cancelled",
                "base_model": "yolov11m",
                "base_model_display_name": "YOLOv11 Medium",
                "requested_device_mode": "cpu",
                "effective_device": "cpu",
                "progress_percent": 55,
                "progress_stage": "cancelled",
                "params": {"device_mode": "cpu"},
            },
        ]

        with mock.patch.object(self.api, "list_training_jobs", return_value=jobs):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            screen.select_tab("Training")
            screen.update_idletasks()

            self.assertEqual(screen.train_status_filter.get(), "All")
            self.assertEqual(screen.train_jobs.size(), 5)

            screen.train_status_filter.set("Failed")
            screen._on_training_filter_changed()
            screen.update_idletasks()
            self.assertEqual(screen.train_jobs.size(), 1)
            self.assertIn("train-f", screen.train_jobs.get(0))
            self.assertIsNotNone(screen._training_auto_refresh_job)

            screen.train_status_filter.set("Completed")
            screen._on_training_filter_changed()
            screen.update_idletasks()
            self.assertEqual(screen.train_jobs.size(), 1)
            self.assertIn("train-c", screen.train_jobs.get(0))

            screen.train_status_filter.set("Active")
            screen._on_training_filter_changed()
            screen.update_idletasks()
            self.assertEqual(screen.train_jobs.size(), 2)
            active_rows = [str(item) for item in screen.train_jobs.get(0, "end")]
            self.assertTrue(any("train-q" in row for row in active_rows))
            self.assertTrue(any("train-r" in row for row in active_rows))

            screen.destroy()

    def test_engineer_training_status_colors_and_log_autoscroll(self) -> None:
        jobs = [
            {
                "id": "train-003",
                "dataset_id": "ds-train",
                "status": "failed",
                "base_model": "yolov11m",
                "base_model_display_name": "YOLOv11 Medium",
                "requested_device_mode": "gpu",
                "effective_device": "cpu",
                "progress_percent": 95,
                "progress_stage": "failed",
                "progress_message": "Training failed due to timeout.",
                "error": "timeout",
                "log": [f"log line {index}" for index in range(80)],
                "params": {"device_mode": "gpu"},
            }
        ]

        with mock.patch.object(self.api, "list_training_jobs", return_value=jobs):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            screen.select_tab("Training")
            screen.update_idletasks()

            screen.train_jobs.selection_clear(0, "end")
            screen.train_jobs.selection_set(0)
            screen.on_training_selected()

            self.assertEqual(screen._training_status_text_color("failed"), "#dc2626")
            self.assertEqual(screen._training_stage_text_color("failed", status="failed"), "#dc2626")
            self.assertEqual(screen.training_summary._labels["status"].cget("text_color"), "#dc2626")

            rendered_log = screen.training_log_text.get("1.0", "end-1c")
            self.assertIn("log line 79", rendered_log)
            self.assertGreaterEqual(float(screen.training_log_text.yview()[1]), 0.99)

            screen.destroy()

    def test_engineer_training_refresh_error_dialog_is_not_reentrant(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()

        with mock.patch("client_tk.app.screens.engineer.view.messagebox.showerror") as showerror:
            def _nested_showerror(_title, _message):
                screen._show_training_refresh_error("Training", RuntimeError("second"))

            showerror.side_effect = _nested_showerror
            screen._show_training_refresh_error("Training", RuntimeError("first"))

        self.assertEqual(showerror.call_count, 1)
        screen.destroy()

    def test_engineer_training_job_uses_active_dataset_version_when_combo_is_blank(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        captured_payloads: list[dict] = []

        def create_training_job(payload: dict):
            captured_payloads.append(dict(payload))
            return {"id": "train-x", **payload}

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-vers", "name": "Dataset Versioned"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            return_value=[{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}],
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            return_value=image_bytes,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={"labels": []},
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=[
                {
                    "id": "ver-1",
                    "display_label": "v1 | Snapshot v1 | ready | 1/1 ann",
                    "version_number": 1,
                    "name": "Snapshot v1",
                    "status": "ready",
                    "export_format": "yolo",
                    "export_root": "data/export/ver-1",
                    "image_count": 1,
                    "annotated_image_count": 1,
                    "coverage_percent": 100.0,
                    "class_names": ["K0W-HB0"],
                }
            ],
        ), mock.patch.object(
            self.api,
            "create_training_job",
            side_effect=create_training_job,
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            screen.train_dataset_version.set("")
            screen._active_dataset_version_id = "ver-1"
            screen.create_training_job()

            self.assertTrue(captured_payloads)
            self.assertEqual(captured_payloads[-1].get("dataset_version_id"), "ver-1")
            self.assertEqual(captured_payloads[-1].get("epochs"), 1)
            self.assertEqual(captured_payloads[-1].get("imgsz"), 320)
            self.assertEqual(captured_payloads[-1].get("batch"), 4)
            self.assertEqual(captured_payloads[-1].get("patience"), 5)
            self.assertEqual(captured_payloads[-1].get("workers"), 0)
            self.assertEqual(captured_payloads[-1].get("cache"), False)

            screen.destroy()

    def test_engineer_training_job_includes_custom_hyperparams(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        captured_payloads: list[dict] = []

        def create_training_job(payload: dict):
            captured_payloads.append(dict(payload))
            return {"id": "train-hparams", **payload}

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-vers", "name": "Dataset Versioned"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            return_value=[{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}],
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            return_value=image_bytes,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={"labels": []},
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=[
                {
                    "id": "ver-1",
                    "display_label": "v1 | Snapshot v1 | ready | 1/1 ann",
                    "version_number": 1,
                    "name": "Snapshot v1",
                    "status": "ready",
                    "export_format": "yolo",
                    "export_root": "data/export/ver-1",
                    "image_count": 1,
                    "annotated_image_count": 1,
                    "coverage_percent": 100.0,
                    "class_names": ["K0W-HB0"],
                }
            ],
        ), mock.patch.object(
            self.api,
            "create_training_job",
            side_effect=create_training_job,
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            screen.train_epochs.delete(0, "end")
            screen.train_epochs.insert(0, "3")
            screen.train_imgsz.delete(0, "end")
            screen.train_imgsz.insert(0, "640")
            screen.train_batch.delete(0, "end")
            screen.train_batch.insert(0, "8")
            screen.train_patience.delete(0, "end")
            screen.train_patience.insert(0, "20")
            screen.train_workers.delete(0, "end")
            screen.train_workers.insert(0, "2")
            screen.train_cache_var.set(True)

            screen.create_training_job()

            self.assertTrue(captured_payloads)
            payload = captured_payloads[-1]
            self.assertEqual(payload.get("epochs"), 3)
            self.assertEqual(payload.get("imgsz"), 640)
            self.assertEqual(payload.get("batch"), 8)
            self.assertEqual(payload.get("patience"), 20)
            self.assertEqual(payload.get("workers"), 2)
            self.assertEqual(payload.get("cache"), True)

            screen.destroy()

    def test_engineer_dataset_version_selection_auto_updates_detail(self) -> None:
        versions = [
            {
                "id": "ver-1",
                "display_label": "v1 | Snapshot v1 | ready | 1/1 ann",
                "version_number": 1,
                "name": "Snapshot v1",
                "status": "ready",
                "export_format": "yolo",
                "export_root": "data/export/ver-1",
                "image_count": 1,
                "annotated_image_count": 1,
                "coverage_percent": 100.0,
                "class_names": ["K0W-HB0"],
            },
            {
                "id": "ver-2",
                "display_label": "v2 | Snapshot v2 | ready | 2/2 ann",
                "version_number": 2,
                "name": "Snapshot v2",
                "status": "ready",
                "export_format": "yolo",
                "export_root": "data/export/ver-2",
                "image_count": 2,
                "annotated_image_count": 2,
                "coverage_percent": 100.0,
                "class_names": ["K0W-HB0", "K1Z-FA0"],
            },
        ]
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()

        screen._dataset_version_cache = [dict(item) for item in versions]
        screen._dataset_version_lookup = {
            versions[0]["display_label"]: versions[0],
            versions[0]["id"]: versions[0],
            versions[1]["display_label"]: versions[1],
            versions[1]["id"]: versions[1],
        }
        screen.dataset_versions.delete(0, "end")
        screen.dataset_versions.insert("end", f"{versions[0]['display_label']} | yolo | data/export/ver-1")
        screen.dataset_versions.insert("end", f"{versions[1]['display_label']} | yolo | data/export/ver-2")
        screen.train_dataset_version.configure(values=[versions[0]["display_label"], versions[1]["display_label"]])
        screen.train_dataset_version.set(versions[0]["display_label"])
        screen._active_dataset_version_id = "ver-1"
        screen.dataset_version_detail.set_payload(versions[0])

        screen._ignore_next_dataset_version_selection_events = 1
        screen.dataset_versions.selection_clear(0, "end")
        screen.dataset_versions.selection_set(1)
        screen.on_dataset_version_selected(SimpleNamespace(widget=screen.dataset_versions))

        self.assertEqual(screen.train_dataset_version.get(), versions[1]["display_label"])
        self.assertEqual(screen._active_dataset_version_id, "ver-2")
        self.assertEqual(screen.dataset_version_detail.get_payload()["id"], "ver-2")

        screen.destroy()

    def test_engineer_dataset_selection_does_not_refetch_same_dataset(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()

        screen._dataset_cache = [{"id": "ds-1", "name": "Dataset One"}]
        screen.dataset_list.delete(0, "end")
        screen.dataset_list.insert("end", "ds-1 | Dataset One | 0 imgs / 0 ann / 0 aug")
        screen.dataset_list.selection_set(0)

        with mock.patch.object(screen, "refresh_dataset_files") as refresh_files, mock.patch.object(
            screen,
            "refresh_annotation_images",
        ) as refresh_images, mock.patch.object(screen, "refresh_dataset_versions") as refresh_versions:
            screen.on_dataset_selected()
            screen.on_dataset_selected()

        self.assertEqual(refresh_files.call_count, 1)
        self.assertEqual(refresh_images.call_count, 1)
        self.assertEqual(refresh_versions.call_count, 1)

        screen.destroy()

    def test_engineer_training_tab_prefills_dataset_from_current_context(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()

        screen._annotation_dataset_id = "ds-context"
        screen.train_dataset.delete(0, "end")

        screen.select_tab("Training")

        self.assertEqual(screen.train_dataset.get().strip(), "ds-context")

        screen.destroy()

    def test_engineer_training_job_uses_selected_dataset_version(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        captured_payloads: list[dict] = []
        versions = [
            {
                "id": "ver-1",
                "display_label": "v1 | Snapshot v1 | ready | 1/1 ann",
                "version_number": 1,
                "name": "Snapshot v1",
                "status": "ready",
                "export_format": "yolo",
                "export_root": "data/export/ver-1",
                "image_count": 1,
                "annotated_image_count": 1,
                "coverage_percent": 100.0,
                "class_names": ["K0W-HB0"],
            },
            {
                "id": "ver-2",
                "display_label": "v2 | Snapshot v2 | ready | 2/2 ann",
                "version_number": 2,
                "name": "Snapshot v2",
                "status": "ready",
                "export_format": "yolo",
                "export_root": "data/export/ver-2",
                "image_count": 2,
                "annotated_image_count": 2,
                "coverage_percent": 100.0,
                "class_names": ["K0W-HB0", "K1Z-FA0"],
            },
        ]

        def list_training_jobs():
            if not captured_payloads:
                return []
            payload = captured_payloads[-1]
            return [
                {
                    "id": "train-x",
                    "dataset_id": payload.get("dataset_id", "ds-vers"),
                    "status": "queued",
                    "base_model": payload.get("base_model", "yolov5s"),
                    "base_model_display_name": payload.get("base_model_display_name", "YOLOv5 Small"),
                    "requested_device_mode": payload.get("device_mode", "auto"),
                    "effective_device": "pending",
                    "params": {"device_mode": payload.get("device_mode", "auto")},
                    "dataset_version_id": payload.get("dataset_version_id"),
                    "dataset_version_display_label": payload.get("dataset_version_display_label"),
                }
            ]

        def create_training_job(payload: dict):
            captured_payloads.append(dict(payload))
            return {"id": "train-x", **payload}

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-vers", "name": "Dataset Versioned"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            return_value=[{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}],
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            return_value=image_bytes,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={"labels": []},
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=[dict(item) for item in versions],
        ), mock.patch.object(
            self.api,
            "list_training_jobs",
            side_effect=list_training_jobs,
        ), mock.patch.object(
            self.api,
            "create_training_job",
            side_effect=create_training_job,
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            screen.train_dataset_version.set(versions[1]["display_label"])
            screen.on_dataset_version_selected()
            screen.create_training_job()

            self.assertTrue(captured_payloads)
            self.assertEqual(captured_payloads[-1].get("dataset_version_id"), "ver-2")
            self.assertEqual(screen._active_dataset_version_id, "ver-2")
            self.assertEqual(screen.train_dataset_version.get(), versions[1]["display_label"])

            screen.destroy()

    def test_engineer_dataset_version_selection_guard_decrements_by_two_on_clear(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()

        screen._ignore_next_dataset_version_selection_events = 4
        screen._clear_dataset_version_selection_guard()

        self.assertEqual(screen._ignore_next_dataset_version_selection_events, 2)

        screen.destroy()

    def test_engineer_annotation_dataset_follows_selected_dataset(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        screen._dataset_cache = [{"id": "ds-123", "name": "Sample Dataset"}]
        screen.dataset_list.delete(0, "end")
        screen.dataset_list.insert("end", "ds-123 | Sample Dataset | 0 imgs / 0 ann / 0 aug")
        screen._sync_annotation_dataset_selector()
        screen.dataset_list.selection_clear(0, "end")
        screen.dataset_list.selection_set(0)
        screen.on_dataset_selected()
        self.assertEqual(screen.annot_dataset_var.get(), "Sample Dataset")
        self.assertEqual(screen._annotation_dataset_id, "ds-123")
        self.assertEqual(str(screen.annot_dataset["state"]), "readonly")
        self.assertEqual(tuple(screen.annot_dataset["values"]), ("Sample Dataset",))
        self.assertEqual(screen._resolve_annotation_dataset_id(), "ds-123")
        screen.destroy()

    def test_engineer_annotation_dataset_dropdown_selects_dataset(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        screen._dataset_cache = [
            {"id": "ds-1", "name": "Dataset One"},
            {"id": "ds-2", "name": "Dataset Two"},
        ]
        screen.dataset_list.delete(0, "end")
        screen.dataset_list.insert("end", "ds-1 | Dataset One | 0 imgs / 0 ann / 0 aug")
        screen.dataset_list.insert("end", "ds-2 | Dataset Two | 0 imgs / 0 ann / 0 aug")
        screen._sync_annotation_dataset_selector()

        self.assertEqual(tuple(screen.annot_dataset["values"]), ("Dataset One", "Dataset Two"))

        screen.annot_dataset_var.set("Dataset Two")
        screen._on_annotation_dataset_selected()

        self.assertEqual(screen._annotation_dataset_id, "ds-2")
        self.assertEqual(screen.annot_dataset_var.get(), "Dataset Two")
        self.assertEqual(screen.upload_dataset_id.get().strip(), "ds-2")
        self.assertEqual(screen._selected_dataset_id(), "ds-2")
        self.assertEqual(screen._resolve_annotation_dataset_id(), "ds-2")
        screen.destroy()

    def test_engineer_dataset_version_selection_ignores_duplicate_click(self) -> None:
        versions = [
            {
                "id": "ver-1",
                "display_label": "v1 | Snapshot v1 | ready | 1/1 ann",
                "version_number": 1,
                "name": "Snapshot v1",
                "status": "ready",
                "export_format": "yolo",
                "export_root": "data/export/ver-1",
                "image_count": 1,
                "annotated_image_count": 1,
                "coverage_percent": 100.0,
                "class_names": ["K0W-HB0"],
            },
            {
                "id": "ver-2",
                "display_label": "v2 | Snapshot v2 | ready | 2/2 ann",
                "version_number": 2,
                "name": "Snapshot v2",
                "status": "ready",
                "export_format": "yolo",
                "export_root": "data/export/ver-2",
                "image_count": 2,
                "annotated_image_count": 2,
                "coverage_percent": 100.0,
                "class_names": ["K0W-HB0", "K1Z-FA0"],
            },
        ]
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()

        screen._dataset_version_cache = [dict(item) for item in versions]
        screen._dataset_version_lookup = {
            versions[0]["display_label"]: versions[0],
            versions[0]["id"]: versions[0],
            versions[1]["display_label"]: versions[1],
            versions[1]["id"]: versions[1],
        }
        screen.dataset_versions.delete(0, "end")
        screen.dataset_versions.insert("end", f"{versions[0]['display_label']} | yolo | data/export/ver-1")
        screen.dataset_versions.insert("end", f"{versions[1]['display_label']} | yolo | data/export/ver-2")
        screen.train_dataset_version.configure(values=[versions[0]["display_label"], versions[1]["display_label"]])
        screen.train_dataset_version.set(versions[0]["display_label"])
        screen._active_dataset_version_id = "ver-1"
        screen.dataset_version_detail.set_payload(versions[0])

        with mock.patch.object(screen, "_reload_active_annotation_image") as reload_image:
            screen.dataset_versions.selection_clear(0, "end")
            screen.dataset_versions.selection_set(1)
            screen.on_dataset_version_selected(SimpleNamespace(widget=screen.dataset_versions))
            screen.on_dataset_version_selected(SimpleNamespace(widget=screen.dataset_versions))

        self.assertEqual(screen.train_dataset_version.get(), versions[1]["display_label"])
        self.assertEqual(screen._active_dataset_version_id, "ver-2")
        self.assertEqual(screen.dataset_version_detail.get_payload()["id"], "ver-2")
        self.assertEqual(reload_image.call_count, 1)

        screen.destroy()

    def test_engineer_annotation_dataset_resolution_handles_destroyed_listbox(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        screen.annot_dataset_var.set("")
        screen._annotation_dataset_id = None
        screen.dataset_list.destroy()

        self.assertIsNone(screen._selected_dataset_id())
        self.assertIsNone(screen._resolve_annotation_dataset_id())

        screen.destroy()

    def test_engineer_empty_dataset_selection_keeps_annotation_context(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-safe", "name": "Dataset Safe"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            return_value=[{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}],
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            return_value=image_bytes,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={"labels": []},
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=[],
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            self.assertEqual(screen._annotation_dataset_id, "ds-safe")
            self.assertEqual(screen.annot_dataset_var.get(), "Dataset Safe")

            screen.dataset_list.selection_clear(0, "end")
            screen.on_dataset_selected()
            screen.update_idletasks()

            self.assertEqual(screen._annotation_dataset_id, "ds-safe")
            self.assertEqual(screen.annot_dataset_var.get(), "Dataset Safe")
            self.assertEqual(screen.annot_image_var.get(), "sample.png")
            self.assertIsNotNone(screen.annotation_canvas._source_frame)
            self.assertIsNotNone(screen.annotation_canvas._photo)
            self.assertEqual(screen.dataset_list.curselection(), (0,))

        screen.destroy()

    def test_engineer_annotation_toolbar_changes_keep_context_visible(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-keep", "name": "Dataset Keep"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            return_value=[{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}],
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            return_value=image_bytes,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={"labels": []},
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=[],
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            self.assertEqual(screen.annot_dataset_var.get(), "Dataset Keep")
            self.assertEqual(screen._selected_dataset_id(), "ds-keep")
            self.assertEqual(screen.annot_image_var.get(), "sample.png")
            self.assertIsNotNone(screen.annotation_canvas._photo)

            screen.annot_shape.set("polygon")
            screen._sync_annotation_mode()
            screen.update_idletasks()
            self.assertEqual(screen.annot_dataset_var.get(), "Dataset Keep")
            self.assertEqual(screen._selected_dataset_id(), "ds-keep")
            self.assertIsNotNone(screen.annotation_canvas._photo)

            screen.annot_class_var.set("manual-class")
            screen._on_annotation_class_input()
            screen.update_idletasks()
            self.assertEqual(screen.annot_dataset_var.get(), "Dataset Keep")
            self.assertEqual(screen._selected_dataset_id(), "ds-keep")
            self.assertIsNotNone(screen.annotation_canvas._photo)

        screen.destroy()

    def test_engineer_manual_save_annotation_button_persists_current_labels(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        save_calls: list[tuple[str, str, list[dict]]] = []

        def save_annotation(dataset_id: str, image_name: str, labels: list[dict]):
            save_calls.append((dataset_id, image_name, labels))
            return {"labels": labels}

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-save", "name": "Dataset Save"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            return_value=[{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}],
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            return_value=image_bytes,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={"labels": []},
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=[],
        ), mock.patch.object(
            self.api,
            "save_annotation",
            side_effect=save_annotation,
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            save_calls.clear()
            screen.annotation_canvas._labels = [
                {
                    "type": "bbox",
                    "shape_type": "bbox",
                    "class_name": "manual-save",
                    "class": "manual-save",
                    "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
                    "normalized": True,
                }
            ]
            screen.save_current_annotation()

            self.assertTrue(save_calls)
            self.assertEqual(save_calls[-1][0], "ds-save")
            self.assertEqual(save_calls[-1][1], "sample.png")
            self.assertEqual(save_calls[-1][2][0].get("class_name"), "manual-save")
            self.assertEqual(screen.annotation_status.get(), "Saved sample.png")

        screen.destroy()

    def test_engineer_annotation_class_input_auto_applies_to_selected_label(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        save_calls: list[list[dict]] = []

        def save_annotation(dataset_id: str, image_name: str, labels: list[dict]):
            save_calls.append(labels)
            return {"labels": labels}

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-auto", "name": "Dataset Auto"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            return_value=[{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}],
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            return_value=image_bytes,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={
                "labels": [
                    {
                        "type": "bbox",
                        "shape_type": "bbox",
                        "class_name": "object",
                        "class": "object",
                        "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
                        "normalized": True,
                    }
                ]
            },
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=[],
        ), mock.patch.object(
            self.api,
            "save_annotation",
            side_effect=save_annotation,
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            self.assertTrue(screen.annotation_canvas.set_selected_label_index(0))
            screen.annot_class_var.set("K0W-HB0")
            screen._on_annotation_class_input()
            screen.update_idletasks()

            self.assertTrue(save_calls)
            self.assertEqual(screen.annotation_canvas.get_labels()[0].get("class_name"), "K0W-HB0")
            self.assertEqual(screen.annotation_canvas.get_selected_label_index(), 0)
            self.assertEqual(screen.annot_class_var.get(), "K0W-HB0")

        screen.destroy()

    def test_engineer_refresh_auto_selects_first_dataset(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-abc", "name": "Dataset A"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            return_value=[{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}],
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            wraps=self.api.download_dataset_image,
        ) as download_mock:
            screen.refresh_datasets()
        self.assertEqual(screen.annot_dataset_var.get(), "Dataset A")
        self.assertEqual(screen.annot_image_var.get(), "sample.png")
        self.assertEqual(screen._annotation_dataset_id, "ds-abc")
        self.assertIsNotNone(screen.annotation_canvas._source_frame)
        self.assertIsNotNone(screen.annotation_canvas._photo)
        self.assertIn("loaded via backend", screen.annotation_status.get())
        download_mock.assert_called_once_with("ds-abc", "sample.png")
        screen.destroy()

    def test_engineer_data_scroll_event_restores_annotation_canvas(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-scroll", "name": "Dataset Scroll"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            return_value=[{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}],
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            return_value=image_bytes,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={"labels": []},
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=[],
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            self._wait_for_engineer_annotation_load(screen)

            self.assertIsNotNone(screen.annotation_canvas._photo)

            # Simulate a render-loss edge case, then confirm scroll event path restores it.
            screen.annotation_canvas._photo = None
            screen.annotation_canvas._needs_redraw_on_map = True
            data_scroller = screen._tab_scrollers.get("data")
            self.assertIsNotNone(data_scroller)
            assert data_scroller is not None
            data_scroller.event_generate("<<ScrollableFrameScrolled>>")
            screen.update_idletasks()

            self.assertIsNotNone(screen.annotation_canvas._photo)

        screen.destroy()

    def test_engineer_annotation_image_cache_reuses_loaded_image(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        download_calls: list[tuple[str, str]] = []

        def download_dataset_image(dataset_id: str, image_name: str):
            download_calls.append((dataset_id, image_name))
            return image_bytes

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-cache", "name": "Dataset Cache"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            return_value=[{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}],
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            side_effect=download_dataset_image,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={"labels": []},
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=[],
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            self._wait_for_engineer_annotation_load(screen)

            self.assertEqual(download_calls, [("ds-cache", "sample.png")])

            screen._load_annotation_for_index(0, save_current=False)
            screen.update_idletasks()

            self.assertEqual(download_calls, [("ds-cache", "sample.png")])
            self.assertEqual(len(screen._annotation_asset_cache), 1)

            screen.destroy()

    def test_engineer_annotation_image_cache_is_bounded(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        download_calls: list[tuple[str, str]] = []

        def list_dataset_files(dataset_id: str, target: str = "images"):
            return [
                {"name": f"sample-{index}.png", "path": f"Z:/missing/sample-{index}.png", "size": 123}
                for index in range(3)
            ]

        def download_dataset_image(dataset_id: str, image_name: str):
            download_calls.append((dataset_id, image_name))
            return image_bytes

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-bound", "name": "Dataset Bound"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            side_effect=list_dataset_files,
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            side_effect=download_dataset_image,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={"labels": []},
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=[],
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen._annotation_cache_max_items = 2
            self._wait_for_engineer_annotation_load(screen)

            screen._load_annotation_for_index(1, save_current=False)
            screen.update_idletasks()
            screen._load_annotation_for_index(2, save_current=False)
            screen.update_idletasks()

            self.assertLessEqual(len(screen._annotation_asset_cache), 2)
            cached_keys = list(screen._annotation_asset_cache.keys())
            self.assertNotIn(("ds-bound", "sample-0.png"), cached_keys)
            self.assertIn(("ds-bound", "sample-2.png"), cached_keys)
            self.assertGreaterEqual(len(download_calls), 3)

            screen.destroy()

    def test_engineer_annotation_class_combines_sources_and_edits_selected_label(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        versions = [
            {
                "id": "ver-1",
                "display_label": "v1 | Snapshot v1 | ready | 1/1 ann",
                "version_number": 1,
                "name": "Snapshot v1",
                "status": "ready",
                "export_format": "yolo",
                "export_root": "data/export/ver-1",
                "image_count": 1,
                "annotated_image_count": 1,
                "coverage_percent": 100.0,
                "class_names": ["K0W-HB0", "K1Z-FA0"],
            }
        ]

        save_calls: list[list[dict]] = []

        def list_dataset_files(dataset_id: str, target: str = "images"):
            if target == "images":
                return [{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}]
            return []

        def save_annotation(dataset_id: str, image_name: str, labels: list[dict]):
            save_calls.append(labels)
            return {"labels": labels}

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-123", "name": "Dataset A"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            side_effect=list_dataset_files,
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            return_value=image_bytes,
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=versions,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={
                "labels": [
                    {
                        "type": "bbox",
                        "class_name": "manual-old",
                        "class": "manual-old",
                        "bbox": {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
                        "normalized": True,
                    }
                ]
            },
        ), mock.patch.object(
            self.api,
            "save_annotation",
            side_effect=save_annotation,
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            class_values = list(screen.annot_class["values"])
            self.assertIn("K0W-HB0", class_values)
            self.assertIn("K1Z-FA0", class_values)
            self.assertIn("manual-old", class_values)

            self.assertTrue(screen.annotation_canvas.set_selected_label_index(0))
            screen.update_idletasks()
            self.assertEqual(str(screen.annot_apply_class_button["state"]), "normal")

            screen.annot_class_var.set("manual-new")
            screen._apply_class_to_selected_annotation()
            screen.update_idletasks()

            self.assertTrue(save_calls)
            self.assertEqual(save_calls[-1][0].get("class_name"), "manual-new")
            self.assertEqual(screen.annotation_canvas.get_labels()[0].get("class_name"), "manual-new")
            self.assertIn("manual-new", list(screen.annot_class["values"]))

        screen.destroy()

    def test_engineer_create_dataset_version_focuses_latest(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        versions = [
            {
                "id": "ver-1",
                "display_label": "v1 | Snapshot v1 | ready | 1/1 ann",
                "version_number": 1,
                "name": "Snapshot v1",
                "status": "ready",
                "export_format": "yolo",
                "export_root": "data/export/ver-1",
                "image_count": 1,
                "annotated_image_count": 1,
                "coverage_percent": 100.0,
                "class_names": ["K0W-HB0"],
            }
        ]

        def list_dataset_versions(dataset_id: str):
            return [dict(item) for item in versions]

        def create_dataset_version(dataset_id: str, payload: dict):
            created = {
                "id": "ver-2",
                "display_label": "v2 | Snapshot v2 | ready | 1/1 ann",
                "version_number": 2,
                "name": payload.get("name") or "Snapshot v2",
                "status": "ready",
                "export_format": "yolo",
                "export_root": "data/export/ver-2",
                "image_count": 1,
                "annotated_image_count": 1,
                "coverage_percent": 100.0,
                "class_names": ["K0W-HB0", "K1Z-FA0"],
            }
            versions.insert(0, dict(created))
            return created

        def list_dataset_files(dataset_id: str, target: str = "images"):
            if target == "images":
                return [{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}]
            return []

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-1", "name": "Dataset One"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            side_effect=list_dataset_files,
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            return_value=image_bytes,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={"labels": []},
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            side_effect=list_dataset_versions,
        ), mock.patch.object(
            self.api,
            "create_dataset_version",
            side_effect=create_dataset_version,
        ), mock.patch(
            "client_tk.app.screens.engineer.view.messagebox.showinfo",
            return_value=None,
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            self.assertEqual(screen._active_dataset_version_id, "ver-1")
            self.assertIsNotNone(screen.annotation_canvas._source_frame)

            # Simulate render/state drop before create version; implementation should restore active image.
            screen.annotation_canvas.clear()
            self.assertIsNone(screen.annotation_canvas._source_frame)

            screen.version_name.delete(0, "end")
            screen.version_name.insert(0, "Snapshot v2")
            screen.create_dataset_version()
            screen.update_idletasks()

            self.assertEqual(screen._active_dataset_version_id, "ver-2")
            selected = screen._selected_dataset_version_record()
            self.assertIsNotNone(selected)
            self.assertEqual(selected.get("id"), "ver-2")
            self.assertEqual(screen.train_dataset_version.get(), "v2 | Snapshot v2 | ready | 1/1 ann")
            self.assertEqual(screen.annot_dataset_var.get(), "Dataset One")
            self.assertEqual(screen.annot_image_var.get(), "sample.png")
            self.assertIsNotNone(screen.annotation_canvas._source_frame)
            self.assertIsNotNone(screen.annotation_canvas._photo)

        screen.destroy()

    def test_engineer_update_dataset_version_metadata_uses_selected_version(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        versions = [
            {
                "id": "ver-1",
                "display_label": "v1 | Snapshot v1 | ready | 1/1 ann",
                "version_number": 1,
                "name": "Snapshot v1",
                "description": "initial",
                "status": "ready",
                "export_format": "yolo",
                "export_root": "data/export/ver-1",
                "image_count": 1,
                "annotated_image_count": 1,
                "coverage_percent": 100.0,
                "class_names": ["K0W-HB0"],
            }
        ]
        calls: list[tuple[str, str, dict]] = []

        def list_dataset_versions(dataset_id: str):
            return [dict(item) for item in versions]

        def update_dataset_version(dataset_id: str, version_id: str, payload: dict):
            calls.append((dataset_id, version_id, dict(payload)))
            updated = dict(versions[0])
            updated.update(payload)
            updated["display_label"] = f"v1 | {updated.get('name')} | {updated.get('status')} | 1/1 ann"
            versions[0] = dict(updated)
            return dict(updated)

        with mock.patch.object(
            self.api,
            "list_datasets",
            return_value=[{"id": "ds-1", "name": "Dataset One"}],
        ), mock.patch.object(
            self.api,
            "list_dataset_files",
            return_value=[{"name": "sample.png", "path": "Z:/missing/sample.png", "size": 123}],
        ), mock.patch.object(
            self.api,
            "download_dataset_image",
            return_value=image_bytes,
        ), mock.patch.object(
            self.api,
            "get_annotation",
            return_value={"labels": []},
        ), mock.patch.object(
            self.api,
            "list_dataset_versions",
            side_effect=list_dataset_versions,
        ), mock.patch.object(
            self.api,
            "update_dataset_version",
            side_effect=update_dataset_version,
        ), mock.patch(
            "client_tk.app.screens.engineer.view.messagebox.showinfo",
            return_value=None,
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()

            self.assertEqual(screen._active_dataset_version_id, "ver-1")
            screen.version_name.delete(0, "end")
            screen.version_name.insert(0, "Snapshot v1 revised")
            screen.version_description.delete(0, "end")
            screen.version_description.insert(0, "metadata update")
            screen.version_status.set("archived")

            screen.update_dataset_version_metadata()
            screen.update_idletasks()

            self.assertTrue(calls)
            dataset_id, version_id, payload = calls[-1]
            self.assertEqual(dataset_id, "ds-1")
            self.assertEqual(version_id, "ver-1")
            self.assertEqual(payload["name"], "Snapshot v1 revised")
            self.assertEqual(payload["description"], "metadata update")
            self.assertEqual(payload["status"], "archived")
            self.assertEqual(screen.version_status.get(), "archived")
            self.assertEqual(screen._active_dataset_version_id, "ver-1")

            screen.destroy()

    def test_annotation_canvas_right_click_resize_selected_bbox(self) -> None:
        host = ttk.Frame(self.root, width=220, height=220)
        host.pack(fill="both", expand=False)
        canvas = AnnotationCanvas(host, title="Test Canvas", size=(200, 200))
        canvas.pack(fill="both", expand=False)
        self.root.update_idletasks()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        canvas.load_bgr(frame, image_name="sample.png")
        canvas.set_labels(
            [
                {
                    "type": "bbox",
                    "shape_type": "bbox",
                    "class_name": "object",
                    "class": "object",
                    "bbox": {"x": 0.2, "y": 0.2, "w": 0.3, "h": 0.3},
                    "normalized": True,
                }
            ]
        )
        self.assertTrue(canvas.set_selected_label_index(0))

        # Start resize from top-left handle, then drag outward.
        canvas._on_right_click(SimpleNamespace(x=40, y=40))
        canvas._on_right_drag(SimpleNamespace(x=20, y=20))
        canvas._on_right_release(SimpleNamespace(x=20, y=20))
        self.root.update_idletasks()

        labels = canvas.get_labels()
        self.assertEqual(len(labels), 1)
        bbox = labels[0].get("bbox") or {}
        self.assertLess(float(bbox.get("x", 1.0)), 0.2)
        self.assertLess(float(bbox.get("y", 1.0)), 0.2)
        self.assertGreater(float(bbox.get("w", 0.0)), 0.3)
        self.assertGreater(float(bbox.get("h", 0.0)), 0.3)

        canvas.destroy()
        host.destroy()

    def test_engineer_delete_dataset_reselects_remaining_dataset(self) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", frame)
        self.assertTrue(ok)
        image_bytes = buffer.tobytes()

        datasets = [
            {"id": "ds-1", "name": "Dataset One"},
            {"id": "ds-2", "name": "Dataset Two"},
        ]

        def list_datasets():
            return [dict(item) for item in datasets]

        def list_dataset_files(dataset_id: str, target: str = "images"):
            image_name = "first.png" if dataset_id == "ds-1" else "second.png"
            return [{"name": image_name, "path": f"Z:/missing/{image_name}", "size": 123}]

        def download_dataset_image(dataset_id: str, image_name: str):
            return image_bytes

        def delete_dataset(dataset_id: str):
            datasets[:] = [item for item in datasets if item["id"] != dataset_id]
            return {"deleted": True, "id": dataset_id}

        with mock.patch.object(self.api, "list_datasets", side_effect=list_datasets), mock.patch.object(
            self.api,
            "list_dataset_versions",
            return_value=[],
        ), mock.patch.object(self.api, "list_dataset_files", side_effect=list_dataset_files), mock.patch.object(
            self.api,
            "download_dataset_image",
            side_effect=download_dataset_image,
        ), mock.patch.object(self.api, "delete_dataset", side_effect=delete_dataset) as delete_mock, mock.patch(
            "client_tk.app.screens.engineer.view.messagebox.askyesno",
            return_value=True,
        ):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            self.assertEqual(screen._selected_dataset_id(), "ds-1")
            self.assertEqual(screen.annot_dataset_var.get(), "Dataset One")
            self.assertEqual(screen.annot_image_var.get(), "first.png")
            self.assertIsNotNone(screen.annotation_canvas._source_frame)

            screen.dataset_list.selection_clear(0, "end")
            self.assertIsNone(screen._selected_dataset_id())

            screen.delete_dataset()

        delete_mock.assert_called_once_with("ds-1")
        self.assertEqual(screen._selected_dataset_id(), "ds-2")
        self.assertEqual(screen.annot_dataset_var.get(), "Dataset Two")
        self.assertEqual(screen.annot_image_var.get(), "second.png")
        self.assertIsNotNone(screen.annotation_canvas._source_frame)
        self.assertEqual(screen.dataset_files.size(), 1)
        screen.destroy()

    def test_engineer_layout_switches_to_compact(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        screen._ensure_models_tab_built()
        screen._ensure_calibration_tab_built()
        screen._layout_compact = None
        with mock.patch.object(screen, "winfo_width", return_value=1000), mock.patch.object(
            screen.winfo_toplevel(),
            "winfo_width",
            return_value=1000,
        ):
            screen._apply_responsive_layout()
        self.assertTrue(screen._layout_compact)
        self.assertEqual(int(screen.data_top_container.grid_info()["row"]), 0)
        self.assertEqual(int(screen.data_annotation_shell.grid_info()["row"]), 1)
        self.assertEqual(int(screen.dataset_panel.grid_info()["row"]), 0)
        self.assertEqual(int(screen.upload_panel.grid_info()["row"]), 1)
        self.assertEqual(int(screen.augment_panel.grid_info()["row"]), 0)
        self.assertEqual(int(screen.train_panel.grid_info()["row"]), 1)
        self.assertEqual(int(screen.training_jobs_panel.grid_info()["row"]), 0)
        self.assertEqual(int(screen.training_detail_panel.grid_info()["row"]), 1)
        self.assertEqual(int(screen.models_left_panel.grid_info()["row"]), 0)
        self.assertEqual(int(screen.models_right_panel.grid_info()["row"]), 1)
        self.assertEqual(int(screen.calibration_left_panel.grid_info()["row"]), 0)
        self.assertEqual(int(screen.calibration_right_outer.grid_info()["row"]), 1)
        self.assertEqual(int(screen.annotation_canvas.grid_info()["row"]), 2)
        self.assertEqual(int(screen.annotation_canvas.grid_info()["column"]), 0)
        self.assertEqual(screen.annot_prev_button.pack_info()["side"], "left")
        self.assertEqual(screen.annot_next_button.pack_info()["side"], "right")
        screen.destroy()

    def test_engineer_calibration_roi_preview_refreshes(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        screen._ensure_calibration_tab_built()
        screen.calibration_image = np.zeros((120, 200, 3), dtype=np.uint8)
        screen.calibration_image[:, :] = (10, 120, 220)
        for entry, value in (
            (screen.calib_roi_x, "0.1"),
            (screen.calib_roi_y, "0.2"),
            (screen.calib_roi_w, "0.3"),
            (screen.calib_roi_h, "0.4"),
        ):
            entry.delete(0, "end")
            entry.insert(0, value)
        screen._refresh_calibration_preview()
        screen.update_idletasks()
        self.assertIn("crop", screen.calibration_preview_info.get().lower())
        self.assertIsNotNone(screen.calibration_source_preview._photo)
        self.assertIsNotNone(screen.calibration_crop_preview._photo)
        screen.destroy()

    def test_engineer_calibration_roi_preview_shows_validation_message_for_invalid_bounds(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        screen._ensure_calibration_tab_built()
        screen.calibration_image = np.zeros((120, 200, 3), dtype=np.uint8)
        for entry, value in (
            (screen.calib_roi_x, "1"),
            (screen.calib_roi_y, "1"),
            (screen.calib_roi_w, "1"),
            (screen.calib_roi_h, "1"),
        ):
            entry.delete(0, "end")
            entry.insert(0, value)

        screen._refresh_calibration_preview()

        self.assertIn("rentang [0, 1)", screen.calibration_preview_info.get())
        screen.destroy()

    def test_engineer_calibration_roi_rejects_tiny_crop_before_compute(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        screen._ensure_calibration_tab_built()
        screen.calibration_image = np.zeros((120, 200, 3), dtype=np.uint8)
        for entry, value in (
            (screen.calib_roi_x, "0.99"),
            (screen.calib_roi_y, "0.99"),
            (screen.calib_roi_w, "0.01"),
            (screen.calib_roi_h, "0.01"),
        ):
            entry.delete(0, "end")
            entry.insert(0, value)

        screen._refresh_calibration_preview()
        self.assertIn("terlalu kecil", screen.calibration_preview_info.get().lower())

        with self.assertRaisesRegex(ValueError, "terlalu kecil"):
            screen._calibration_roi()
        screen.destroy()

    def test_scrollable_frame_dispatches_to_nested_body(self) -> None:
        container = ttk.Frame(self.root, width=240, height=140)
        container.pack_propagate(False)
        container.pack(fill="both", expand=False)

        outer = ScrollableFrame(container)
        outer.pack(fill="both", expand=True)
        for index in range(6):
            ttk.Label(outer.body, text=f"Outer Row {index}").pack(anchor="w")

        inner = ScrollableFrame(outer.body)
        inner.pack(fill="both", expand=True)
        for index in range(40):
            ttk.Label(inner.body, text=f"Inner Row {index}").pack(anchor="w")

        for index in range(6, 12):
            ttk.Label(outer.body, text=f"Outer Row {index}").pack(anchor="w")

        self.root.update_idletasks()
        outer_before = outer.canvas.yview()
        inner_before = inner.canvas.yview()

        _dispatch_mousewheel(SimpleNamespace(widget=inner.body, num=5, delta=0))
        self.root.update_idletasks()

        outer_after = outer.canvas.yview()
        inner_after = inner.canvas.yview()

        self.assertGreater(inner_after[0], inner_before[0])
        self.assertEqual(outer_after, outer_before)

    def test_live_view_keeps_fixed_size_after_image_load(self) -> None:
        container = ttk.Frame(self.root, width=360, height=260)
        container.pack_propagate(False)
        container.pack(fill="both", expand=False)

        live_view = LiveView(container, "Preview", size=(320, 200))
        live_view.pack(fill="both", expand=False)
        self.root.update_idletasks()

        initial_size = (live_view.winfo_width(), live_view.winfo_height())
        image = np.zeros((1200, 1600, 3), dtype=np.uint8)
        live_view.update_bgr(image)
        self.root.update_idletasks()

        self.assertEqual((live_view.winfo_width(), live_view.winfo_height()), initial_size)

    def test_live_view_reset_does_not_emit_ctkimage_warning(self) -> None:
        container = ttk.Frame(self.root, width=360, height=260)
        container.pack_propagate(False)
        container.pack(fill="both", expand=False)

        live_view = LiveView(container, "Preview", size=(320, 200))
        live_view.pack(fill="both", expand=False)
        self.root.update_idletasks()
        live_view.update_bgr(np.zeros((180, 240, 3), dtype=np.uint8))
        self.root.update_idletasks()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            live_view.reset()

        ctk_warnings = [
            warning
            for warning in caught
            if "Given image is not CTkImage" in str(warning.message)
        ]
        self.assertFalse(ctk_warnings)

    def test_live_view_reset_then_update_does_not_raise_tclerror(self) -> None:
        container = ttk.Frame(self.root, width=360, height=260)
        container.pack_propagate(False)
        container.pack(fill="both", expand=False)

        live_view = LiveView(container, "Preview", size=(320, 200))
        live_view.pack(fill="both", expand=False)
        self.root.update_idletasks()

        frame = np.zeros((180, 240, 3), dtype=np.uint8)
        live_view.update_bgr(frame)
        self.root.update_idletasks()
        live_view.reset()
        self.root.update_idletasks()

        try:
            live_view.update_bgr(frame)
            self.root.update_idletasks()
        except tk.TclError as exc:
            self.fail(f"LiveView update after reset raised TclError: {exc}")

    # ------------------------------------------------------------------
    # Training summary formatter regression tests
    # ------------------------------------------------------------------

    def test_format_training_percent_zero_float_shows_zero_percent(self) -> None:
        """0.0 must render as '0.00%', not '-'."""
        self.assertEqual(EngineerScreen._format_training_percent(0.0), "0.00%")

    def test_format_training_percent_zero_int_shows_zero_percent(self) -> None:
        self.assertEqual(EngineerScreen._format_training_percent(0), "0.00%")

    def test_format_training_percent_none_shows_dash(self) -> None:
        self.assertEqual(EngineerScreen._format_training_percent(None), "-")

    def test_format_training_percent_nonzero_value(self) -> None:
        self.assertEqual(EngineerScreen._format_training_percent(0.875), "87.50%")

    def test_format_training_decimal_zero_float_shows_zeros(self) -> None:
        """0.0 must render as '0.000', not '-'."""
        self.assertEqual(EngineerScreen._format_training_decimal(0.0), "0.000")

    def test_format_training_decimal_zero_int_shows_zeros(self) -> None:
        self.assertEqual(EngineerScreen._format_training_decimal(0), "0.000")

    def test_format_training_decimal_none_shows_dash(self) -> None:
        self.assertEqual(EngineerScreen._format_training_decimal(None), "-")

    def test_training_error_summary_reads_val_box_loss(self) -> None:
        """_training_error_summary falls back to val_box_loss when no RMSE/Loss present."""
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        job = {
            "status": "completed",
            "evaluation": {"val_box_loss": 0.9234},
        }
        summary = screen._training_error_summary(job)
        self.assertTrue(
            summary.startswith("Box"),
            f"Expected 'Box ...' but got: {summary!r}",
        )
        self.assertIn("0.9234", summary)
        screen.destroy()

    def test_training_error_summary_prefers_rmse_over_box_loss(self) -> None:
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        job = {
            "status": "completed",
            "metrics": {"rmse": 0.1111},
            "evaluation": {"val_box_loss": 0.9234},
        }
        summary = screen._training_error_summary(job)
        self.assertTrue(summary.startswith("RMSE"), f"Expected 'RMSE ...' but got: {summary!r}")
        screen.destroy()

    def test_apply_training_jobs_default_selects_newest_not_oldest(self) -> None:
        """When no prior selection, the newest job (last in list) is auto-selected."""
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()

        older = {"id": "job-old", "status": "completed", "dataset_id": "ds1",
                 "params": {}, "base_model": "yolov5s"}
        newer = {"id": "job-new", "status": "completed", "dataset_id": "ds1",
                 "params": {}, "base_model": "yolov5s"}
        screen._active_training_job_id = None
        screen._apply_training_jobs([older, newer])
        screen.update_idletasks()

        self.assertEqual(
            screen._active_training_job_id, "job-new",
            "Newest job should be auto-selected when no prior selection exists",
        )
        screen.destroy()

    def test_apply_training_jobs_prefers_active_over_newest(self) -> None:
        """When an active job exists, it should be preferred over the newest completed."""
        screen = EngineerScreen(self.root, self.api, self.state)
        screen.update_idletasks()

        completed = {"id": "job-done", "status": "completed", "dataset_id": "ds1",
                     "params": {}, "base_model": "yolov5s"}
        running   = {"id": "job-run",  "status": "running",   "dataset_id": "ds1",
                     "params": {}, "base_model": "yolov5s"}
        screen._active_training_job_id = None
        screen._apply_training_jobs([completed, running])
        screen.update_idletasks()

        self.assertEqual(
            screen._active_training_job_id, "job-run",
            "Active (running) job should be preferred over completed one",
        )
        screen.destroy()

    def test_engineer_rename_selected_model_calls_api(self) -> None:
        """rename_selected_model calls api.update_model with the new name and refreshes the list."""
        models = [{"id": 7, "name": "Old Name", "path": "models/m7.pt", "source": "trained"}]
        calls: list[tuple[int, dict]] = []

        def update_model(model_id: int, payload: dict):
            calls.append((model_id, dict(payload)))
            return {"id": model_id, "name": payload.get("name"), "path": "models/m7.pt", "source": "trained"}

        with mock.patch.object(self.api, "list_models", return_value=models), \
             mock.patch.object(self.api, "update_model", side_effect=update_model), \
             mock.patch("client_tk.app.screens.engineer.view.simpledialog.askstring", return_value="New Name"):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            screen.select_tab("Models")
            screen.models_list.selection_set(0)

            screen.rename_selected_model()

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0], (7, {"name": "New Name"}))
            screen.destroy()

    def test_engineer_rename_seeded_default_shows_warning_not_api(self) -> None:
        """rename_selected_model must show a warning and NOT call api.update_model for seeded-default."""
        models = [{"id": 1, "name": "AKH Sticker Detector", "path": "models/default.pt", "source": "seeded-default"}]
        api_calls: list = []

        with mock.patch.object(self.api, "list_models", return_value=models), \
             mock.patch.object(self.api, "update_model", side_effect=lambda *a, **kw: api_calls.append(a)), \
             mock.patch("client_tk.app.screens.engineer.view.messagebox.showwarning") as mock_warn:
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            screen.select_tab("Models")
            screen.models_list.selection_set(0)

            screen.rename_selected_model()

            self.assertEqual(api_calls, [], "update_model must NOT be called for seeded-default")
            mock_warn.assert_called_once()
            screen.destroy()

    def test_engineer_rename_no_selection_shows_warning(self) -> None:
        """rename_selected_model without a selection shows a warning."""
        with mock.patch.object(self.api, "list_models", return_value=[]), \
             mock.patch("client_tk.app.screens.engineer.view.messagebox.showwarning") as mock_warn:
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            screen.select_tab("Models")

            screen.rename_selected_model()

            mock_warn.assert_called_once()
            screen.destroy()

    def test_engineer_export_selected_model_writes_archive(self) -> None:
        models = [{"id": 42, "name": "Model Export", "path": "models/m42.pt", "source": "trained"}]
        export_calls: list[int] = []

        def export_model_archive(model_id: int):
            export_calls.append(model_id)
            return b"fake-zip-contents"

        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "model-42.zip"
            with mock.patch.object(self.api, "list_models", return_value=models), \
                 mock.patch.object(self.api, "export_model_archive", side_effect=export_model_archive), \
                 mock.patch("client_tk.app.screens.engineer.view.filedialog.asksaveasfilename", return_value=str(export_path)), \
                 mock.patch("client_tk.app.screens.engineer.view.messagebox.showinfo"), \
                 mock.patch("client_tk.app.screens.engineer.view.messagebox.showwarning") as mock_warn:
                screen = EngineerScreen(self.root, self.api, self.state)
                screen.update_idletasks()
                screen.select_tab("Models")
                screen.models_list.selection_set(0)
                screen.on_model_selected()

                self.assertEqual(str(screen.models_export_button.cget("state")), "normal")

                screen.export_selected_model()

                self.assertEqual(export_calls, [42])
                self.assertEqual(export_path.read_bytes(), b"fake-zip-contents")
                mock_warn.assert_not_called()
                screen.destroy()

    def test_engineer_import_model_archive_refreshes_and_selects_new_model(self) -> None:
        initial_models = [{"id": 10, "name": "Existing Model", "path": "models/existing.pt"}]
        imported_model = {"id": 77, "name": "Imported Model", "path": "models/imported.pt", "source": "import"}
        import_calls: list[tuple[str, str, bool, bool]] = []

        def import_model_archive(archive_path: str, *, target_lifecycle: str = "draft", skip_validation: bool = False, force_rename: bool = False):
            import_calls.append((archive_path, target_lifecycle, skip_validation, force_rename))
            return {"status": "success", "model_id": 77, "imported_name": "Imported Model", "warnings": []}

        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "import-model.zip"
            archive_path.write_bytes(b"fake zip archive")
            with mock.patch.object(self.api, "list_models", side_effect=[initial_models, [*initial_models, imported_model]]), \
                 mock.patch.object(self.api, "import_model_archive", side_effect=import_model_archive), \
                 mock.patch("client_tk.app.screens.engineer.view.filedialog.askopenfilename", return_value=str(archive_path)), \
                 mock.patch("client_tk.app.screens.engineer.view.messagebox.showinfo"), \
                 mock.patch("client_tk.app.screens.engineer.view.messagebox.showerror"):
                screen = EngineerScreen(self.root, self.api, self.state)
                screen.update_idletasks()
                screen.select_tab("Models")

                self.assertEqual(screen.models_list.curselection(), ())

                screen.import_model_archive()

                self.assertEqual(import_calls, [(str(archive_path), "draft", False, False)])
                self.assertEqual(screen.models_list.curselection(), (1,))
                self.assertEqual(screen._selected_model_record()["id"], 77)
                self.assertEqual(str(screen.models_export_button.cget("state")), "normal")
                screen.destroy()

    # ------------------------------------------------------------------
    # Phase 6: Model Registry selection stability
    # ------------------------------------------------------------------

    def test_engineer_models_list_exportselection_false(self) -> None:
        """models_list must have exportselection=False so highlight persists when focus moves."""
        with mock.patch.object(self.api, "list_models", return_value=[]):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            screen.select_tab("Models")
            # cget returns the configured value; False is stored as 0 in tkinter
            value = screen.models_list.cget("exportselection")
            self.assertFalse(bool(value), "models_list must use exportselection=False")
            screen.destroy()

    def test_engineer_apply_models_restores_selection_by_model_id(self) -> None:
        """_apply_models must restore the previous selection when the same model_id is still present."""
        models_v1 = [
            {"id": 10, "name": "Model A", "path": "models/a.pt"},
            {"id": 11, "name": "Model B", "path": "models/b.pt"},
        ]
        models_v2 = [
            {"id": 10, "name": "Model A (renamed)", "path": "models/a.pt"},
            {"id": 11, "name": "Model B", "path": "models/b.pt"},
            {"id": 12, "name": "Model C", "path": "models/c.pt"},
        ]
        with mock.patch.object(self.api, "list_models", return_value=models_v1):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            screen.select_tab("Models")
            # Select item at index 1 (Model B, id=11)
            screen.models_list.selection_set(1)
            screen.on_model_selected()
            self.assertEqual(len(screen._model_cache), 2)

            # Simulate refresh with an updated list (Model A renamed, Model C added)
            screen._apply_models(models_v2)
            screen.update_idletasks()

            # Selection must have been restored to the item with id=11 (now at index 1 still)
            restored_index = screen.models_list.curselection()
            self.assertTrue(restored_index, "Selection must be restored after _apply_models")
            self.assertEqual(int(restored_index[0]), 1, "Model B (id=11) must still be at index 1")
            # model_detail must also be updated to the restored item
            self.assertEqual(screen._model_cache[restored_index[0]]["id"], 11)
            screen.destroy()

    def test_engineer_apply_models_clears_detail_when_selection_gone(self) -> None:
        """When the previously selected model is removed from the list, detail is left as-is (no crash)."""
        models_v1 = [{"id": 20, "name": "Ephemeral", "path": "models/e.pt"}]
        models_v2 = [{"id": 21, "name": "New Model", "path": "models/n.pt"}]
        with mock.patch.object(self.api, "list_models", return_value=models_v1):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            screen.select_tab("Models")
            screen.models_list.selection_set(0)
            screen.on_model_selected()

            # Refresh with a list that no longer contains id=20
            screen._apply_models(models_v2)
            screen.update_idletasks()

            # No selection must remain (model is gone)
            self.assertEqual(screen.models_list.curselection(), (), "No item must be selected when model is gone")
            screen.destroy()

    def test_engineer_delete_after_selection_uses_correct_item(self) -> None:
        """delete_selected_model must use the item currently selected in the list, not a stale cache index."""
        models = [
            {"id": 30, "name": "Keep", "path": "models/keep.pt"},
            {"id": 31, "name": "Delete Me", "path": "models/del.pt"},
        ]
        deleted_ids: list[int] = []

        def delete_model(model_id: int, *, purge_files: bool = False):
            deleted_ids.append(model_id)
            return {"deleted": True, "id": model_id}

        with mock.patch.object(self.api, "list_models", return_value=models), \
             mock.patch.object(self.api, "delete_model", side_effect=delete_model), \
             mock.patch("client_tk.app.screens.engineer.view.messagebox.askyesno", return_value=True):
            screen = EngineerScreen(self.root, self.api, self.state)
            screen.update_idletasks()
            screen.select_tab("Models")
            screen.models_list.selection_set(1)   # select "Delete Me" at index 1

            screen.delete_selected_model()

            self.assertEqual(deleted_ids, [31], "Must delete model id=31 (Delete Me), not id=30")
