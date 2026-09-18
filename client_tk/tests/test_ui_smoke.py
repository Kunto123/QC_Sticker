from __future__ import annotations

import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from tkinter import ttk

try:
    import tkinter as tk
except Exception:  # noqa: BLE001
    tk = None

from client_tk.app.screens.admin.view import AdminScreen
from client_tk.app.components.live_view import LiveView

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

    def import_model_archive(self, archive_path: str, *, name: str | None = None, target_lifecycle: str = "draft", skip_validation: bool = False, force_rename: bool = False):
        return {
            "status": "success",
            "model_id": 101,
            "imported_name": name or Path(archive_path).stem,
            "warnings": [],
            "lifecycle_status": target_lifecycle,
        }

    def get_runtime_template(self, version_id: int):
        return {"schema_version": 1, "template_id": f"template-v{version_id}", "name": "QC Line A"}

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








    


    
    
    def update_model(self, model_id: int, payload: dict):
        return {"id": model_id, **payload}

    def delete_model(self, model_id: int, *, purge_files: bool = False):
        return {"deleted": True, "id": model_id, "purge_files": purge_files}
    
    def list_workstations(self):
        return []
    
    def delete_workstation(self, machine_id: str):
        return {"deleted": True, "machine_id": machine_id}

    # Machine Settings tab
    def get_machine_settings(self):
        from backend.app.models.machine_settings import MachineSettings

        return {**MachineSettings().to_dict(), "restart_required": False}

    def update_machine_settings(self, payload: dict):
        return {**self.get_machine_settings(), **payload}

    def get_plc_diagnostics(self):
        return {"enabled": False}

    def test_plc_coil(self, address: int, duration_ms: int, confirm: bool = True):
        return {"ok": True}

    def plc_all_off(self):
        return {"ok": True}



@unittest.skipIf(tk is None, "Tkinter is not available in this environment")
class UiSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
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
            mock.patch("client_tk.app.screens.admin.tabs.machine_settings_tab.messagebox.showerror", new=lambda *a, **k: None),
        ]
        for patcher in self._async_patchers:
            patcher.start()

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

    def test_admin_screen_initializes(self) -> None:
        screen = AdminScreen(self.root, self.api, self.state)
        screen.update_idletasks()
        self.assertTrue(screen.winfo_exists())
        self.assertEqual(screen._notebook.tabs(), ["Templates", "Models", "Operators", "Monitor", "Machine Settings"])
        self.assertNotIn("Raw JSON", screen._notebook.tabs())
        self.assertNotIn("Engineer", screen._notebook.tabs())
        self.assertFalse(hasattr(screen, "template_form"))
        self.assertFalse(hasattr(screen, "workstation_tools_screen"))
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


