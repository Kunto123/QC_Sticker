from __future__ import annotations

from typing import Any

from shared.contracts.templates import InspectionTemplate


class TemplateConfigManager:
    @staticmethod
    def to_runtime_template(template: InspectionTemplate) -> dict[str, Any]:
        part_ready = template.part_ready
        sticker = template.sticker
        vision = template.vision
        return {
            "schema_version": 1,
            "template_id": f"template-{template.id or 'draft'}-v{template.version_id or template.version_number}",
            "name": template.name,
            "camera": {
                "camera_index": template.camera.camera_index,
                "width": template.camera.width,
                "height": template.camera.height,
                "fps": template.camera.fps,
            },
            "rois": {
                "part_ready": {
                    "x": template.part_ready_roi.x,
                    "y": template.part_ready_roi.y,
                    "w": template.part_ready_roi.w,
                    "h": template.part_ready_roi.h,
                },
                "sticker": {
                    "x": template.sticker_roi.x,
                    "y": template.sticker_roi.y,
                    "w": template.sticker_roi.w,
                    "h": template.sticker_roi.h,
                },
            },
            "part_ready": {
                "method": getattr(part_ready, "method", "gap_template_match") or "gap_template_match",
                "hsv_lower": list(getattr(part_ready, "hsv_lower", [0, 0, 0]) or [0, 0, 0]),
                "hsv_upper": list(getattr(part_ready, "hsv_upper", [180, 255, 80]) or [180, 255, 80]),
                "min_ratio": part_ready.min_match_ratio if part_ready.min_match_ratio is not None else 0.75,
                "stable_ms": int(getattr(part_ready, "stable_ms", 500) or 500),
                "release_ms": int(getattr(part_ready, "release_ms", 300) or 300),
            },
            "inspection": {
                "expected_class": sticker.expected_class,
                "model_path": vision.model_path,
                "model_meta_path": vision.model_meta_path,
                "yolo_confidence": vision.conf_threshold,
            },
            "runtime": {
                "state_machine": True,
                "result_hold_until_part_removed": True,
                "save_result_to_db": template.persistence.write_to_db,
                "plc_enabled": True,
            },
        }

