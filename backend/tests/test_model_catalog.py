from __future__ import annotations

import unittest

from backend.app.core.model_catalog import get_base_model, list_base_models, resolve_base_model


class ModelCatalogTest(unittest.TestCase):
    def test_catalog_lists_all_yolo_variants(self) -> None:
        items = list_base_models()
        self.assertEqual(len(items), 10)
        self.assertEqual(items[0]["id"], "yolov5n")
        self.assertEqual(items[-1]["id"], "yolov11x")

    def test_resolve_base_model_supports_key_and_family_variant(self) -> None:
        by_key = resolve_base_model("yolov5s")
        self.assertIsNotNone(by_key)
        self.assertEqual(by_key["family"], "yolov5")
        self.assertEqual(by_key["variant"], "s")

        by_family_variant = resolve_base_model(family="yolov11", variant="m")
        self.assertIsNotNone(by_family_variant)
        self.assertEqual(by_family_variant["id"], "yolov11m")
        self.assertEqual(by_family_variant["display_name"], "YOLOv11 Medium")

    def test_get_base_model_returns_none_for_unknown_id(self) -> None:
        self.assertIsNone(get_base_model("does-not-exist"))

    def test_yolo11_catalog_uses_canonical_weights_name(self) -> None:
        """YOLO11 catalog entries must use 'yolo11' (not 'yolov11') as the weights prefix."""
        for variant in ("n", "s", "m", "l", "x"):
            with self.subTest(variant=variant):
                item = get_base_model(f"yolov11{variant}")
                self.assertIsNotNone(item)
                self.assertEqual(item["weights_name"], f"yolo11{variant}.pt")

    def test_yolov5_catalog_weights_name_unaffected(self) -> None:
        """YOLOv5 weights names must stay as 'yolov5*.pt' — no unintended rename."""
        for variant in ("n", "s", "m", "l", "x"):
            with self.subTest(variant=variant):
                item = get_base_model(f"yolov5{variant}")
                self.assertIsNotNone(item)
                self.assertEqual(item["weights_name"], f"yolov5{variant}.pt")
