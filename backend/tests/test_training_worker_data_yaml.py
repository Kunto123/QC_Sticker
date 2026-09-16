from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.workers.training_worker import TrainingWorker


class TrainingWorkerDataYamlTest(unittest.TestCase):
    def test_normalize_data_yaml_rewrites_relative_path_to_absolute(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qc-suite-training-worker-") as temp_dir:
            export_root = Path(temp_dir)
            data_yaml = export_root / "data.yaml"
            data_yaml.write_text(
                "path: .\n"
                "train: images/train\n"
                "val: images/valid\n"
                "test: images/test\n",
                encoding="utf-8",
            )

            TrainingWorker._normalize_data_yaml(data_yaml_path=data_yaml, export_root=export_root)

            content = data_yaml.read_text(encoding="utf-8")
            self.assertIn(f"path: {export_root.resolve().as_posix()}", content)
            self.assertIn("train: images/train", content)
            self.assertIn("val: images/valid", content)

    def test_normalize_data_yaml_adds_path_when_missing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qc-suite-training-worker-") as temp_dir:
            export_root = Path(temp_dir)
            data_yaml = export_root / "data.yaml"
            data_yaml.write_text(
                "train: images/train\n"
                "val: images/valid\n"
                "test: images/test\n",
                encoding="utf-8",
            )

            TrainingWorker._normalize_data_yaml(data_yaml_path=data_yaml, export_root=export_root)

            content = data_yaml.read_text(encoding="utf-8")
            self.assertTrue(content.startswith(f"path: {export_root.resolve().as_posix()}\n"))
