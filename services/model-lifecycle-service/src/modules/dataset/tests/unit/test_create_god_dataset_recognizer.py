from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

import cv2
import numpy as np

from src.modules.dataset.application.use_case.create_god_dataset_recognizer import (
    CreateGodDatasetRecognizer,
)
from src.platform.logger import Logger


def _make_fake_image(path: Path) -> None:
    img = (np.random.rand(32, 64, 3) * 255).astype("uint8")
    cv2.imwrite(str(path), img)


class CreateGodDatasetRecognizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.source = self.tmp / "source"
        train_data = self.source / "train_data" / "group_001"
        train_data.mkdir(parents=True)
        labels = {}
        for i in range(10):
            _make_fake_image(train_data / f"img_{i:03d}.jpg")
            labels[f"img_{i:03d}.jpg"] = f"label_{i}"
        (train_data / "label.json").write_text(json.dumps(labels))
        self.output = self.tmp / "output"
        self.output.mkdir()
        self.logger = MagicMock(spec=Logger)
        self.creator = CreateGodDatasetRecognizer(logger=self.logger)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_raises_on_invalid_percent(self) -> None:
        with self.assertRaises(ValueError):
            self.creator.execute(source_data_path=str(self.source), subset_percent=101)

    def test_raises_on_missing_source(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.creator.execute(source_data_path="/nonexistent", subset_percent=10)

    def test_selects_all_when_100_percent(self) -> None:
        result = self.creator.execute(
            source_data_path=str(self.source),
            subset_percent=100,
            output_root=str(self.output),
        )
        self.assertEqual(result.selected_samples, 8)
        self.assertTrue(Path(result.archive_path).exists())
        self.assertTrue(Path(result.manifest_path).exists())

    def test_output_structure(self) -> None:
        result = self.creator.execute(
            source_data_path=str(self.source),
            subset_percent=100,
            output_root=str(self.output),
        )
        subset = Path(result.dataset_root)
        self.assertTrue((subset / "train_data" / "group_001").exists())
        self.assertTrue((subset / "train_data" / "group_001" / "label.json").exists())


if __name__ == "__main__":
    unittest.main()
