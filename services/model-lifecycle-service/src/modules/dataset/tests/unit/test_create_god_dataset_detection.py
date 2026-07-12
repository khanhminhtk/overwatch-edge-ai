from __future__ import annotations

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

from src.modules.dataset.application.use_case.create_god_dataset_detection import (
    CreateGodDatasetDetection,
)
from src.platform.logger import Logger


def _make_fake_image(path: Path) -> None:
    img = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    cv2.imwrite(str(path), img)


class CreateGodDatasetDetectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.source = self.tmp / "source"
        (self.source / "train" / "images").mkdir(parents=True)
        (self.source / "train" / "labels").mkdir(parents=True)
        for i in range(20):
            _make_fake_image(self.source / "train" / "images" / f"img_{i}.jpg")
            (self.source / "train" / "labels" / f"img_{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        self.output = self.tmp / "output"
        self.output.mkdir()
        self.logger = MagicMock(spec=Logger)
        self.creator = CreateGodDatasetDetection(logger=self.logger)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_raises_on_invalid_percent(self) -> None:
        with self.assertRaises(ValueError):
            self.creator.execute(source_data_path=str(self.source), subset_percent=0)

    def test_raises_on_missing_source(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.creator.execute(source_data_path="/nonexistent", subset_percent=10)

    def test_selects_correct_count(self) -> None:
        result = self.creator.execute(
            source_data_path=str(self.source),
            subset_percent=50,
            output_root=str(self.output),
        )
        self.assertEqual(result.selected_samples, 10)

    def test_output_structure(self) -> None:
        result = self.creator.execute(
            source_data_path=str(self.source),
            subset_percent=100,
            output_root=str(self.output),
        )
        subset = Path(result.dataset_root)
        self.assertTrue((subset / "train" / "images").exists())
        self.assertTrue((subset / "train" / "labels").exists())
        img_count = len(list((subset / "train" / "images").iterdir()))
        lbl_count = len(list((subset / "train" / "labels").iterdir()))
        self.assertEqual(img_count, 20)
        self.assertEqual(lbl_count, 20)
        self.assertTrue(Path(result.archive_path).exists())
        self.assertTrue(Path(result.manifest_path).exists())


if __name__ == "__main__":
    unittest.main()
