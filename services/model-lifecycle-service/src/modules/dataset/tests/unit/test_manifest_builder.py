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

from src.modules.dataset.application.manifest_builder import ManifestBuilder
from src.platform.logger import Logger


class ManifestBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "train" / "images").mkdir(parents=True)
        for i in range(3):
            img = (np.random.rand(64, 64, 3) * 255).astype("uint8")
            cv2.imwrite(str(self.tmp / "train" / "images" / f"img_{i}.jpg"), img)
        self.logger = MagicMock(spec=Logger)
        self.builder = ManifestBuilder(data_types="detection", logger=self.logger)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_build_detection_manifest(self) -> None:
        manifest, archive, manifest_path = self.builder.build(self.tmp, dataset_version="v1")
        self.assertIn("schema_version", manifest)
        self.assertIn("expected_structure", manifest)
        self.assertEqual(manifest["task"], "detection")
        self.assertTrue(archive.exists())
        self.assertTrue(manifest_path.exists())

    def test_detection_splits(self) -> None:
        splits = self.builder._detection_splits(self.tmp)
        self.assertIn("train", splits)
        self.assertTrue(splits["train"]["enabled"])
        self.assertEqual(splits["train"]["images"], 3)

    def test_raises_on_unsupported_type(self) -> None:
        with self.assertRaises(ValueError):
            ManifestBuilder(data_types="unsupported", logger=self.logger)


if __name__ == "__main__":
    unittest.main()
