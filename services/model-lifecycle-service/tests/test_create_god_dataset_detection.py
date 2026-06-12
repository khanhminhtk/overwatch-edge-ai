from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
from zipfile import ZipFile

from src.applications.use_cases.create_god_dataset_detection import CreateGodDatasetDetection


class CreateGodDatasetDetectionTest(unittest.TestCase):
    def test_execute_creates_train_only_subset_zip_and_manifest(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            detection_root = root / "detection"
            (detection_root / "train" / "images").mkdir(parents=True)
            (detection_root / "train" / "labels").mkdir(parents=True)
            (detection_root / "val" / "images").mkdir(parents=True)
            (detection_root / "val" / "labels").mkdir(parents=True)

            (detection_root / "data.yaml").write_text("names:\n  0: text\n", encoding="utf-8")

            self._write_bmp_image(detection_root / "train" / "images" / "img_1.bmp", rgb=(255, 0, 0))
            self._write_bmp_image(detection_root / "train" / "images" / "img_2.bmp", rgb=(0, 255, 0))
            self._write_bmp_image(detection_root / "train" / "images" / "img_3.bmp", rgb=(0, 0, 255))
            self._write_bmp_image(detection_root / "train" / "images" / "img_4.bmp", rgb=(255, 255, 255))
            self._write_bmp_image(detection_root / "val" / "images" / "img_val.bmp", rgb=(0, 0, 0))

            for stem in ("img_1", "img_2", "img_3", "img_4"):
                (detection_root / "train" / "labels" / f"{stem}.txt").write_text(
                    "0 0.5 0.5 0.2 0.2\n",
                    encoding="utf-8",
                )
            (detection_root / "val" / "labels" / "img_val.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )

            result = CreateGodDatasetDetection(logger=Mock()).execute(
                source_data_path=str(detection_root),
                subset_percent=50,
            )

            dataset_root = root / "god_dataset_detection"
            archive_path = root / "god_dataset_detection.zip"
            manifest_path = root / "god_dataset_detection_dataset_manifest.json"

            self.assertEqual(result["dataset_root"], str(dataset_root.resolve()))
            self.assertEqual(result["archive_path"], str(archive_path.resolve()))
            self.assertEqual(result["manifest_path"], str(manifest_path.resolve()))
            self.assertEqual(result["selected_samples"], 2)

            self.assertTrue((dataset_root / "train" / "images").exists())
            self.assertTrue((dataset_root / "train" / "labels").exists())
            self.assertFalse((dataset_root / "val").exists())
            self.assertTrue(archive_path.exists())
            self.assertTrue(manifest_path.exists())

            selected_images = sorted(path.name for path in (dataset_root / "train" / "images").iterdir())
            selected_labels = sorted(path.name for path in (dataset_root / "train" / "labels").iterdir())
            self.assertEqual(len(selected_images), 2)
            self.assertEqual(selected_labels, [name.replace(".bmp", ".txt") for name in selected_images])

            with manifest_path.open("r", encoding="utf-8") as file:
                manifest = json.load(file)
            self.assertEqual(manifest["dataset_name"], "god_dataset_detection")
            self.assertTrue(manifest["expected_structure"]["splits"]["train"]["enabled"])
            self.assertEqual(manifest["expected_structure"]["splits"]["train"]["images_samples"], 2)
            self.assertFalse(manifest["expected_structure"]["splits"]["val"]["enabled"])
            self.assertFalse(manifest["expected_structure"]["splits"]["test"]["enabled"])

            with ZipFile(archive_path, "r") as zip_file:
                archived_files = set(zip_file.namelist())
            self.assertIn("train/images/" + selected_images[0], archived_files)
            self.assertIn("train/images/" + selected_images[1], archived_files)

    def test_execute_rejects_invalid_subset_percent(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            detection_root = root / "detection"
            (detection_root / "train" / "images").mkdir(parents=True)
            (detection_root / "train" / "labels").mkdir(parents=True)
            (detection_root / "data.yaml").write_text("names:\n  0: text\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "subset_percent"):
                CreateGodDatasetDetection(logger=Mock()).execute(
                    source_data_path=str(detection_root),
                    subset_percent=0,
                )

    def test_execute_skips_missing_label_samples_before_selection(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            detection_root = root / "detection"
            (detection_root / "train" / "images").mkdir(parents=True)
            (detection_root / "train" / "labels").mkdir(parents=True)
            (detection_root / "data.yaml").write_text("names:\n  0: text\n", encoding="utf-8")

            self._write_bmp_image(detection_root / "train" / "images" / "img_1.bmp", rgb=(255, 0, 0))
            self._write_bmp_image(detection_root / "train" / "images" / "img_2.bmp", rgb=(0, 255, 0))
            (detection_root / "train" / "labels" / "img_1.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )

            result = CreateGodDatasetDetection(logger=Mock()).execute(
                source_data_path=str(detection_root),
                subset_percent=100,
            )

            self.assertEqual(result["selected_samples"], 1)

    def test_execute_replaces_previous_god_dataset_output(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            detection_root = root / "detection"
            (detection_root / "train" / "images").mkdir(parents=True)
            (detection_root / "train" / "labels").mkdir(parents=True)
            (detection_root / "data.yaml").write_text("names:\n  0: text\n", encoding="utf-8")

            self._write_bmp_image(detection_root / "train" / "images" / "img_1.bmp", rgb=(255, 0, 0))
            self._write_bmp_image(detection_root / "train" / "images" / "img_2.bmp", rgb=(0, 255, 0))
            (detection_root / "train" / "labels" / "img_1.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )
            (detection_root / "train" / "labels" / "img_2.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )

            use_case = CreateGodDatasetDetection(logger=Mock())
            use_case.execute(source_data_path=str(detection_root), subset_percent=50)
            use_case.execute(source_data_path=str(detection_root), subset_percent=50)

            selected_images = list((root / "god_dataset_detection" / "train" / "images").iterdir())
            self.assertEqual(len(selected_images), 1)

    @staticmethod
    def _write_bmp_image(path: Path, *, rgb: tuple[int, int, int]) -> None:
        red, green, blue = rgb
        bgr_bytes = bytes((blue, green, red))
        pixel_data = bgr_bytes + b"\x00"
        file_size = 14 + 40 + len(pixel_data)
        file_header = b"BM" + struct.pack("<IHHI", file_size, 0, 0, 54)
        dib_header = struct.pack(
            "<IIIHHIIIIII",
            40,
            1,
            1,
            1,
            24,
            0,
            len(pixel_data),
            2835,
            2835,
            0,
            0,
        )
        path.write_bytes(file_header + dib_header + pixel_data)


if __name__ == "__main__":
    unittest.main()
