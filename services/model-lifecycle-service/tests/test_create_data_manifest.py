from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
from zipfile import ZipFile

from src.applications.use_cases.create_data_manifest import CreateDataManifest


class CreateDataManifestTest(unittest.TestCase):
    def test_execute_builds_detection_manifest_and_writes_zip_and_json_outputs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            detection_root = root / "detection"
            (detection_root / "train" / "images").mkdir(parents=True)
            (detection_root / "train" / "labels").mkdir(parents=True)
            (detection_root / "val" / "images").mkdir(parents=True)
            (detection_root / "val" / "labels").mkdir(parents=True)

            (detection_root / "data.yaml").write_text(
                "names:\n  0: handwriting_text\n",
                encoding="utf-8",
            )

            self._write_bmp_image(detection_root / "train" / "images" / "img_1.bmp", rgb=(255, 0, 0))
            self._write_bmp_image(detection_root / "train" / "images" / "img_2.bmp", rgb=(0, 255, 0))
            self._write_bmp_image(detection_root / "val" / "images" / "img_3.bmp", rgb=(0, 0, 255))

            (detection_root / "train" / "labels" / "img_1.txt").write_text(
                "0 0.5 0.5 0.4 0.3\n",
                encoding="utf-8",
            )
            (detection_root / "train" / "labels" / "img_2.txt").write_text(
                "0 0.4 0.4 0.2 0.2\n",
                encoding="utf-8",
            )
            (detection_root / "val" / "labels" / "img_3.txt").write_text(
                "0 0.6 0.6 0.2 0.2\n",
                encoding="utf-8",
            )

            manifest = CreateDataManifest(
                data_types="detection",
                logger=Mock(),
            ).execute(
                str(detection_root),
                dataset_version="v1",
                dataset_name="handwriting_detection",
                storage_config={
                    "provider": "minio",
                    "bucket": "data",
                    "prefix": "ml/training/data/detection/v1",
                    "archive_uri": "minio://data/ml/training/data/detection/v1/detection.zip",
                },
            )

            archive_path = root / "detection.zip"
            manifest_path = root / "detection_dataset_manifest.json"

            self.assertTrue(archive_path.exists())
            self.assertTrue(manifest_path.exists())

            self.assertEqual(manifest["schema_version"], "1.0")
            self.assertEqual(manifest["dataset_name"], "handwriting_detection")
            self.assertEqual(manifest["dataset_version"], "v1")
            self.assertEqual(manifest["task"], "detection")
            self.assertTrue(manifest["created_at"].endswith("Z"))

            self.assertEqual(manifest["storage"]["provider"], "minio")
            self.assertEqual(manifest["storage"]["bucket"], "data")
            self.assertEqual(manifest["archive"]["format"], "zip")
            self.assertEqual(manifest["archive"]["compression"], "deflate")
            self.assertEqual(manifest["archive"]["file_path"], str(archive_path.resolve()))
            self.assertGreater(manifest["archive"]["size_bytes"], 0)
            self.assertEqual(manifest["archive"]["size_bytes"], archive_path.stat().st_size)
            self.assertTrue(manifest["archive"]["checksum_sha256"].startswith("sha256:"))

            self.assertEqual(manifest["data_schema"]["label_format"], "yolo")
            self.assertEqual(manifest["data_schema"]["classes"], [{"id": 0, "name": "handwriting_text"}])

            self.assertEqual(manifest["expected_structure"]["root_dir"], "detection")
            self.assertEqual(manifest["expected_structure"]["data_yaml"], "detection/data.yaml")
            self.assertEqual(manifest["expected_structure"]["splits"]["train"]["images_samples"], 2)
            self.assertEqual(manifest["expected_structure"]["splits"]["train"]["labels_samples"], 2)
            self.assertEqual(manifest["expected_structure"]["splits"]["val"]["images_samples"], 1)
            self.assertEqual(manifest["expected_structure"]["splits"]["val"]["labels_samples"], 1)
            self.assertFalse(manifest["expected_structure"]["splits"]["test"]["enabled"])

            self.assertEqual(manifest["stats"]["total_images"], 3)
            self.assertEqual(manifest["stats"]["total_labels"], 3)
            self.assertEqual(manifest["stats"]["train_ratio"], 0.6667)
            self.assertEqual(manifest["stats"]["val_ratio"], 0.3333)
            self.assertEqual(manifest["stats"]["test_ratio"], 0.0)
            self.assertEqual(
                manifest["image_statistics"],
                {
                    "train": {
                        "mean_rgb": {"red": 0.5, "green": 0.5, "blue": 0.0},
                        "std_rgb": {"red": 0.5, "green": 0.5, "blue": 0.0},
                    },
                    "val": {
                        "mean_rgb": {"red": 0.0, "green": 0.0, "blue": 1.0},
                        "std_rgb": {"red": 0.0, "green": 0.0, "blue": 0.0},
                    },
                },
            )

            self.assertEqual(manifest["quality"]["validation_status"], "passed")
            self.assertEqual(manifest["quality"]["missing_images"], 0)
            self.assertEqual(manifest["quality"]["missing_labels"], 0)
            self.assertEqual(manifest["quality"]["invalid_labels"], 0)
            self.assertEqual(manifest["quality"]["corrupted_images"], 0)
            self.assertEqual(manifest["generation_policy"]["split_method"], "random")
            self.assertEqual(manifest["generation_policy"]["random_seed"], 42)

            with manifest_path.open("r", encoding="utf-8") as file:
                persisted_manifest = json.load(file)
            self.assertEqual(persisted_manifest, manifest)

            with ZipFile(archive_path, "r") as zip_file:
                archived_files = set(zip_file.namelist())
            self.assertIn("data.yaml", archived_files)
            self.assertIn("train/images/img_1.bmp", archived_files)
            self.assertIn("train/images/img_2.bmp", archived_files)
            self.assertIn("val/images/img_3.bmp", archived_files)

    def test_execute_builds_recognition_manifest_from_valid_mapped_samples_only(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            recognizer_root = root / "recognizer"
            (recognizer_root / "train_data" / "batch_a").mkdir(parents=True)
            (recognizer_root / "val_data" / "batch_b").mkdir(parents=True)

            self._write_bmp_image(
                recognizer_root / "train_data" / "batch_a" / "img_1.bmp",
                rgb=(255, 0, 0),
            )
            self._write_bmp_image(
                recognizer_root / "train_data" / "batch_a" / "img_orphan.bmp",
                rgb=(0, 255, 0),
            )
            self._write_bmp_image(
                recognizer_root / "val_data" / "batch_b" / "img_2.bmp",
                rgb=(0, 0, 255),
            )
            (recognizer_root / "train_data" / "batch_a" / "label.json").write_text(
                json.dumps(
                    {
                        "img_1.bmp": "thức",
                        "img_missing.bmp": "bỏ",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (recognizer_root / "val_data" / "batch_b" / "label.json").write_text(
                json.dumps({"img_2.bmp": "A1"}, ensure_ascii=False),
                encoding="utf-8",
            )

            manifest = CreateDataManifest(
                data_types="recognition",
                logger=Mock(),
            ).execute(
                str(recognizer_root),
                dataset_version="v1",
                dataset_name="handwriting_recognizer",
            )

            archive_path = root / "recognizer.zip"
            manifest_path = root / "recognition_dataset_manifest.json"

            self.assertTrue(archive_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertEqual(manifest["dataset_name"], "handwriting_recognizer")
            self.assertEqual(manifest["task"], "recognition")
            self.assertEqual(manifest["data_schema"]["label_format"], "label_json")
            self.assertEqual(
                manifest["expected_structure"]["splits"]["train"]["samples"],
                1,
            )
            self.assertEqual(
                manifest["expected_structure"]["splits"]["val"]["samples"],
                1,
            )
            self.assertEqual(
                manifest["stats"],
                {
                    "total_samples": 2,
                    "train_ratio": 0.5,
                    "val_ratio": 0.5,
                    "test_ratio": 0.0,
                },
            )
            self.assertEqual(
                manifest["image_statistics"],
                {
                    "train": {
                        "mean_rgb": {"red": 1.0, "green": 0.0, "blue": 0.0},
                        "std_rgb": {"red": 0.0, "green": 0.0, "blue": 0.0},
                    },
                    "val": {
                        "mean_rgb": {"red": 0.0, "green": 0.0, "blue": 1.0},
                        "std_rgb": {"red": 0.0, "green": 0.0, "blue": 0.0},
                    },
                },
            )
            self.assertEqual(manifest["quality"]["missing_images"], 1)
            self.assertEqual(manifest["quality"]["missing_labels"], 1)
            self.assertEqual(manifest["quality"]["corrupted_images"], 0)

            with manifest_path.open("r", encoding="utf-8") as file:
                persisted_manifest = json.load(file)
            self.assertEqual(persisted_manifest, manifest)

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
