from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
from zipfile import ZipFile

from src.applications.use_cases.create_data_manifest import RecognitionSample
from src.applications.use_cases.create_god_dataset_recognizer import (
    CreateGodDatasetRecognizer,
    RecognizerSampleFeature,
)


class CreateGodDatasetRecognizerTest(unittest.TestCase):
    def test_execute_creates_train_only_subset_with_label_json(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            recognizer_root = root / "recognizer"
            batch_dir = recognizer_root / "train_data" / "batch_a"
            batch_dir.mkdir(parents=True)

            labels = {
                "img_1.bmp": "aa",
                "img_2.bmp": "aa",
                "img_3.bmp": "bb",
                "img_4.bmp": "bb",
                "img_5.bmp": "cc",
                "img_missing.bmp": "drop",
            }
            for image_name in ("img_1.bmp", "img_2.bmp", "img_3.bmp", "img_4.bmp", "img_5.bmp"):
                self._write_bmp_image(batch_dir / image_name, rgb=(255, 255, 255))
            (batch_dir / "label.json").write_text(
                json.dumps(labels, ensure_ascii=False),
                encoding="utf-8",
            )
            (recognizer_root / "test_data" / "ignored").mkdir(parents=True)
            self._write_bmp_image(
                recognizer_root / "test_data" / "ignored" / "test_only.bmp",
                rgb=(0, 0, 0),
            )
            (recognizer_root / "test_data" / "ignored" / "label.json").write_text(
                json.dumps({"test_only.bmp": "zzz"}, ensure_ascii=False),
                encoding="utf-8",
            )

            result = CreateGodDatasetRecognizer(logger=Mock()).execute(
                source_data_path=str(recognizer_root),
                subset_percent=50,
            )

            dataset_root = root / "god_dataset_recognizer"
            archive_path = root / "god_dataset_recognizer.zip"
            manifest_path = root / "god_dataset_recognizer_dataset_manifest.json"

            self.assertEqual(result["dataset_root"], str(dataset_root.resolve()))
            self.assertEqual(result["archive_path"], str(archive_path.resolve()))
            self.assertEqual(result["manifest_path"], str(manifest_path.resolve()))
            self.assertEqual(result["selected_samples"], 2)

            self.assertTrue((dataset_root / "train_data").exists())
            self.assertFalse((dataset_root / "val_data").exists())
            self.assertFalse((dataset_root / "test_data").exists())

            label_files = sorted((dataset_root / "train_data").glob("*/label.json"))
            self.assertEqual(len(label_files), 1)
            with label_files[0].open("r", encoding="utf-8") as file:
                selected_labels = json.load(file)

            self.assertEqual(len(selected_labels), 2)
            self.assertEqual(sorted(selected_labels.values()), ["aa", "bb"])
            self.assertEqual(
                sorted(path.name for path in label_files[0].parent.iterdir() if path.suffix == ".bmp"),
                sorted(selected_labels.keys()),
            )

            with manifest_path.open("r", encoding="utf-8") as file:
                manifest = json.load(file)
            self.assertEqual(manifest["dataset_name"], "god_dataset_recognizer")
            self.assertEqual(manifest["task"], "recognition")
            self.assertEqual(manifest["expected_structure"]["splits"]["train"]["samples"], 2)
            self.assertFalse(manifest["expected_structure"]["splits"]["val"]["enabled"])
            self.assertFalse(manifest["expected_structure"]["splits"]["test"]["enabled"])

            with ZipFile(archive_path, "r") as zip_file:
                archived_files = set(zip_file.namelist())
            for image_name in selected_labels:
                self.assertIn(f"train_data/batch_a/{image_name}", archived_files)
            self.assertNotIn("test_data/ignored/test_only.bmp", archived_files)

    def test_execute_rejects_invalid_subset_percent(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            recognizer_root = root / "recognizer"
            batch_dir = recognizer_root / "train_data" / "batch_a"
            batch_dir.mkdir(parents=True)
            self._write_bmp_image(batch_dir / "img_1.bmp", rgb=(255, 255, 255))
            (batch_dir / "label.json").write_text(
                json.dumps({"img_1.bmp": "aa"}, ensure_ascii=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "subset_percent"):
                CreateGodDatasetRecognizer(logger=Mock()).execute(
                    source_data_path=str(recognizer_root),
                    subset_percent=0,
                )

    def test_select_subset_keeps_label_distribution_by_largest_remainder(self) -> None:
        use_case = CreateGodDatasetRecognizer(logger=Mock())
        candidates = [
            self._build_feature(use_case, "img_1.bmp", "aa"),
            self._build_feature(use_case, "img_2.bmp", "aa"),
            self._build_feature(use_case, "img_3.bmp", "aa"),
            self._build_feature(use_case, "img_4.bmp", "bb"),
            self._build_feature(use_case, "img_5.bmp", "bb"),
        ]

        selected = use_case._select_subset(candidates, 50, split_seed=0)

        self.assertEqual(sorted(feature.sample.label for feature in selected), ["aa", "bb"])

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

    @staticmethod
    def _build_feature(
        use_case: CreateGodDatasetRecognizer,
        image_name: str,
        label: str,
    ) -> RecognizerSampleFeature:
        sample = RecognitionSample(
            split_name="train",
            group_name="batch_a",
            image_name=image_name,
            image_path=Path(f"/tmp/{image_name}"),
            label=label,
        )
        return RecognizerSampleFeature(
            sample=sample,
            pixel_count=1,
            sum_rgb=(1.0, 1.0, 1.0),
            sum_sq_rgb=(1.0, 1.0, 1.0),
            character_counts=use_case._count_characters(label),
        )


if __name__ == "__main__":
    unittest.main()
