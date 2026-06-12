from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any, Literal

import yaml

from src.utils.logger import Logger
from src.utils.zipfiledata import zip_directory


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASS_NAME = "handwriting_text"
DEFAULT_DATA_PATH = (
    "/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus/ml/training/data/detection"
)


@dataclass(frozen=True)
class RecognitionSample:
    split_name: str
    group_name: str
    image_name: str
    image_path: Path
    label: str


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class CreateDataManifest:
    def __init__(
        self,
        data_types: Literal["detection", "recognition"],
        logger: Logger,
    ) -> None:
        self._data_types = data_types
        self._logger = logger

    def execute(
        self,
        data_path: str,
        *,
        dataset_version: str = "v1",
        dataset_name: str | None = None,
        storage_config: dict[str, str] | None = None,
        archive_path: str | None = None,
        generation_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started_at = datetime.now(timezone.utc)
        data_root = self._resolve_data_root(data_path)
        storage_config = storage_config or {}
        generation_policy = generation_policy or {}

        if self._data_types == "detection":
            manifest, archive_file, manifest_path = self._execute_detection(
                data_root=data_root,
                dataset_version=dataset_version,
                dataset_name=dataset_name,
                storage_config=storage_config,
                archive_path=archive_path,
                generation_policy=generation_policy,
            )
        elif self._data_types == "recognition":
            manifest, archive_file, manifest_path = self._execute_recognition(
                data_root=data_root,
                dataset_version=dataset_version,
                dataset_name=dataset_name,
                storage_config=storage_config,
                archive_path=archive_path,
                generation_policy=generation_policy,
            )
        else:
            raise NotImplementedError(f"Unsupported data_types: {self._data_types}")

        duration_ms = int((datetime.now(timezone.utc) - started_at).total_seconds() * 1000)
        self._logger.info(
            "[CREATE_DATA_MANIFEST_OK]",
            f"task={self._data_types}",
            f"data_path={data_path}",
            f"dataset_version={dataset_version}",
            f"archive_path={archive_file}",
            f"manifest_path={manifest_path}",
            f"duration_ms={duration_ms}",
        )
        return manifest

    def _execute_detection(
        self,
        *,
        data_root: Path,
        dataset_version: str,
        dataset_name: str | None,
        storage_config: dict[str, str],
        archive_path: str | None,
        generation_policy: dict[str, Any],
    ) -> tuple[dict[str, Any], Path, Path]:
        classes = self._load_classes(data_root / "data.yaml")
        split_manifests = self._build_split_manifests(data_root)
        quality = self._build_quality(data_root, split_manifests, classes)
        stats = self._build_stats(split_manifests)
        image_statistics = self._build_image_statistics(data_root, split_manifests)
        archive_file = self._build_archive(data_root, archive_path)

        manifest = {
            "schema_version": "1.0",
            "dataset_name": dataset_name or "handwriting_detection",
            "dataset_version": dataset_version,
            "task": "detection",
            "created_at": _utcnow_iso(),
            "storage": self._build_storage(data_root.name, dataset_version, storage_config),
            "archive": self._build_archive_metadata(archive_file),
            "data_schema": {
                "input_type": "image",
                "label_format": "yolo",
                "coordinate_format": "normalized_xywh",
                "classes": classes,
            },
            "expected_structure": {
                "root_dir": data_root.name,
                "data_yaml": f"{data_root.name}/data.yaml",
                "splits": split_manifests,
            },
            "stats": stats,
            "image_statistics": image_statistics,
            "quality": quality,
            "generation_policy": {
                "split_method": str(generation_policy.get("split_method", "random")),
                "random_seed": int(generation_policy.get("random_seed", 42)),
            },
        }

        manifest_path = archive_file.parent / "detection_dataset_manifest.json"
        self._write_manifest_json(manifest_path, manifest)
        return manifest, archive_file, manifest_path

    def _execute_recognition(
        self,
        *,
        data_root: Path,
        dataset_version: str,
        dataset_name: str | None,
        storage_config: dict[str, str],
        archive_path: str | None,
        generation_policy: dict[str, Any],
    ) -> tuple[dict[str, Any], Path, Path]:
        split_samples = self._build_recognition_split_samples(data_root)
        split_manifests = self._build_recognition_split_manifests(data_root, split_samples)
        quality = self._build_recognition_quality(data_root)
        stats = self._build_recognition_stats(split_manifests)
        image_statistics = self._build_recognition_image_statistics(split_samples)
        archive_file = self._build_archive(data_root, archive_path)

        manifest = {
            "schema_version": "1.0",
            "dataset_name": dataset_name or "handwriting_recognizer",
            "dataset_version": dataset_version,
            "task": "recognition",
            "created_at": _utcnow_iso(),
            "storage": self._build_storage(data_root.name, dataset_version, storage_config),
            "archive": self._build_archive_metadata(archive_file),
            "data_schema": {
                "input_type": "image",
                "label_format": "label_json",
                "label_type": "text",
            },
            "expected_structure": {
                "root_dir": data_root.name,
                "splits": split_manifests,
            },
            "stats": stats,
            "image_statistics": image_statistics,
            "quality": quality,
            "generation_policy": {
                "split_method": str(generation_policy.get("split_method", "random")),
                "random_seed": int(generation_policy.get("random_seed", 42)),
            },
        }

        manifest_path = archive_file.parent / "recognition_dataset_manifest.json"
        self._write_manifest_json(manifest_path, manifest)
        return manifest, archive_file, manifest_path

    @staticmethod
    def _resolve_data_root(data_path: str) -> Path:
        data_root = Path(data_path).expanduser().resolve()
        if not data_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
        if not data_root.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {data_root}")
        return data_root

    def _load_classes(self, data_yaml_path: Path) -> list[dict[str, Any]]:
        if not data_yaml_path.exists():
            return [{"id": 0, "name": DEFAULT_CLASS_NAME}]

        with data_yaml_path.open("r", encoding="utf-8") as file:
            raw_yaml = yaml.safe_load(file) or {}

        names = raw_yaml.get("names")
        if isinstance(names, list):
            return [{"id": index, "name": str(name)} for index, name in enumerate(names)]
        if isinstance(names, dict):
            return [
                {"id": int(index), "name": str(name)}
                for index, name in sorted(names.items(), key=lambda item: int(item[0]))
            ]
        return [{"id": 0, "name": DEFAULT_CLASS_NAME}]

    def _build_split_manifests(self, data_root: Path) -> dict[str, dict[str, Any]]:
        manifests: dict[str, dict[str, Any]] = {}
        for split_name in ("train", "val", "test"):
            images_dir = data_root / split_name / "images"
            labels_dir = data_root / split_name / "labels"
            manifests[split_name] = {
                "enabled": images_dir.exists() or labels_dir.exists(),
                "images_path": f"{data_root.name}/{split_name}/images" if images_dir.exists() else None,
                "labels_path": f"{data_root.name}/{split_name}/labels" if labels_dir.exists() else None,
                "images_samples": self._count_image_files(images_dir),
                "labels_samples": self._count_label_files(labels_dir),
            }
        return manifests

    def _build_storage(
        self,
        root_dir: str,
        dataset_version: str,
        storage_config: dict[str, str],
    ) -> dict[str, str]:
        provider = storage_config.get("provider", "minio")
        bucket = storage_config.get("bucket", "data")
        prefix = storage_config.get("prefix", f"ml/training/data/{root_dir}/{dataset_version}")
        archive_name = f"{root_dir}.zip"
        return {
            "provider": provider,
            "bucket": bucket,
            "prefix": prefix,
            "archive_uri": storage_config.get(
                "archive_uri",
                f"minio://{bucket}/{prefix}/{archive_name}",
            ),
        }

    def _build_archive(self, data_root: Path, archive_path: str | None) -> Path:
        destination = (
            Path(archive_path).expanduser().resolve()
            if archive_path
            else (data_root.parent / f"{data_root.name}.zip").resolve()
        )
        return zip_directory(data_root, destination)

    def _build_archive_metadata(self, archive_path: Path) -> dict[str, Any]:
        return {
            "format": "zip",
            "compression": "deflate",
            "file_path": str(archive_path),
            "size_bytes": archive_path.stat().st_size,
            "checksum_sha256": self._compute_sha256(archive_path),
        }

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    @staticmethod
    def _build_stats(split_manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
        total_images = sum(split["images_samples"] for split in split_manifests.values())
        total_labels = sum(split["labels_samples"] for split in split_manifests.values())
        denominator = total_images or 1
        return {
            "total_images": total_images,
            "total_labels": total_labels,
            "train_ratio": round(split_manifests["train"]["images_samples"] / denominator, 4),
            "val_ratio": round(split_manifests["val"]["images_samples"] / denominator, 4),
            "test_ratio": round(split_manifests["test"]["images_samples"] / denominator, 4),
        }

    def _build_image_statistics(
        self,
        data_root: Path,
        split_manifests: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "train": self._compute_split_image_statistics(data_root / "train" / "images", split_manifests["train"]),
            "val": self._compute_split_image_statistics(data_root / "val" / "images", split_manifests["val"]),
        }

    def _build_recognition_split_samples(
        self,
        data_root: Path,
    ) -> dict[str, list[RecognitionSample]]:
        return {
            split_name: self.collect_recognition_valid_samples(data_root / split_dir, split_name=split_name)
            for split_name, split_dir in (("train", "train_data"), ("val", "val_data"), ("test", "test_data"))
        }

    def _build_recognition_split_manifests(
        self,
        data_root: Path,
        split_samples: dict[str, list[RecognitionSample]],
    ) -> dict[str, dict[str, Any]]:
        manifests: dict[str, dict[str, Any]] = {}
        for split_name, split_dir in (("train", "train_data"), ("val", "val_data"), ("test", "test_data")):
            split_root = data_root / split_dir
            manifests[split_name] = {
                "enabled": split_root.exists(),
                "data_path": f"{data_root.name}/{split_dir}" if split_root.exists() else None,
                "groups": self._count_recognition_groups(split_root),
                "samples": len(split_samples[split_name]),
            }
        return manifests

    @staticmethod
    def _count_recognition_groups(split_root: Path) -> int:
        if not split_root.exists():
            return 0
        return sum(1 for path in split_root.iterdir() if path.is_dir())

    @staticmethod
    def _build_recognition_stats(split_manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
        total_samples = sum(split["samples"] for split in split_manifests.values())
        denominator = total_samples or 1
        return {
            "total_samples": total_samples,
            "train_ratio": round(split_manifests["train"]["samples"] / denominator, 4),
            "val_ratio": round(split_manifests["val"]["samples"] / denominator, 4),
            "test_ratio": round(split_manifests["test"]["samples"] / denominator, 4),
        }

    def _build_recognition_image_statistics(
        self,
        split_samples: dict[str, list[RecognitionSample]],
    ) -> dict[str, Any]:
        return {
            "train": self._compute_sample_image_statistics(split_samples["train"]),
            "val": self._compute_sample_image_statistics(split_samples["val"]),
        }

    def _compute_sample_image_statistics(
        self,
        samples: list[RecognitionSample],
    ) -> dict[str, dict[str, float]]:
        channel_sums = [0.0, 0.0, 0.0]
        channel_square_sums = [0.0, 0.0, 0.0]
        total_pixels = 0

        for sample in samples:
            rgb_image = self._load_image_rgb(sample.image_path)
            if rgb_image is None:
                continue
            pixels = rgb_image.reshape(-1, 3)
            total_pixels += int(pixels.shape[0])
            sums = pixels.sum(axis=0)
            square_sums = (pixels * pixels).sum(axis=0)
            for index in range(3):
                channel_sums[index] += float(sums[index])
                channel_square_sums[index] += float(square_sums[index])

        if total_pixels == 0:
            return self._empty_image_statistics()

        means = [round(value / total_pixels, 6) for value in channel_sums]
        stds = []
        for index, mean_value in enumerate(means):
            variance = max((channel_square_sums[index] / total_pixels) - (mean_value * mean_value), 0.0)
            stds.append(round(sqrt(variance), 6))

        return {
            "mean_rgb": self._rgb_map(means),
            "std_rgb": self._rgb_map(stds),
        }

    def _compute_split_image_statistics(
        self,
        images_dir: Path,
        split_manifest: dict[str, Any],
    ) -> dict[str, dict[str, float]]:
        if not split_manifest["enabled"] or split_manifest["images_path"] is None:
            return self._empty_image_statistics()

        channel_sums = [0.0, 0.0, 0.0]
        channel_square_sums = [0.0, 0.0, 0.0]
        total_pixels = 0

        for image_path in self._iter_image_files(images_dir):
            rgb_image = self._load_image_rgb(image_path)
            if rgb_image is None:
                continue
            pixels = rgb_image.reshape(-1, 3)
            total_pixels += int(pixels.shape[0])
            sums = pixels.sum(axis=0)
            square_sums = (pixels * pixels).sum(axis=0)
            for index in range(3):
                channel_sums[index] += float(sums[index])
                channel_square_sums[index] += float(square_sums[index])

        if total_pixels == 0:
            return self._empty_image_statistics()

        means = [round(value / total_pixels, 6) for value in channel_sums]
        stds = []
        for index, mean_value in enumerate(means):
            variance = max((channel_square_sums[index] / total_pixels) - (mean_value * mean_value), 0.0)
            stds.append(round(sqrt(variance), 6))

        return {
            "mean_rgb": self._rgb_map(means),
            "std_rgb": self._rgb_map(stds),
        }

    def _build_quality(
        self,
        data_root: Path,
        split_manifests: dict[str, dict[str, Any]],
        classes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        class_ids = {item["id"] for item in classes}
        missing_images = 0
        missing_labels = 0
        invalid_labels = 0
        corrupted_images = 0

        for split_name, split_manifest in split_manifests.items():
            if not split_manifest["enabled"]:
                continue

            images_dir = data_root / split_name / "images"
            labels_dir = data_root / split_name / "labels"
            image_stems = {path.stem: path for path in self._iter_image_files(images_dir)}
            label_files = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
            label_stems = {path.stem: path for path in label_files}

            missing_labels += len(set(image_stems) - set(label_stems))
            missing_images += len(set(label_stems) - set(image_stems))

            for image_path in image_stems.values():
                if self._is_corrupted_image(image_path):
                    corrupted_images += 1

            for label_path in label_files:
                invalid_labels += self._count_invalid_yolo_rows(label_path, class_ids)

        return {
            "validation_status": "failed"
            if any((missing_images, missing_labels, invalid_labels, corrupted_images))
            else "passed",
            "missing_images": missing_images,
            "missing_labels": missing_labels,
            "invalid_labels": invalid_labels,
            "corrupted_images": corrupted_images,
        }

    def _build_recognition_quality(self, data_root: Path) -> dict[str, Any]:
        missing_images = 0
        missing_labels = 0
        invalid_labels = 0
        corrupted_images = 0

        for split_dir in ("train_data", "val_data", "test_data"):
            split_root = data_root / split_dir
            if not split_root.exists():
                continue
            for sample_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
                label_path = sample_dir / "label.json"
                image_files = {
                    path.name: path
                    for path in self._iter_image_files(sample_dir)
                }
                if not label_path.exists():
                    missing_labels += len(image_files)
                    continue

                try:
                    labels = json.loads(label_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    invalid_labels += 1
                    continue

                if not isinstance(labels, dict):
                    invalid_labels += 1
                    continue

                label_names = {str(name) for name in labels.keys()}
                missing_images += len([name for name in label_names if name not in image_files])
                missing_labels += len(set(image_files) - label_names)

                for image_name in label_names & set(image_files):
                    if self._is_corrupted_image(image_files[image_name]):
                        corrupted_images += 1

        return {
            "validation_status": "failed"
            if any((missing_images, missing_labels, invalid_labels, corrupted_images))
            else "passed",
            "missing_images": missing_images,
            "missing_labels": missing_labels,
            "invalid_labels": invalid_labels,
            "corrupted_images": corrupted_images,
        }

    @staticmethod
    def collect_recognition_valid_samples(
        split_root: Path,
        *,
        split_name: str,
    ) -> list[RecognitionSample]:
        if not split_root.exists():
            return []

        samples: list[RecognitionSample] = []
        for sample_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            label_path = sample_dir / "label.json"
            if not label_path.exists():
                continue

            try:
                labels = json.loads(label_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            if not isinstance(labels, dict):
                continue

            for image_name in sorted(str(name) for name in labels.keys()):
                image_path = sample_dir / image_name
                if not image_path.is_file():
                    continue
                samples.append(
                    RecognitionSample(
                        split_name=split_name,
                        group_name=sample_dir.name,
                        image_name=image_name,
                        image_path=image_path,
                        label=str(labels[image_name]),
                    )
                )
        return samples

    @staticmethod
    def _count_invalid_yolo_rows(label_file: Path, class_ids: set[int]) -> int:
        invalid_rows = 0
        with label_file.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    invalid_rows += 1
                    continue
                try:
                    class_id = int(parts[0])
                    coordinates = [float(value) for value in parts[1:]]
                except ValueError:
                    invalid_rows += 1
                    continue
                if class_id not in class_ids:
                    invalid_rows += 1
                    continue
                if any(value < 0.0 or value > 1.0 for value in coordinates):
                    invalid_rows += 1
        return invalid_rows

    @staticmethod
    def _count_image_files(directory: Path) -> int:
        return sum(1 for _ in CreateDataManifest._iter_image_files(directory))

    @staticmethod
    def _iter_image_files(directory: Path):
        if not directory.exists():
            return iter(())
        return (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )

    @staticmethod
    def _count_label_files(directory: Path) -> int:
        if not directory.exists():
            return 0
        return sum(1 for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".txt")

    @staticmethod
    def _empty_image_statistics() -> dict[str, dict[str, float]]:
        return {
            "mean_rgb": CreateDataManifest._rgb_map([0.0, 0.0, 0.0]),
            "std_rgb": CreateDataManifest._rgb_map([0.0, 0.0, 0.0]),
        }

    @staticmethod
    def _rgb_map(values: list[float]) -> dict[str, float]:
        return {"red": values[0], "green": values[1], "blue": values[2]}

    @staticmethod
    def _load_image_rgb(image_path: Path):
        try:
            import cv2  # type: ignore
        except ImportError:
            cv2 = None

        if cv2 is not None:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float64") / 255.0

        if image_path.suffix.lower() == ".bmp":
            return CreateDataManifest._load_bmp_image_rgb(image_path)
        return None

    @staticmethod
    def _load_bmp_image_rgb(image_path: Path):
        import numpy as np

        data = image_path.read_bytes()
        if len(data) < 54 or data[:2] != b"BM":
            return None

        pixel_offset = int.from_bytes(data[10:14], "little")
        width = int.from_bytes(data[18:22], "little")
        height = int.from_bytes(data[22:26], "little")
        bits_per_pixel = int.from_bytes(data[28:30], "little")
        compression = int.from_bytes(data[30:34], "little")
        if width <= 0 or height == 0 or bits_per_pixel != 24 or compression != 0:
            return None

        row_stride = ((width * 3 + 3) // 4) * 4
        pixel_bytes = data[pixel_offset:]
        expected_size = row_stride * abs(height)
        if len(pixel_bytes) < expected_size:
            return None

        rows = []
        for row_index in range(abs(height)):
            start = row_index * row_stride
            end = start + (width * 3)
            row = np.frombuffer(pixel_bytes[start:end], dtype=np.uint8).reshape(width, 3)
            rows.append(row[:, ::-1])

        if height > 0:
            rows.reverse()
        return np.stack(rows).astype("float64") / 255.0

    @staticmethod
    def _is_corrupted_image(image_path: Path) -> bool:
        return CreateDataManifest._load_image_rgb(image_path) is None

    @staticmethod
    def _write_manifest_json(manifest_path: Path, manifest: dict[str, Any]) -> None:
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate recognition dataset manifest")
    parser.add_argument("--data_path", nargs="?", default=DEFAULT_DATA_PATH)
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--dataset-name", default="handwriting_detection")
    parser.add_argument("--archive-path", default=None)
    parser.add_argument("--storage-provider", default="minio")
    parser.add_argument("--storage-bucket", default="data")
    parser.add_argument("--storage-prefix", default="ml/training/data/recognition/v1")
    parser.add_argument("--archive-uri", default=None)
    args = parser.parse_args()

    manifest = CreateDataManifest(data_types="recognition", logger=Logger(__name__)).execute(
        data_path=args.data_path,
        dataset_version=args.dataset_version,
        dataset_name=args.dataset_name,
        archive_path=args.archive_path,
        storage_config={
            "provider": args.storage_provider,
            "bucket": args.storage_bucket,
            "prefix": args.storage_prefix,
            **({"archive_uri": args.archive_uri} if args.archive_uri else {}),
        },
    )
    print(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
