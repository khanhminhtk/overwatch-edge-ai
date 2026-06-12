from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.applications.use_cases.create_data_manifest import CreateDataManifest
from src.utils.logger import Logger


@dataclass(frozen=True)
class DetectionSampleFeature:
    image_path: Path
    label_path: Path
    pixel_count: int
    sum_rgb: tuple[float, float, float]
    sum_sq_rgb: tuple[float, float, float]


class CreateGodDatasetDetection:
    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        self._manifest_creator = CreateDataManifest(data_types="detection", logger=logger)

    def execute(
        self,
        *,
        source_data_path: str,
        subset_percent: float,
        output_root: str | None = None,
        dataset_version: str = "v1",
    ) -> dict[str, Any]:
        if subset_percent <= 0 or subset_percent > 100:
            raise ValueError("subset_percent must be within (0, 100]")

        source_root = Path(source_data_path).expanduser().resolve()
        train_images_dir = source_root / "train" / "images"
        train_labels_dir = source_root / "train" / "labels"
        if not train_images_dir.exists() or not train_labels_dir.exists():
            raise FileNotFoundError("Source detection train/images and train/labels are required")

        output_parent = Path(output_root).expanduser().resolve() if output_root else source_root.parent
        subset_root = output_parent / "god_dataset_detection"
        archive_path = output_parent / "god_dataset_detection.zip"
        manifest_path = output_parent / "god_dataset_detection_dataset_manifest.json"

        candidates = self._collect_train_candidates(train_images_dir, train_labels_dir)
        if not candidates:
            raise ValueError("No valid train samples available for subset generation")

        selected = self._select_subset(candidates, subset_percent)
        self._reset_output_root(subset_root, archive_path, manifest_path)
        self._write_subset_dataset(
            subset_root=subset_root,
            source_root=source_root,
            selected=selected,
        )

        self._manifest_creator.execute(
            str(subset_root),
            dataset_version=dataset_version,
            dataset_name="god_dataset_detection",
            archive_path=str(archive_path),
        )

        generated_manifest_path = archive_path.parent / "detection_dataset_manifest.json"
        if generated_manifest_path.exists():
            generated_manifest_path.replace(manifest_path)

        return {
            "dataset_root": str(subset_root.resolve()),
            "archive_path": str(archive_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "selected_samples": len(selected),
        }

    def _collect_train_candidates(
        self,
        train_images_dir: Path,
        train_labels_dir: Path,
    ) -> list[DetectionSampleFeature]:
        candidates: list[DetectionSampleFeature] = []
        for image_path in sorted(self._iter_image_files(train_images_dir), key=lambda path: path.name):
            label_path = train_labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            rgb_image = CreateDataManifest._load_image_rgb(image_path)
            if rgb_image is None:
                continue

            pixels = rgb_image.reshape(-1, 3)
            sums = pixels.sum(axis=0)
            square_sums = (pixels * pixels).sum(axis=0)
            candidates.append(
                DetectionSampleFeature(
                    image_path=image_path,
                    label_path=label_path,
                    pixel_count=int(pixels.shape[0]),
                    sum_rgb=(float(sums[0]), float(sums[1]), float(sums[2])),
                    sum_sq_rgb=(
                        float(square_sums[0]),
                        float(square_sums[1]),
                        float(square_sums[2]),
                    ),
                )
            )
        return candidates

    def _select_subset(
        self,
        candidates: list[DetectionSampleFeature],
        subset_percent: float,
    ) -> list[DetectionSampleFeature]:
        target_count = round(len(candidates) * subset_percent / 100)
        if target_count < 1:
            raise ValueError("subset_percent produces zero selected samples")

        target_mean, target_std = self._compute_feature_stats(candidates)
        selected: list[DetectionSampleFeature] = []
        remaining = candidates[:]
        running_sum = [0.0, 0.0, 0.0]
        running_sq_sum = [0.0, 0.0, 0.0]
        running_pixels = 0

        while len(selected) < target_count:
            best_feature: DetectionSampleFeature | None = None
            best_score: float | None = None
            for feature in remaining:
                score = self._score_candidate(
                    feature=feature,
                    current_sum=running_sum,
                    current_sq_sum=running_sq_sum,
                    current_pixels=running_pixels,
                    target_mean=target_mean,
                    target_std=target_std,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_feature = feature

            if best_feature is None:
                raise ValueError("Unable to select subset samples")

            selected.append(best_feature)
            remaining.remove(best_feature)
            running_pixels += best_feature.pixel_count
            for index in range(3):
                running_sum[index] += best_feature.sum_rgb[index]
                running_sq_sum[index] += best_feature.sum_sq_rgb[index]

        return selected

    @staticmethod
    def _compute_feature_stats(
        candidates: list[DetectionSampleFeature],
    ) -> tuple[list[float], list[float]]:
        total_pixels = sum(feature.pixel_count for feature in candidates)
        if total_pixels == 0:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

        sums = [0.0, 0.0, 0.0]
        square_sums = [0.0, 0.0, 0.0]
        for feature in candidates:
            for index in range(3):
                sums[index] += feature.sum_rgb[index]
                square_sums[index] += feature.sum_sq_rgb[index]

        means = [value / total_pixels for value in sums]
        stds = []
        for index, mean_value in enumerate(means):
            variance = max((square_sums[index] / total_pixels) - (mean_value * mean_value), 0.0)
            stds.append(variance ** 0.5)
        return means, stds

    @staticmethod
    def _score_candidate(
        *,
        feature: DetectionSampleFeature,
        current_sum: list[float],
        current_sq_sum: list[float],
        current_pixels: int,
        target_mean: list[float],
        target_std: list[float],
    ) -> float:
        next_pixels = current_pixels + feature.pixel_count
        next_sums = [current_sum[index] + feature.sum_rgb[index] for index in range(3)]
        next_sq_sums = [current_sq_sum[index] + feature.sum_sq_rgb[index] for index in range(3)]
        next_means = [value / next_pixels for value in next_sums]
        next_stds = []
        for index, mean_value in enumerate(next_means):
            variance = max((next_sq_sums[index] / next_pixels) - (mean_value * mean_value), 0.0)
            next_stds.append(variance ** 0.5)

        mean_error = sum(abs(next_means[index] - target_mean[index]) for index in range(3))
        std_error = sum(abs(next_stds[index] - target_std[index]) for index in range(3))
        return mean_error + std_error

    @staticmethod
    def _write_subset_dataset(
        *,
        subset_root: Path,
        source_root: Path,
        selected: list[DetectionSampleFeature],
    ) -> None:
        images_dir = subset_root / "train" / "images"
        labels_dir = subset_root / "train" / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        data_yaml_path = source_root / "data.yaml"
        if data_yaml_path.exists():
            shutil.copy2(data_yaml_path, subset_root / "data.yaml")

        for feature in selected:
            shutil.copy2(feature.image_path, images_dir / feature.image_path.name)
            shutil.copy2(feature.label_path, labels_dir / feature.label_path.name)

    @staticmethod
    def _reset_output_root(subset_root: Path, archive_path: Path, manifest_path: Path) -> None:
        if subset_root.exists():
            shutil.rmtree(subset_root)
        if archive_path.exists():
            archive_path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    @staticmethod
    def _iter_image_files(directory: Path):
        return CreateDataManifest._iter_image_files(directory)


if __name__ == "__main__":
    logger = Logger()
    creator = CreateGodDatasetDetection(logger=logger)
    result = creator.execute(
        source_data_path="/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus/ml/training/data/detection",
        subset_percent=10.0,
        output_root="/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus/ml/training/data",
        dataset_version="v1",
    )
    print(result)