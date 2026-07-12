from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.modules.dataset.application.image_utils import iter_image_files, load_image_rgb
from src.modules.dataset.application.manifest_builder import ManifestBuilder
from src.modules.dataset.domain.value_objects.feature_vectors import (
    DetectionSampleFeature,
    GodDatasetResult,
)
from src.platform.logger import Logger


class CreateGodDatasetDetection:
    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        self._manifest_builder = ManifestBuilder(data_types="detection", logger=logger)

    def execute(
        self,
        *,
        source_data_path: str,
        subset_percent: float,
        output_root: str | None = None,
        dataset_version: str = "v1",
    ) -> GodDatasetResult:
        if subset_percent <= 0 or subset_percent > 100:
            raise ValueError("subset_percent must be within (0, 100]")

        source_root = Path(source_data_path).expanduser().resolve()
        train_images = source_root / "train" / "images"
        train_labels = source_root / "train" / "labels"
        if not train_images.exists() or not train_labels.exists():
            raise FileNotFoundError("Source detection train/images and train/labels are required")

        output_parent = Path(output_root).expanduser().resolve() if output_root else source_root.parent
        subset_root = output_parent / "god_dataset_detection"
        archive_path = output_parent / "god_dataset_detection.zip"
        manifest_path = output_parent / "god_dataset_detection_dataset_manifest.json"

        candidates = self._collect_candidates(train_images, train_labels)
        if not candidates:
            raise ValueError("No valid train samples available")

        selected = self._select_subset(candidates, subset_percent)
        self._reset_output(subset_root, archive_path, manifest_path)
        self._write_subset(subset_root, source_root, selected)
        self._manifest_builder.build(
            subset_root,
            dataset_version=dataset_version,
            dataset_name="god_dataset_detection",
            archive_path=archive_path,
        )

        generated_manifest = archive_path.parent / "detection_dataset_manifest.json"
        if generated_manifest.exists():
            generated_manifest.replace(manifest_path)

        return GodDatasetResult(
            dataset_root=str(subset_root.resolve()),
            archive_path=str(archive_path.resolve()),
            manifest_path=str(manifest_path.resolve()),
            selected_samples=len(selected),
        )

    def _collect_candidates(
        self, images_dir: Path, labels_dir: Path
    ) -> list[DetectionSampleFeature]:
        image_files = sorted(iter_image_files(images_dir), key=lambda p: p.name)
        candidates: list[DetectionSampleFeature] = []
        for img_path in tqdm(image_files, desc="Collect candidates"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            rgb = load_image_rgb(img_path)
            if rgb is None:
                continue
            pixels = rgb.reshape(-1, 3)
            s = pixels.sum(axis=0)
            sq = (pixels * pixels).sum(axis=0)
            candidates.append(
                DetectionSampleFeature(
                    image_path=img_path,
                    label_path=label_path,
                    pixel_count=int(pixels.shape[0]),
                    sum_rgb=(float(s[0]), float(s[1]), float(s[2])),
                    sum_sq_rgb=(float(sq[0]), float(sq[1]), float(sq[2])),
                )
            )
        return candidates

    def _select_subset(
        self, candidates: list[DetectionSampleFeature], subset_percent: float
    ) -> list[DetectionSampleFeature]:
        target_count = round(len(candidates) * subset_percent / 100)
        if target_count < 1:
            raise ValueError("subset_percent produces zero samples")

        target_mean, target_std = self._compute_stats(candidates)
        selected: list[DetectionSampleFeature] = []
        remaining = candidates[:]
        running_sum = [0.0, 0.0, 0.0]
        running_sq = [0.0, 0.0, 0.0]
        running_px = 0

        with tqdm(total=target_count, desc="Selecting subset") as pbar:
            while len(selected) < target_count:
                best: DetectionSampleFeature | None = None
                best_score: float | None = None
                for feat in remaining:
                    score = self._score(feat, running_sum, running_sq, running_px, target_mean, target_std)
                    if best_score is None or score < best_score:
                        best_score = score
                        best = feat
                if best is None:
                    raise ValueError("Unable to select subset")
                selected.append(best)
                remaining.remove(best)
                running_px += best.pixel_count
                for i in range(3):
                    running_sum[i] += best.sum_rgb[i]
                    running_sq[i] += best.sum_sq_rgb[i]
                pbar.update(1)
        return selected

    @staticmethod
    def _compute_stats(candidates: list[DetectionSampleFeature]) -> tuple[list[float], list[float]]:
        total_px = sum(f.pixel_count for f in candidates) or 1
        sums = [0.0, 0.0, 0.0]
        sq_sums = [0.0, 0.0, 0.0]
        for f in candidates:
            for i in range(3):
                sums[i] += f.sum_rgb[i]
                sq_sums[i] += f.sum_sq_rgb[i]
        means = [s / total_px for s in sums]
        stds = [max((sq_sums[i] / total_px) - means[i] ** 2, 0.0) ** 0.5 for i in range(3)]
        return means, stds

    @staticmethod
    def _score(
        feat: DetectionSampleFeature,
        curr_sum: list[float],
        curr_sq: list[float],
        curr_px: int,
        target_mean: list[float],
        target_std: list[float],
    ) -> float:
        next_px = curr_px + feat.pixel_count
        next_means = [(curr_sum[i] + feat.sum_rgb[i]) / next_px for i in range(3)]
        next_stds = [
            max(((curr_sq[i] + feat.sum_sq_rgb[i]) / next_px) - next_means[i] ** 2, 0.0) ** 0.5
            for i in range(3)
        ]
        return sum(abs(next_means[i] - target_mean[i]) for i in range(3)) + sum(
            abs(next_stds[i] - target_std[i]) for i in range(3)
        )

    @staticmethod
    def _write_subset(
        subset_root: Path, source_root: Path, selected: list[DetectionSampleFeature]
    ) -> None:
        img_dir = subset_root / "train" / "images"
        lbl_dir = subset_root / "train" / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        data_yaml = source_root / "data.yaml"
        if data_yaml.exists():
            shutil.copy2(data_yaml, subset_root / "data.yaml")
        for feat in selected:
            shutil.copy2(feat.image_path, img_dir / feat.image_path.name)
            shutil.copy2(feat.label_path, lbl_dir / feat.label_path.name)

    @staticmethod
    def _reset_output(subset_root: Path, archive_path: Path, manifest_path: Path) -> None:
        for p in (subset_root, archive_path, manifest_path):
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
