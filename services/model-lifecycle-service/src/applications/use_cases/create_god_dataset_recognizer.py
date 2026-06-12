from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.applications.use_cases.create_data_manifest import CreateDataManifest, RecognitionSample
from src.utils.logger import Logger


RECOGNIZER_CHARACTER_SET = (
    "0123456789aàáảãạăằắẳẵặâầấẩẫậbcdđeèéẻẽẹêềếểễệghiìíỉĩịklmno"
    "òóỏõọôồốổỗộơờớởỡợpqrstuùúủũụưừứửữựvxyAÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬBCDĐ"
    "EÈÉẺẼẸÊỀẾỂỄỆGHIÌÍỈĨỊKLMNOÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢPQRSTUÙÚỦŨỤƯỪỨỬỮỰVXY"
)


@dataclass(frozen=True)
class RecognizerSampleFeature:
    sample: RecognitionSample
    pixel_count: int
    sum_rgb: tuple[float, float, float]
    sum_sq_rgb: tuple[float, float, float]
    character_counts: tuple[int, ...]


class CreateGodDatasetRecognizer:
    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        self._manifest_creator = CreateDataManifest(data_types="recognition", logger=logger)
        self._character_to_index = {
            character: index for index, character in enumerate(RECOGNIZER_CHARACTER_SET)
        }

    def execute(
        self,
        *,
        source_data_path: str,
        subset_percent: float,
        output_root: str | None = None,
        dataset_version: str = "v1",
        split_seed: int = 42,
    ) -> dict[str, Any]:
        if subset_percent <= 0 or subset_percent > 100:
            raise ValueError("subset_percent must be within (0, 100]")

        source_root = Path(source_data_path).expanduser().resolve()
        source_train_root = source_root / "train_data"
        if not source_train_root.exists():
            raise FileNotFoundError("Source recognizer train_data is required")

        valid_samples = CreateDataManifest.collect_recognition_valid_samples(
            source_train_root,
            split_name="train",
        )
        if not valid_samples:
            raise ValueError("No valid recognizer train samples available for subset generation")
        self._logger.info(
            "[GOD_RECOGNIZER_VALID_SAMPLES]",
            f"count={len(valid_samples)}",
        )

        split_train_samples, _ = self._split_train_val(valid_samples, split_seed)
        if not split_train_samples:
            raise ValueError("No train samples available after recognizer split")
        self._logger.info(
            "[GOD_RECOGNIZER_SPLIT_TRAIN_SAMPLES]",
            f"count={len(split_train_samples)}",
            f"seed={split_seed}",
        )

        candidates = self._build_features(split_train_samples)
        self._logger.info(
            "[GOD_RECOGNIZER_FEATURES_BUILT]",
            f"count={len(candidates)}",
        )
        selected = self._select_subset(candidates, subset_percent, split_seed=split_seed)
        self._logger.info(
            "[GOD_RECOGNIZER_SUBSET_SELECTED]",
            f"count={len(selected)}",
            f"subset_percent={subset_percent}",
        )

        output_parent = Path(output_root).expanduser().resolve() if output_root else source_root.parent
        subset_root = output_parent / "god_dataset_recognizer"
        archive_path = output_parent / "god_dataset_recognizer.zip"
        manifest_path = output_parent / "god_dataset_recognizer_dataset_manifest.json"

        self._reset_output_root(subset_root, archive_path, manifest_path)
        self._write_subset_dataset(subset_root=subset_root, selected=selected)
        self._manifest_creator.execute(
            str(subset_root),
            dataset_version=dataset_version,
            dataset_name="god_dataset_recognizer",
            archive_path=str(archive_path),
        )

        generated_manifest_path = archive_path.parent / "recognition_dataset_manifest.json"
        if generated_manifest_path.exists():
            generated_manifest_path.replace(manifest_path)

        return {
            "dataset_root": str(subset_root.resolve()),
            "archive_path": str(archive_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "selected_samples": len(selected),
        }

    @staticmethod
    def _split_train_val(
        samples: list[RecognitionSample],
        split_seed: int,
    ) -> tuple[list[RecognitionSample], list[RecognitionSample]]:
        shuffled = samples[:]
        random.Random(split_seed).shuffle(shuffled)
        train_count = max(1, int(len(shuffled) * 0.8))
        train_count = min(train_count, len(shuffled))
        return shuffled[:train_count], shuffled[train_count:]

    def _build_features(
        self,
        samples: list[RecognitionSample],
    ) -> list[RecognizerSampleFeature]:
        features: list[RecognizerSampleFeature] = []
        for sample in samples:
            rgb_image = CreateDataManifest._load_image_rgb(sample.image_path)
            if rgb_image is None:
                continue

            pixels = rgb_image.reshape(-1, 3)
            sums = pixels.sum(axis=0)
            square_sums = (pixels * pixels).sum(axis=0)
            features.append(
                RecognizerSampleFeature(
                    sample=sample,
                    pixel_count=int(pixels.shape[0]),
                    sum_rgb=(float(sums[0]), float(sums[1]), float(sums[2])),
                    sum_sq_rgb=(
                        float(square_sums[0]),
                        float(square_sums[1]),
                        float(square_sums[2]),
                    ),
                    character_counts=self._count_characters(sample.label),
                )
            )
        return features

    def _select_subset(
        self,
        candidates: list[RecognizerSampleFeature],
        subset_percent: float,
        *,
        split_seed: int,
    ) -> list[RecognizerSampleFeature]:
        target_count = round(len(candidates) * subset_percent / 100)
        if target_count < 1:
            raise ValueError("subset_percent produces zero selected samples")

        target_mean, target_std = self._compute_feature_stats(candidates)
        buckets: dict[str, list[RecognizerSampleFeature]] = defaultdict(list)
        for feature in candidates:
            buckets[feature.sample.label].append(feature)

        quotas, remainder = self._allocate_label_quotas(
            buckets=buckets,
            target_count=target_count,
        )
        selected: list[RecognizerSampleFeature] = []
        rng = random.Random(split_seed)

        for label, features in buckets.items():
            quota = quotas.get(label, 0)
            if quota <= 0:
                continue
            shuffled = features[:]
            rng.shuffle(shuffled)
            ranked = sorted(
                shuffled,
                key=lambda feature: self._sample_image_error(
                    feature=feature,
                    target_mean=target_mean,
                    target_std=target_std,
                ),
            )
            selected.extend(ranked[:quota])

        if len(selected) > target_count:
            selected = selected[:target_count]
        elif len(selected) < target_count:
            already_selected = {id(feature) for feature in selected}
            remaining = [feature for feature in candidates if id(feature) not in already_selected]
            rng.shuffle(remaining)
            ranked_remaining = sorted(
                remaining,
                key=lambda feature: (
                    -remainder.get(feature.sample.label, 0.0),
                    self._sample_image_error(
                        feature=feature,
                        target_mean=target_mean,
                        target_std=target_std,
                    ),
                ),
            )
            selected.extend(ranked_remaining[: target_count - len(selected)])

        return selected

    @staticmethod
    def _allocate_label_quotas(
        *,
        buckets: dict[str, list[RecognizerSampleFeature]],
        target_count: int,
    ) -> tuple[dict[str, int], dict[str, float]]:
        total_count = sum(len(features) for features in buckets.values())
        if total_count == 0:
            return {}, {}

        quotas: dict[str, int] = {}
        remainders: dict[str, float] = {}
        assigned = 0
        for label, features in buckets.items():
            exact_quota = len(features) * target_count / total_count
            base_quota = int(exact_quota)
            quotas[label] = min(base_quota, len(features))
            remainders[label] = exact_quota - base_quota
            assigned += quotas[label]

        remaining_slots = target_count - assigned
        ranked_labels = sorted(
            buckets.keys(),
            key=lambda label: (-remainders[label], -len(buckets[label]), label),
        )
        for label in ranked_labels:
            if remaining_slots <= 0:
                break
            if quotas[label] >= len(buckets[label]):
                continue
            quotas[label] += 1
            remaining_slots -= 1

        return quotas, remainders

    @staticmethod
    def _sample_image_error(
        *,
        feature: RecognizerSampleFeature,
        target_mean: list[float],
        target_std: list[float],
    ) -> float:
        sample_means = [feature.sum_rgb[index] / feature.pixel_count for index in range(3)]
        sample_stds = []
        for index, mean_value in enumerate(sample_means):
            variance = max((feature.sum_sq_rgb[index] / feature.pixel_count) - (mean_value * mean_value), 0.0)
            sample_stds.append(variance ** 0.5)

        mean_error = sum(abs(sample_means[index] - target_mean[index]) for index in range(3))
        std_error = sum(abs(sample_stds[index] - target_std[index]) for index in range(3))
        return mean_error + std_error

    @staticmethod
    def _compute_feature_stats(
        candidates: list[RecognizerSampleFeature],
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

    def _count_characters(self, label: str) -> tuple[int, ...]:
        counts = [0] * len(self._character_to_index)
        for character in label:
            index = self._character_to_index.get(character)
            if index is not None:
                counts[index] += 1
        return tuple(counts)

    @staticmethod
    def _compute_character_distribution(character_counts_list: list[tuple[int, ...]]) -> list[float]:
        if not character_counts_list:
            return []

        totals = [0] * len(character_counts_list[0])
        for counts in character_counts_list:
            for index, count in enumerate(counts):
                totals[index] += count

        total_characters = sum(totals)
        if total_characters == 0:
            return [0.0] * len(totals)
        return [count / total_characters for count in totals]

    @staticmethod
    def _write_subset_dataset(
        *,
        subset_root: Path,
        selected: list[RecognizerSampleFeature],
    ) -> None:
        grouped_labels: dict[str, dict[str, str]] = defaultdict(dict)
        for feature in selected:
            destination_dir = subset_root / "train_data" / feature.sample.group_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                feature.sample.image_path,
                destination_dir / feature.sample.image_name,
            )
            grouped_labels[feature.sample.group_name][feature.sample.image_name] = feature.sample.label

        for group_name, labels in grouped_labels.items():
            label_path = subset_root / "train_data" / group_name / "label.json"
            label_path.write_text(
                json.dumps(dict(sorted(labels.items())), ensure_ascii=False),
                encoding="utf-8",
            )

    @staticmethod
    def _reset_output_root(subset_root: Path, archive_path: Path, manifest_path: Path) -> None:
        if subset_root.exists():
            shutil.rmtree(subset_root)
        if archive_path.exists():
            archive_path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()


if __name__ == "__main__":
    logger = Logger()
    creator = CreateGodDatasetRecognizer(logger=logger)
    result = creator.execute(
        source_data_path="/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus/ml/training/data/recognizer",
        subset_percent=10.0,
        output_root="/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus/ml/training/data",
        dataset_version="v1",
    )
    print(result)
