from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.modules.dataset.application.image_utils import load_image_rgb
from src.modules.dataset.application.manifest_builder import ManifestBuilder
from src.modules.dataset.domain.value_objects.feature_vectors import (
    GodDatasetResult,
    RecognizerSampleFeature,
    _RecognitionSampleRef,
)
from src.platform.logger import Logger


class CreateGodDatasetRecognizer:
    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        self._manifest_builder = ManifestBuilder(data_types="recognition", logger=logger)

    def execute(
        self,
        *,
        source_data_path: str,
        subset_percent: float,
        output_root: str | None = None,
        dataset_version: str = "v1",
        split_seed: int = 42,
    ) -> GodDatasetResult:
        if subset_percent <= 0 or subset_percent > 100:
            raise ValueError("subset_percent must be within (0, 100]")

        source_root = Path(source_data_path).expanduser().resolve()
        source_train = source_root / "train_data"
        if not source_train.exists():
            raise FileNotFoundError("Source recognizer train_data required")

        valid = self._collect_valid_samples(source_train, split_name="train")
        if not valid:
            raise ValueError("No valid train samples available")

        self._logger.info("[GOD_RECOGNIZER_VALID]", f"count={len(valid)}")

        train_samples, _ = self._train_val_split(valid, split_seed)
        if not train_samples:
            raise ValueError("No train samples after split")

        candidates = self._build_features(train_samples)
        selected = self._select_subset(candidates, subset_percent, split_seed=split_seed)

        output_parent = Path(output_root).expanduser().resolve() if output_root else source_root.parent
        subset_root = output_parent / "god_dataset_recognizer"
        archive_path = output_parent / "god_dataset_recognizer.zip"
        manifest_path = output_parent / "god_dataset_recognizer_dataset_manifest.json"

        self._reset_output(subset_root, archive_path, manifest_path)
        self._write_subset(subset_root, selected)
        self._manifest_builder.build(
            subset_root,
            dataset_version=dataset_version,
            dataset_name="god_dataset_recognizer",
            archive_path=archive_path,
        )

        generated_manifest = archive_path.parent / "recognition_dataset_manifest.json"
        if generated_manifest.exists():
            generated_manifest.replace(manifest_path)

        return GodDatasetResult(
            dataset_root=str(subset_root.resolve()),
            archive_path=str(archive_path.resolve()),
            manifest_path=str(manifest_path.resolve()),
            selected_samples=len(selected),
        )

    @staticmethod
    def _collect_valid_samples(
        split_root: Path, split_name: str
    ) -> list[_RecognitionSampleRef]:
        if not split_root.exists():
            return []
        samples: list[_RecognitionSampleRef] = []
        for group_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            label_path = group_dir / "label.json"
            if not label_path.exists():
                continue
            try:
                labels = json.loads(label_path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(labels, dict):
                continue
            for img_name, label in sorted(labels.items()):
                img_path = group_dir / img_name
                if img_path.is_file():
                    samples.append(
                        _RecognitionSampleRef(
                            group_name=group_dir.name,
                            image_name=img_name,
                            image_path=img_path,
                            label=str(label),
                        )
                    )
        return samples

    @staticmethod
    def _train_val_split(
        samples: list[_RecognitionSampleRef], seed: int
    ) -> tuple[list[_RecognitionSampleRef], list[_RecognitionSampleRef]]:
        shuffled = samples[:]
        random.Random(seed).shuffle(shuffled)
        n = max(1, int(len(shuffled) * 0.8))
        return shuffled[:n], shuffled[n:]

    def _build_features(self, samples: list[_RecognitionSampleRef]) -> list[RecognizerSampleFeature]:
        features: list[RecognizerSampleFeature] = []
        for s in tqdm(samples, desc="Building features"):
            rgb = load_image_rgb(s.image_path)
            if rgb is None:
                continue
            px = rgb.reshape(-1, 3)
            sums = px.sum(axis=0)
            sq = (px * px).sum(axis=0)
            features.append(
                RecognizerSampleFeature(
                    sample=s,
                    pixel_count=int(px.shape[0]),
                    sum_rgb=(float(sums[0]), float(sums[1]), float(sums[2])),
                    sum_sq_rgb=(float(sq[0]), float(sq[1]), float(sq[2])),
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
            raise ValueError("subset_percent produces zero samples")

        target_mean, target_std = self._compute_stats(candidates)
        selected: list[RecognizerSampleFeature] = []
        remaining = candidates[:]
        running_sum = [0.0, 0.0, 0.0]
        running_sq = [0.0, 0.0, 0.0]
        running_px = 0

        with tqdm(total=target_count, desc="Selecting subset") as pbar:
            while len(selected) < target_count:
                best: RecognizerSampleFeature | None = None
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
    def _compute_stats(candidates: list[RecognizerSampleFeature]) -> tuple[list[float], list[float]]:
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
        feat: RecognizerSampleFeature,
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
    def _write_subset(subset_root: Path, selected: list[RecognizerSampleFeature]) -> None:
        grouped: dict[str, dict[str, str]] = defaultdict(dict)
        for feat in selected:
            dest_dir = subset_root / "train_data" / feat.sample.group_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(feat.sample.image_path, dest_dir / feat.sample.image_name)
            grouped[feat.sample.group_name][feat.sample.image_name] = feat.sample.label
        for group_name, labels in grouped.items():
            (subset_root / "train_data" / group_name / "label.json").write_text(
                json.dumps(dict(sorted(labels.items())), ensure_ascii=False),
                encoding="utf-8",
            )

    @staticmethod
    def _reset_output(subset_root: Path, archive_path: Path, manifest_path: Path) -> None:
        for p in (subset_root, archive_path, manifest_path):
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
