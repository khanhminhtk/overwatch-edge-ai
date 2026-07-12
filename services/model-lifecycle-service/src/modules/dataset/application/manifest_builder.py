from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.modules.dataset.application.image_utils import (
    compute_sha256,
    is_corrupted_image,
    iter_image_files,
    load_image_rgb,
    zip_directory,
)
from src.platform.logger import Logger


class ManifestBuilder:
    def __init__(self, data_types: str, logger: Logger) -> None:
        if data_types not in ("detection", "recognition"):
            raise ValueError(f"Unsupported data_types: {data_types}")
        self._data_types = data_types
        self._logger = logger

    def build(
        self,
        data_root: Path,
        *,
        dataset_version: str = "v1",
        dataset_name: str | None = None,
        archive_path: Path | None = None,
    ) -> tuple[dict[str, Any], Path, Path]:
        archive_file = self._build_archive(data_root, archive_path)
        if self._data_types == "detection":
            manifest = self._build_detection_manifest(data_root, dataset_version, dataset_name, archive_file)
        else:
            manifest = self._build_recognition_manifest(data_root, dataset_version, dataset_name, archive_file)

        manifest_path = archive_file.parent / f"{self._data_types}_dataset_manifest.json"
        self._write_json(manifest_path, manifest)
        return manifest, archive_file, manifest_path

    def _build_detection_manifest(
        self, data_root: Path, version: str, name: str | None, archive: Path
    ) -> dict[str, Any]:
        splits = self._detection_splits(data_root)
        return {
            "schema_version": "1.0",
            "dataset_name": name or "handwriting_detection",
            "dataset_version": version,
            "task": "detection",
            "created_at": _utcnow_iso(),
            "archive": self._archive_meta(archive),
            "expected_structure": {"root_dir": data_root.name, "splits": splits},
            "stats": self._detection_stats(splits),
            "image_statistics": self._detection_image_stats(data_root, splits),
            "quality": self._detection_quality(data_root, splits),
        }

    def _build_recognition_manifest(
        self, data_root: Path, version: str, name: str | None, archive: Path
    ) -> dict[str, Any]:
        samples = self._collect_recognition_samples(data_root)
        splits = self._recognition_splits(data_root, samples)
        return {
            "schema_version": "1.0",
            "dataset_name": name or "handwriting_recognizer",
            "dataset_version": version,
            "task": "recognition",
            "created_at": _utcnow_iso(),
            "archive": self._archive_meta(archive),
            "expected_structure": {"root_dir": data_root.name, "splits": splits},
            "stats": self._recognition_stats(splits),
            "image_statistics": self._recognition_image_stats(samples),
            "quality": self._recognition_quality(data_root),
        }

    def _detection_splits(self, data_root: Path) -> dict[str, dict[str, Any]]:
        result = {}
        for split in ("train", "val", "test"):
            img_dir = data_root / split / "images"
            result[split] = {
                "enabled": img_dir.exists(),
                "images": sum(1 for _ in iter_image_files(img_dir)) if img_dir.exists() else 0,
            }
        return result

    def _collect_recognition_samples(
        self, data_root: Path
    ) -> dict[str, list[tuple[str, str, Path, str]]]:
        result: dict[str, list[tuple[str, str, Path, str]]] = {}
        for split_name, split_dir in (("train", "train_data"), ("val", "val_data"), ("test", "test_data")):
            split_root = data_root / split_dir
            samples: list[tuple[str, str, Path, str]] = []
            if split_root.exists():
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
                            samples.append((group_dir.name, img_name, img_path, str(label)))
            result[split_name] = samples
        return result

    def _recognition_splits(
        self, data_root: Path, samples: dict
    ) -> dict[str, dict[str, Any]]:
        result = {}
        for split_name, split_dir in (("train", "train_data"), ("val", "val_data"), ("test", "test_data")):
            split_root = data_root / split_dir
            result[split_name] = {"enabled": split_root.exists(), "samples": len(samples[split_name])}
        return result

    @staticmethod
    def _detection_stats(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
        total = sum(s["images"] for s in splits.values())
        denom = total or 1
        return {k: {"images": v["images"], "ratio": round(v["images"] / denom, 4)} for k, v in splits.items()}

    @staticmethod
    def _recognition_stats(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
        total = sum(s["samples"] for s in splits.values())
        denom = total or 1
        return {k: {"samples": v["samples"], "ratio": round(v["samples"] / denom, 4)} for k, v in splits.items()}

    def _detection_image_stats(self, data_root: Path, splits: dict) -> dict[str, Any]:
        return {k: self._compute_image_stats(data_root / k / "images") for k, v in splits.items() if v["enabled"]}

    def _recognition_image_stats(
        self, samples: dict[str, list[tuple[str, str, Path, str]]]
    ) -> dict[str, Any]:
        result = {}
        for split_name, split_samples in samples.items():
            if not split_samples:
                continue
            ch_sums = [0.0, 0.0, 0.0]
            ch_sq_sums = [0.0, 0.0, 0.0]
            total_px = 0
            for _, _, img_path, _ in split_samples:
                rgb = load_image_rgb(img_path)
                if rgb is None:
                    continue
                px = rgb.reshape(-1, 3)
                total_px += px.shape[0]
                s = px.sum(axis=0)
                sq = (px * px).sum(axis=0)
                for i in range(3):
                    ch_sums[i] += float(s[i])
                    ch_sq_sums[i] += float(sq[i])
            result[split_name] = (
                self._mean_std_map(ch_sums, ch_sq_sums, total_px) if total_px
                else {"mean_rgb": {"red": 0, "green": 0, "blue": 0}, "std_rgb": {"red": 0, "green": 0, "blue": 0}}
            )
        return result

    def _compute_image_stats(self, img_dir: Path) -> dict:
        ch_sums = [0.0, 0.0, 0.0]
        ch_sq_sums = [0.0, 0.0, 0.0]
        total_px = 0
        for img_path in iter_image_files(img_dir):
            rgb = load_image_rgb(img_path)
            if rgb is None:
                continue
            px = rgb.reshape(-1, 3)
            total_px += px.shape[0]
            s = px.sum(axis=0)
            sq = (px * px).sum(axis=0)
            for i in range(3):
                ch_sums[i] += float(s[i])
                ch_sq_sums[i] += float(sq[i])
        return (
            self._mean_std_map(ch_sums, ch_sq_sums, total_px) if total_px
            else {"mean_rgb": {"red": 0, "green": 0, "blue": 0}, "std_rgb": {"red": 0, "green": 0, "blue": 0}}
        )

    @staticmethod
    def _mean_std_map(sums: list[float], sq_sums: list[float], total_px: int) -> dict:
        means = [s / total_px for s in sums]
        stds = [max((sq_sums[i] / total_px) - means[i] ** 2, 0.0) ** 0.5 for i in range(3)]
        return {
            "mean_rgb": {"red": round(means[0], 6), "green": round(means[1], 6), "blue": round(means[2], 6)},
            "std_rgb": {"red": round(stds[0], 6), "green": round(stds[1], 6), "blue": round(stds[2], 6)},
        }

    def _detection_quality(self, data_root: Path, splits: dict) -> dict[str, Any]:
        missing_imgs = 0
        missing_lbls = 0
        corrupt = 0
        for split in ("train", "val", "test"):
            img_dir = data_root / split / "images"
            lbl_dir = data_root / split / "labels"
            if not img_dir.exists():
                continue
            img_stems = {p.stem for p in iter_image_files(img_dir)}
            lbl_stems = {p.stem for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else set()
            missing_lbls += len(img_stems - lbl_stems)
            missing_imgs += len(lbl_stems - img_stems)
            for stem in img_stems:
                for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    p = img_dir / f"{stem}{ext}"
                    if p.exists() and is_corrupted_image(p):
                        corrupt += 1
                        break
        return self._quality_result(missing_imgs, missing_lbls, corrupt, 0)

    def _recognition_quality(self, data_root: Path) -> dict[str, Any]:
        missing_imgs = 0
        missing_lbls = 0
        invalid = 0
        corrupt = 0
        for split_dir in ("train_data", "val_data", "test_data"):
            split_root = data_root / split_dir
            if not split_root.exists():
                continue
            for group_dir in split_root.iterdir():
                if not group_dir.is_dir():
                    continue
                label_path = group_dir / "label.json"
                if not label_path.exists():
                    continue
                img_files = {p.name for p in iter_image_files(group_dir)}
                try:
                    lbl_names = set(json.loads(label_path.read_text("utf-8")).keys())
                except (json.JSONDecodeError, OSError):
                    invalid += 1
                    continue
                missing_imgs += len(lbl_names - img_files)
                missing_lbls += len(img_files - lbl_names)
        return self._quality_result(missing_imgs, missing_lbls, corrupt, invalid)

    @staticmethod
    def _quality_result(missing_imgs: int, missing_lbls: int, corrupt: int, invalid: int) -> dict:
        return {
            "validation_status": "failed" if any((missing_imgs, missing_lbls, corrupt, invalid)) else "passed",
            "missing_images": missing_imgs,
            "missing_labels": missing_lbls,
            "corrupted_images": corrupt,
            "invalid_labels": invalid,
        }

    @staticmethod
    def _build_archive(data_root: Path, archive_path: Path | None) -> Path:
        dest = archive_path or (data_root.parent / f"{data_root.name}.zip")
        return zip_directory(data_root, dest)

    @staticmethod
    def _archive_meta(archive_path: Path) -> dict[str, Any]:
        return {
            "format": "zip",
            "compression": "deflate",
            "file_path": str(archive_path),
            "size_bytes": archive_path.stat().st_size,
            "checksum_sha256": compute_sha256(archive_path),
        }

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
