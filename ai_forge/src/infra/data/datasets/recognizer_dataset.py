import json
from collections.abc import Callable
from typing import List, Tuple
from pathlib import Path

import torch
import cv2
import numpy as np

from src.domain.ports.data.dataset_port import DatasetPort
from src.domain.value_objet.dataset import Sample


class RecognizerDataset(torch.utils.data.Dataset, DatasetPort):
    """
    Dataset cho OCR recognizer.

    Returns:
    {
        "image": Tensor[P, C, H, W], float32,
        "label": str,
        "path": str
    }
    """
    def __init__(
        self,
        root_dir: str | Path,
        data_path: str = "ai_forge/data/recognizer/train_data",
        num_patches: int = 16,
        patch_size: Tuple[int, int] = (64, 64),
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        if num_patches <= 0:
            raise ValueError("RecognizerDataset.__init__: num_patches must be > 0")
        if patch_size[0] <= 0 or patch_size[1] <= 0:
            raise ValueError("RecognizerDataset.__init__: patch_size dimensions must be > 0")

        path_data = Path(root_dir) / data_path
        if not path_data.is_dir():
            raise ValueError(f"RecognizerDataset.__init__: Dataset directory does not exist: {path_data}")

        self.samples: List[Sample] = []
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.transform = transform
        for sample_path in sorted(path_data.iterdir(), key=lambda p: p.name):
            if not sample_path.is_dir():
                continue
            label_path = sample_path / "label.json"
            if not label_path.is_file():
                continue

            with open(label_path, "r", encoding="utf-8") as f:
                labels = json.load(f)

            for image_name in sorted(labels.keys()):
                image_path = sample_path / image_name
                if not image_path.is_file():
                    continue
                self.samples.append(Sample(path=image_path, label=labels[image_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(str(sample.path))
        if image is None:
            raise ValueError(f"RecognizerDataset.__getitem__: Could not read image at {sample.path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, width = image.shape[:2]
        if width < self.num_patches:
            raise ValueError(
                f"RecognizerDataset.__getitem__: Image width ({width}) is smaller than num_patches ({self.num_patches}) for {sample.path}"
            )

        part_width = width // self.num_patches
        patches = []
        for i in range(self.num_patches):
            start_x = i * part_width
            end_x = (i + 1) * part_width if i < self.num_patches - 1 else width
            patch = image[:, start_x:end_x]
            patch = cv2.resize(patch, self.patch_size)
            patches.append(patch)

        image = torch.from_numpy(np.stack(patches, axis=0)).float() / 255.0  # [num_patches, H, W, C]
        image = image.permute(0, 3, 1, 2)  # [num_patches, C, H, W]
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": sample.label,
            "path": str(sample.path)
        }
