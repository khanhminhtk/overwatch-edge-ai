from pathlib import Path
from collections.abc import Callable

import torch

from src.infra.data.dataloaders.ctc_collate import CTCLabelEncoder
from src.infra.data.dataloaders.recognizer_dataloader import RecognizerDataLoader
from src.infra.data.datasets.recognizer_dataset import RecognizerDataset
from src.infra.data.transforms.recognizer_transforms import (
    build_recognizer_train_transform,
    build_recognizer_val_transform,
)
from src.utils.config_loader import ConfigLoader


class _TransformSubset(torch.utils.data.Dataset):
    def __init__(
        self,
        subset: torch.utils.data.Dataset,
        transform: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int) -> dict:
        sample = self.subset[index]
        if self.transform is None:
            return sample
        image = self.transform(sample["image"])
        return {
            **sample,
            "image": image,
        }


def _build_dataloader(
    dataset: torch.utils.data.Dataset,
    encoder: CTCLabelEncoder,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    persistent_workers: bool,
) -> RecognizerDataLoader:
    return RecognizerDataLoader(
        dataset=dataset,
        encoder=encoder,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )


def load_dataloader_vit_ctc(
    config: ConfigLoader,
    encoder: CTCLabelEncoder,
    val_ratio: float | None = None,
    split_seed: int | None = None,
) -> tuple[
    torch.utils.data.Dataset,
    RecognizerDataLoader,
    torch.utils.data.Dataset,
    RecognizerDataLoader,
]:
    cfg = config.load_recognizer()
    train_cfg = cfg.config_training
    dataset_cfg = train_cfg["dataset"]
    dataloader_cfg = train_cfg["dataloader"]
    resolved_val_ratio = float(dataset_cfg["val_ratio"]) if val_ratio is None else float(val_ratio)
    resolved_split_seed = int(dataset_cfg["split_seed"]) if split_seed is None else int(split_seed)

    if not (0.0 < resolved_val_ratio < 1.0):
        raise ValueError(f"load_dataloader_vit_ctc: val_ratio must be in (0, 1), got {resolved_val_ratio}")

    data_root = Path(dataset_cfg["data_root"])
    train_split = str(dataset_cfg["train_split"])

    full_train = RecognizerDataset(
        root_dir=data_root.parent,
        data_path=f"{data_root.name}/{train_split}",
        num_patches=int(dataset_cfg["num_patches"]),
        patch_size=tuple(dataset_cfg["patch_size"]),
    )
    if len(full_train) < 2:
        raise ValueError("load_dataloader_vit_ctc: Need at least 2 samples in train split to create train/val split")

    val_len = max(1, int(len(full_train) * resolved_val_ratio))
    train_len = len(full_train) - val_len
    if train_len <= 0:
        train_len = len(full_train) - 1
        val_len = 1

    generator = torch.Generator().manual_seed(resolved_split_seed)
    train_subset, val_subset = torch.utils.data.random_split(
        full_train,
        [train_len, val_len],
        generator=generator,
    )
    transform_cfg = dict(dataset_cfg.get("transforms", {}))
    enable_transforms = bool(transform_cfg.get("enabled", True))
    train_transform = None
    val_transform = None
    if enable_transforms:
        train_transform = build_recognizer_train_transform(
            **dict(transform_cfg.get("train", {}))
        )
        val_transform = build_recognizer_val_transform(
            **dict(transform_cfg.get("val", {}))
        )

    train_dataset = _TransformSubset(
        subset=train_subset,
        transform=train_transform,
    )
    val_dataset = _TransformSubset(
        subset=val_subset,
        transform=val_transform,
    )

    train_loader = _build_dataloader(
        dataset=train_dataset,
        encoder=encoder,
        batch_size=int(dataloader_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(dataloader_cfg["num_workers"]),
        pin_memory=bool(dataloader_cfg["pin_memory"]),
        drop_last=bool(dataloader_cfg["drop_last"]),
        persistent_workers=bool(dataloader_cfg["persistent_workers"]),
    )
    val_loader = _build_dataloader(
        dataset=val_dataset,
        encoder=encoder,
        batch_size=int(dataloader_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(dataloader_cfg["num_workers"]),
        pin_memory=bool(dataloader_cfg["pin_memory"]),
        drop_last=False,
        persistent_workers=bool(dataloader_cfg["persistent_workers"]),
    )

    return train_dataset, train_loader, val_dataset, val_loader
