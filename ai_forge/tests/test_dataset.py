import os
import json
from pathlib import Path
from collections import defaultdict

import torch

from src.infra.data.datasets.recognizer_dataset import RecognizerDataset
from src.infra.data.transforms.recognizer_transforms import build_recognizer_train_transform, build_recognizer_val_transform


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_DIR = f"{Path(__file__).parent.parent}"
DATA_DIR_PATH = f"{ROOT_DIR}/data_test/recognizer/train_data"
NUM_PATCHES = 10
PATCH_SIZE = (64, 64)
MAX_IMAGES_PER_FOLDER = 25

def test_recognizer_dataset():
    dataset = RecognizerDataset(
        root_dir=ROOT_DIR,
        data_path=DATA_DIR_PATH,
        transform=None,
        num_patches=NUM_PATCHES,
        patch_size=PATCH_SIZE
    )

    total_samples = 0

    for folder in os.listdir(DATA_DIR_PATH):
        with open(os.path.join(DATA_DIR_PATH, folder, "label.json"), "r", encoding="utf-8") as f:
            labels = json.load(f)
            total_samples += len(labels)

    assert len(dataset) == total_samples

    indices_by_folder = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        folder_name = Path(sample.path).parent.name
        if len(indices_by_folder[folder_name]) < MAX_IMAGES_PER_FOLDER:
            indices_by_folder[folder_name].append(idx)

    selected_indices = [idx for indices in indices_by_folder.values() for idx in indices]
    selected_indices = selected_indices[:5]
    assert len(selected_indices) > 0

    for i in selected_indices:
        data = dataset[i]
        assert isinstance(data['image'], torch.Tensor)
        assert isinstance(data['label'], str)
        assert isinstance(data['path'], str)
        assert data['image'].shape == (NUM_PATCHES, 3, PATCH_SIZE[0], PATCH_SIZE[1])
        assert len(data['label']) > 0
        assert os.path.isfile(data['path'])

def test_recognizer_dataset_invalid_num_patches():
    try:
        RecognizerDataset(
            root_dir=ROOT_DIR,
            data_path=DATA_DIR_PATH,
            num_patches=0
        )
    except ValueError as e:
        assert str(e) == "RecognizerDataset.__init__: num_patches must be > 0"

def test_recognizer_dataset_invalid_patch_size():
    try:
        RecognizerDataset(
            root_dir=ROOT_DIR,
            data_path=DATA_DIR_PATH,
            patch_size=(0, 64)
        )
    except ValueError as e:
        assert str(e) == "RecognizerDataset.__init__: patch_size dimensions must be > 0"

    try:
        RecognizerDataset(
            root_dir=ROOT_DIR,
            data_path=DATA_DIR_PATH,
            patch_size=(64, 0)
        )
    except ValueError as e:
        assert str(e) == "RecognizerDataset.__init__: patch_size dimensions must be > 0"

def test_recognizer_dataset_nonexistent_directory():
    try:
        RecognizerDataset(
            root_dir=ROOT_DIR,
            data_path="nonexistent_directory"
        )
    except ValueError as e:
        expected_path = Path(ROOT_DIR) / "nonexistent_directory"
        assert str(e) == f"RecognizerDataset.__init__: Dataset directory does not exist: {expected_path}"

def test_recognizer_dataset_image_width_smaller_than_num_patches():
    dataset = RecognizerDataset(
        root_dir=ROOT_DIR,
        data_path=DATA_DIR_PATH,
        num_patches=NUM_PATCHES,
        patch_size=PATCH_SIZE
    )

    for idx in range(len(dataset)):
        try:
            dataset[idx]
        except ValueError as e:
            assert "Image width" in str(e)

def test_recognizer_dataset_invalid_image_path():
    dataset = RecognizerDataset(
        root_dir=ROOT_DIR,
        data_path=DATA_DIR_PATH,
        num_patches=NUM_PATCHES,
        patch_size=PATCH_SIZE
    )

    for idx in range(len(dataset)):
        sample = dataset.samples[idx]
        if not os.path.isfile(sample.path):
            try:
                dataset[idx]
            except ValueError as e:
                assert f"Could not read image at {sample.path}" in str(e)

def test_recognizer_dataset_with_transform():
    transform = build_recognizer_train_transform()
    dataset = RecognizerDataset(
        root_dir=ROOT_DIR,
        data_path=DATA_DIR_PATH,
        num_patches=NUM_PATCHES,
        patch_size=PATCH_SIZE,
        transform=transform
    )

    for idx in range(len(dataset)):
        data = dataset[idx]
        assert isinstance(data['image'], torch.Tensor)
        assert data['image'].shape == (NUM_PATCHES, 3, PATCH_SIZE[0], PATCH_SIZE[1])