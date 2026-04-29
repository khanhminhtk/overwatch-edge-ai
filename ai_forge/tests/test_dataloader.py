import os
import json
from pathlib import Path
from collections import defaultdict

import torch

from src.infra.data.datasets.recognizer_dataset import RecognizerDataset
from src.infra.data.dataloaders.recognizer_dataloader import RecognizerDataLoader
from src.infra.data.dataloaders.ctc_collate import CTCLabelEncoder, build_ctc_collate_fn
from src.infra.data.transforms.recognizer_transforms import build_recognizer_train_transform, build_recognizer_val_transform


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_DIR = f"{Path(__file__).parent.parent}"
DATA_DIR_PATH = f"{ROOT_DIR}/data_test/recognizer/train_data"
NUM_PATCHES = 10
PATCH_SIZE = (64, 64)
MAX_IMAGES_PER_FOLDER = 25
VI_FULL_VOCAB = (
    "aàáảãạăằắẳẵặâầấẩẫậ"
    "bcdđ"
    "eèéẻẽẹêềếểễệ"
    "g"
    "h"
    "iìíỉĩị"
    "k"
    "l"
    "m"
    "n"
    "oòóỏõọôồốổỗộơờớởỡợ"
    "p"
    "q"
    "r"
    "s"
    "t"
    "uùúủũụưừứửữự"
    "v"
    "x"
    "yỳýỷỹỵ"
    "0123456789"
)


def test_recognizer_dataloader():
    dataset = RecognizerDataset(
        root_dir=ROOT_DIR,
        data_path=DATA_DIR_PATH,
        transform=None,
        num_patches=NUM_PATCHES,
        patch_size=PATCH_SIZE
    )
    encoder = CTCLabelEncoder(vocab=VI_FULL_VOCAB)
    dataloader = RecognizerDataLoader(
        dataset,
        encoder=encoder,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False
        )
    
    num_loops = 0

    for batch in dataloader:
        current_batch_size = batch.images.shape[0]
        assert isinstance(batch.images, torch.Tensor)
        assert isinstance(batch.targets, torch.Tensor)
        assert isinstance(batch.target_lengths, torch.Tensor)
        assert isinstance(batch.texts, list)
        assert isinstance(batch.paths, list)
        assert 1 <= current_batch_size <= 4
        assert len(batch.texts) == current_batch_size
        assert len(batch.paths) == current_batch_size
        assert batch.target_lengths.shape[0] == current_batch_size
        if num_loops == 5:
            break
        num_loops += 1

def test_recognizer_dataloader_invalid_inputs():
    dataset = RecognizerDataset(
        root_dir=ROOT_DIR,
        data_path=DATA_DIR_PATH,
        transform=None,
        num_patches=NUM_PATCHES,
        patch_size=PATCH_SIZE
    )
    encoder = CTCLabelEncoder(vocab=VI_FULL_VOCAB)

    try:
        RecognizerDataLoader(
            dataset="not_a_dataset",
            encoder=encoder
        )
    except ValueError as e:
        assert str(e) == "RecognizerDataLoader.__init__: dataset must implement DatasetPort"

    try:
        RecognizerDataLoader(
            dataset=dataset,
            encoder="not_an_encoder"
        )
    except ValueError as e:
        assert str(e) == "RecognizerDataLoader.__init__: encoder must implement BaseCTCLabelEncoder"

    try:
        RecognizerDataLoader(
            dataset=dataset,
            encoder=encoder,
            batch_size=0
        )
    except ValueError as e:
        assert str(e) == "RecognizerDataLoader.__init__: batch_size must be > 0"

    try:
        RecognizerDataLoader(
            dataset=dataset,
            encoder=encoder,
            num_workers=-1
        )
    except ValueError as e:
        assert str(e) == "RecognizerDataLoader.__init__: num_workers must be >= 0"

