from __future__ import annotations

from tests.recognizer_test_helpers import build_loader
from src.application.use_cases.recognizer.build_dataloader import load_dataloader_vit_ctc


def test_load_dataloader_vit_ctc_with_real_images(tmp_path) -> None:
    loader, encoder = build_loader(tmp_path=tmp_path)

    train_dataset, train_loader, val_dataset, val_loader = load_dataloader_vit_ctc(
        config=loader,
        encoder=encoder,
    )

    assert len(train_dataset) > 0
    assert len(val_dataset) > 0
    first_train_batch = next(iter(train_loader))
    first_val_batch = next(iter(val_loader))
    assert first_train_batch.images.ndim == 5
    assert first_val_batch.images.ndim == 5
