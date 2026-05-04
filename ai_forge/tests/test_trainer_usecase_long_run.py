from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.recognizer_test_helpers import PROJECT_ROOT, TinyVisionModel, torch, build_loader
from src.application.use_cases.recognizer.build_dataloader import load_dataloader_vit_ctc
from src.application.use_cases.recognizer.recognizerCTCModel import RecognizerCTCModel
from src.application.use_cases.recognizer.trainer import RecognizerTrainer


@pytest.mark.skipif(
    os.getenv("RUN_LONG_TRAIN_TESTS", "0") != "1",
    reason="Long training integration test. Set RUN_LONG_TRAIN_TESTS=1 when needed.",
)
def test_trainer_long_run_with_real_dataset_and_resume() -> None:
    long_data_root = PROJECT_ROOT / "tests" / "data_long" / "recognizer_train_real_20"
    long_data_root.mkdir(parents=True, exist_ok=True)
    loader, encoder = build_loader(tmp_path=long_data_root)

    cfg = loader.load_recognizer()
    cfg_training = dict(cfg.config_training)
    cfg_training["loop"] = dict(cfg_training["loop"])
    cfg_training["loop"]["epochs"] = 3
    cfg = type(cfg)(
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_c=cfg.d_c,
        d_c1=cfg.d_c1,
        dim_rope=cfg.dim_rope,
        n_shared_experts=cfg.n_shared_experts,
        n_routed_experts=cfg.n_routed_experts,
        top_k=cfg.top_k,
        expert_hidden_dim=cfg.expert_hidden_dim,
        attn_drop=cfg.attn_drop,
        proj_drop=cfg.proj_drop,
        moe_dropout=cfg.moe_dropout,
        config_training=cfg_training,
    )

    train_dataset, train_loader, val_dataset, val_loader = load_dataloader_vit_ctc(
        config=loader,
        encoder=encoder,
    )
    assert len(train_dataset) >= 10
    assert len(val_dataset) >= 2

    model = RecognizerCTCModel(model=TinyVisionModel(d_model=32), vocab_size=encoder.num_classes)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    trainer = RecognizerTrainer(
        config=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        aux_loss_weight=0.0,
    )
    first_run = trainer.train()
    assert Path(str(first_run["last_checkpoint"])).is_file()
    assert Path(str(first_run["best_checkpoint"])).is_file()
    assert Path(str(first_run["best_loss_checkpoint"])).is_file()
    assert Path(str(first_run["best_cer_checkpoint"])).is_file()

    best_loss_ckpt = torch.load(str(first_run["best_loss_checkpoint"]), map_location="cpu")
    best_cer_ckpt = torch.load(str(first_run["best_cer_checkpoint"]), map_location="cpu")
    assert best_loss_ckpt["saved_by"] == "loss"
    assert best_cer_ckpt["saved_by"] == "cer"
    assert isinstance(best_loss_ckpt.get("epoch_idx"), int)
    assert isinstance(best_loss_ckpt.get("epoch"), int)

    trainer_resume = RecognizerTrainer(
        config=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        aux_loss_weight=0.0,
    )
    second_run = trainer_resume.train()
    assert second_run["start_epoch"] >= 2
