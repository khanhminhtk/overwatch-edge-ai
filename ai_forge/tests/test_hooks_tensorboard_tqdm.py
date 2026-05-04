from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.infra.modeling.hooks import tqdm_wrap
from src.infra.modeling.hooks.tensorboard_scalar import TensorBoardScalarHook


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def test_tqdm_wrap_iterates() -> None:
    data = [1, 2, 3]
    wrapped = tqdm_wrap(data, desc="test", total=3, leave=False)
    assert list(wrapped) == data


def test_tensorboard_scalar_hook_step_validation() -> None:
    model = _tiny_model()
    with TensorBoardScalarHook(model=model, log_dir="/tmp/tb_hook_test", module_names=["0"]) as hook:
        with pytest.raises(ValueError, match="n must be > 0"):
            hook.step(0)


def test_tensorboard_scalar_hook_writes_events(tmp_path: Path) -> None:
    pytest.importorskip("tensorboard")
    model = _tiny_model()
    x = torch.randn(4, 4)
    log_dir = tmp_path / "tb"

    with TensorBoardScalarHook(
        model=model,
        log_dir=str(log_dir),
        module_names=["0", "2"],
        log_every_n_steps=1,
        log_histograms_every_n_steps=1,
        flush_every_n_steps=1,
    ) as hook:
        out = model(x)
        loss = out.pow(2).mean()
        loss.backward()
        hook.log_optimizer_lrs(torch.optim.AdamW(model.parameters(), lr=1e-3))
        hook.step()

    event_files = list(log_dir.glob("events.out.tfevents.*"))
    assert len(event_files) > 0


def test_tensorboard_scalar_hook_missing_module_raises(tmp_path: Path) -> None:
    model = _tiny_model()
    with pytest.raises(ValueError, match="modules not found"):
        TensorBoardScalarHook(model=model, log_dir=str(tmp_path), module_names=["missing"])
