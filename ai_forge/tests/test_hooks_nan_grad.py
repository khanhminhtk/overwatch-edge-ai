from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.infra.modeling.hooks.grad_norm import GradNormHook
from src.infra.modeling.hooks.nan_guard import NaNInfGuardHook


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def test_grad_norm_hook_collects_backward_norms() -> None:
    model = _tiny_model()
    x = torch.randn(5, 4, requires_grad=True)

    with GradNormHook(model=model, module_names=["0"]) as hook:
        y = model(x)
        loss = y.pow(2).mean()
        loss.backward()

    assert "0" in hook.grad_input_norms or "0" in hook.grad_output_norms


def test_grad_norm_hook_missing_module_raises() -> None:
    model = _tiny_model()
    with pytest.raises(ValueError, match="modules not found"):
        GradNormHook(model=model, module_names=["missing_layer"])


def test_nan_guard_hook_raises_on_invalid_output() -> None:
    class BadModule(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x / 0.0

    model = nn.Sequential(BadModule())
    x = torch.ones(2, 3)

    with NaNInfGuardHook(model=model, check_inputs=False, check_outputs=True):
        with pytest.raises(FloatingPointError, match="NaN/Inf detected"):
            _ = model(x)
