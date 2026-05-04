from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.infra.modeling.hooks.feature_capture import FeatureCaptureHook


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def test_feature_capture_hook_captures_target_module_output() -> None:
    model = _tiny_model()
    x = torch.randn(3, 4)

    with FeatureCaptureHook(model=model, module_names=["0"], detach=True, clone=True, to_cpu=True) as hook:
        _ = model(x)

    assert "0" in hook.outputs
    out = hook.outputs["0"]
    assert torch.is_tensor(out)
    assert out.shape == (3, 8)
    assert out.device.type == "cpu"


def test_feature_capture_hook_register_twice_raises() -> None:
    model = _tiny_model()
    hook = FeatureCaptureHook(model=model, module_names=["0"])
    hook.register()
    with pytest.raises(RuntimeError, match="already registered"):
        hook.register()
    hook.remove()


def test_feature_capture_hook_missing_module_raises() -> None:
    model = _tiny_model()
    with pytest.raises(ValueError, match="modules not found"):
        FeatureCaptureHook(model=model, module_names=["not_exists"])
