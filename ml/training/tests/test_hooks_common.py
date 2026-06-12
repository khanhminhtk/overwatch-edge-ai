from __future__ import annotations

import pytest
import torch

from src.infra.modeling.hooks.common import assert_finite, shape_of, tensor_norm


def test_shape_of_tensor_and_nested_data() -> None:
    x = torch.zeros(2, 3, dtype=torch.float32)
    payload = {"a": x, "b": [x, (x,)], "c": 1}

    out = shape_of(payload)

    assert out["a"]["type"] == "tensor"
    assert out["a"]["shape"] == (2, 3)
    assert out["c"] == "int"
    assert out["b"][0]["dtype"] == "torch.float32"


def test_tensor_norm_returns_mean_norm() -> None:
    x = torch.tensor([3.0, 4.0])  # norm = 5
    y = torch.tensor([0.0, 12.0, 5.0])  # norm = 13

    n = tensor_norm([x, y])

    assert n is not None
    assert abs(n - 9.0) < 1e-6


def test_assert_finite_raises_on_inf_or_nan() -> None:
    with pytest.raises(FloatingPointError, match="NaN/Inf detected"):
        assert_finite("bad_tensor", torch.tensor([1.0, float("inf")]))

    with pytest.raises(FloatingPointError, match="NaN/Inf detected"):
        assert_finite("bad_nested", {"x": [torch.tensor([float("nan")])]})


def test_assert_finite_accepts_finite_and_empty() -> None:
    assert_finite("ok_tensor", torch.tensor([1.0, 2.0]))
    assert_finite("ok_empty", torch.tensor([]))
