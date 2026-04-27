import torch
from torch.autograd import gradcheck

from src.modeling.common.activations import SwiGLU

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test_swiglu_forward_shape():
    swiglu = SwiGLU(
        device=DEVICE,
        out_feature=16,
        enable_cache=False
    )
    x = torch.randn(4, 8)
    y = swiglu(x)
    assert y.shape == (4, 16)

def test_swiglu_backward_runs():
    swiglu = SwiGLU(
        device=DEVICE,
        out_feature=16,
        enable_cache=False
    )
    x = torch.rand(2, 8, requires_grad=True)

    y = swiglu(x).sum()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

def test_swiglu_cache_behavior():
    swiglu = SwiGLU(
        device=DEVICE,
        out_feature=16,
        enable_cache=True
    )

    x = torch.rand(2, 8)
    _ = swiglu(x)
    
    assert "out_w_1" in swiglu.cache
    assert "out_w_2" in swiglu.cache
    assert "sigmoid" in swiglu.cache
    assert swiglu.cache["out_w_1"].requires_grad is False
    assert swiglu.cache["out_w_2"].requires_grad is False
    assert swiglu.cache["sigmoid"].requires_grad is False


def test_swiglu_gradcheck():
    swiglu = SwiGLU(
        device="cpu",
        out_feature=16,
        enable_cache=False,
    ).double()

    x = torch.randn(2, 8, dtype=torch.double, requires_grad=True)

    _ = swiglu(x)

    assert gradcheck(swiglu, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)
