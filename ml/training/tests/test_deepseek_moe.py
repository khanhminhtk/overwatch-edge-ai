import torch
import torch.nn as nn
import pytest

from src.modeling.recognizer.deepseek_moe import DeepSeekMOE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test_deepseek_moe_initialization():
    # Valid initialization
    moe = DeepSeekMOE(
        device=DEVICE,
        n_shared_experts=2,
        n_routed_experts=4,
        top_k=2,
        expert_hidden_dim=128,
        d_model=256,
        dropout=0.1
    )
    assert moe.d_model == 256
    assert moe.n_shared_experts == 2
    assert moe.n_routed_experts == 4
    assert moe.top_k == 2
    assert moe.expert_hidden_dim == 128

    # Invalid d_model
    with pytest.raises(ValueError, match=r"^d_model must be > 0, got -1$"):
        DeepSeekMOE(
            device=DEVICE,
            n_shared_experts=2,
            n_routed_experts=4,
            top_k=2,
            expert_hidden_dim=128,
            d_model=-1,
            dropout=0.1
        )

    # Invalid expert_hidden_dim
    with pytest.raises(ValueError, match=r"^expert_hidden_dim must be > 0, got -1$"):
        DeepSeekMOE(
            device=DEVICE,
            n_shared_experts=2,
            n_routed_experts=4,
            top_k=2,
            expert_hidden_dim=-1,
            d_model=256,
            dropout=0.1
        )

    # Invalid n_routed_experts
    with pytest.raises(ValueError, match=r"^n_routed_experts must be > 0, got -1$"):
        DeepSeekMOE(
            device=DEVICE,
            n_shared_experts=2,
            n_routed_experts=-1,
            top_k=2,
            expert_hidden_dim=128,
            d_model=256,
            dropout=0.1
        )

    # Invalid n_shared_experts
    with pytest.raises(ValueError, match=r"^n_shared_experts must be >= 0, got -1$"):
        DeepSeekMOE(
            device=DEVICE,
            n_shared_experts=-1,
            n_routed_experts=4,
            top_k=2,
            expert_hidden_dim=128,
            d_model=256,
            dropout=0.1
        )

    # Invalid top_k
    with pytest.raises(
        ValueError,
        match=r"^top_k must be in \[1, n_routed_experts\], got top_k=0, n_routed_experts=4$"
    ):
        DeepSeekMOE(
            device=DEVICE,
            n_shared_experts=2,
            n_routed_experts=4,
            top_k=0,
            expert_hidden_dim=128,
            d_model=256,
            dropout=0.1
        )

    with pytest.raises(
        ValueError,
        match=r"^top_k must be in \[1, n_routed_experts\], got top_k=5, n_routed_experts=4$"
    ):
        DeepSeekMOE(
            device=DEVICE,
            n_shared_experts=2,
            n_routed_experts=4,
            top_k=5,
            expert_hidden_dim=128,
            d_model=256,
            dropout=0.1
        )

def test_deepseek_moe_forward_pass():
    batch_size = 2
    patch = 10
    d_model = 256

    moe = DeepSeekMOE(
        device=DEVICE,
        n_shared_experts=2,
        n_routed_experts=4,
        top_k=2,
        expert_hidden_dim=128,
        d_model=d_model,
        dropout=0.1
    ).to(DEVICE)

    input_tensor = torch.randn(batch_size, patch, d_model, device=DEVICE)
    output, _ = moe(input_tensor)

    assert output.shape == (batch_size, patch, d_model)

def test_deepseek_moe_auxiliary_loss():
    batch_size = 2
    patch = 10
    d_model = 256

    moe = DeepSeekMOE(
        device=DEVICE,
        n_shared_experts=2,
        n_routed_experts=4,
        top_k=2,
        expert_hidden_dim=128,
        d_model=d_model,
        dropout=0.1
    ).to(DEVICE)

    input_tensor = torch.randn(batch_size, patch, d_model, device=DEVICE)
    output, aux_loss = moe(input_tensor)

    assert output.shape == (batch_size, patch, d_model)
    assert isinstance(aux_loss, torch.Tensor)
    assert aux_loss.dim() == 0  
    assert torch.isfinite(aux_loss).all()
    assert aux_loss.device == output.device
