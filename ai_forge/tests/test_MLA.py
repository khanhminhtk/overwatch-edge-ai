import pytest
import torch

from src.modeling.common.rope import ROPE
from src.modeling.recognizer.multi_latent_attention import MultiLatentAttention

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _build_mla(*, use_rope: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0) -> MultiLatentAttention:
    rope = ROPE(dim=16, base=10000).to(device=DEVICE) if use_rope else None
    return MultiLatentAttention(
        device=DEVICE,
        num_heads=4,
        head_dim=16,
        out_feature=64,
        d_c=128,
        d_c1=64,
        rope=rope,
        attn_drop=attn_drop,
        proj_drop=proj_drop,
    ).to(device=DEVICE)


def test_mla_forward_shape():
    mla = _build_mla()

    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, mla.d_model, device=DEVICE)
    y = mla(x)

    assert y.shape == (batch_size, seq_len, mla.out_feature)


def test_mla_forward_with_attn_mask():
    mla = _build_mla()

    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, mla.d_model, device=DEVICE)
    attn_mask = torch.ones(seq_len, seq_len, device=DEVICE)
    y = mla(x, attn_mask=attn_mask)

    assert y.shape == (batch_size, seq_len, mla.out_feature)


def test_forward_raises_when_x_not_3d():
    mla = _build_mla()
    x = torch.randn(2, mla.d_model, device=DEVICE)

    with pytest.raises(ValueError, match="x must be 3D"):
        mla(x)


def test_forward_raises_when_x_last_dim_mismatch():
    mla = _build_mla()
    x = torch.randn(2, 10, mla.d_model + 1, device=DEVICE)

    with pytest.raises(ValueError, match="x last dim must be"):
        mla(x)


def test_forward_raises_when_both_mask_and_attn_mask_provided():
    mla = _build_mla()

    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, mla.d_model, device=DEVICE)
    mask = torch.ones(seq_len, seq_len, device=DEVICE)

    with pytest.raises(ValueError, match="Provide only one of mask or attn_mask"):
        mla(x, mask=mask, attn_mask=mask)


@pytest.mark.parametrize(
    "shape",
    [
        (10,),
        (1, 1, 1, 10, 10),
    ],
)
def test_attn_mask_invalid_dim_raises(shape: tuple[int, ...]):
    mla = _build_mla()

    seq_len = 10
    x = torch.randn(2, seq_len, mla.d_model, device=DEVICE)
    attn_mask = torch.ones(*shape, device=DEVICE)

    with pytest.raises(ValueError, match="attn_mask must have 2 to 4 dims"):
        mla(x, attn_mask=attn_mask)


def test_attn_mask_invalid_last_two_dims_raises():
    mla = _build_mla()

    seq_len = 10
    x = torch.randn(2, seq_len, mla.d_model, device=DEVICE)
    attn_mask = torch.ones(seq_len, seq_len + 1, device=DEVICE)

    with pytest.raises(ValueError, match="last two dims must be"):
        mla(x, attn_mask=attn_mask)


def test_attn_mask_invalid_batch_dim_for_3d_raises():
    mla = _build_mla()

    bsz = 2
    seq_len = 10
    x = torch.randn(bsz, seq_len, mla.d_model, device=DEVICE)
    attn_mask = torch.ones(bsz + 1, seq_len, seq_len, device=DEVICE)

    with pytest.raises(ValueError, match="batch dim must be 1 or 2"):
        mla(x, attn_mask=attn_mask)


def test_attn_mask_invalid_head_dim_for_4d_raises():
    mla = _build_mla()

    bsz = 2
    seq_len = 10
    x = torch.randn(bsz, seq_len, mla.d_model, device=DEVICE)
    attn_mask = torch.ones(bsz, 2, seq_len, seq_len, device=DEVICE)

    with pytest.raises(ValueError, match="head dim must be 1 or 4"):
        mla(x, attn_mask=attn_mask)


@pytest.mark.parametrize(
    "mask_builder",
    [
        lambda b, h, s: torch.ones(s, s, device=DEVICE),
        lambda b, h, s: torch.ones(1, s, s, device=DEVICE),
        lambda b, h, s: torch.ones(b, s, s, device=DEVICE),
        lambda b, h, s: torch.ones(1, 1, s, s, device=DEVICE),
        lambda b, h, s: torch.ones(b, 1, s, s, device=DEVICE),
        lambda b, h, s: torch.ones(b, h, s, s, device=DEVICE),
    ],
)
def test_attn_mask_accepts_supported_shapes(mask_builder):
    mla = _build_mla()

    bsz = 2
    seq_len = 10
    x = torch.randn(bsz, seq_len, mla.d_model, device=DEVICE)
    attn_mask = mask_builder(bsz, mla.num_heads, seq_len)
    y = mla(x, attn_mask=attn_mask)

    assert y.shape == (bsz, seq_len, mla.out_feature)


def test_mask_zero_positions_change_output():
    torch.manual_seed(0)
    mla = _build_mla(attn_drop=0.0, proj_drop=0.0)
    mla.eval()

    bsz = 2
    seq_len = 10
    x = torch.randn(bsz, seq_len, mla.d_model, device=DEVICE)
    full_mask = torch.ones(seq_len, seq_len, device=DEVICE)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=DEVICE))

    y_full = mla(x, attn_mask=full_mask)
    y_causal = mla(x, attn_mask=causal_mask)

    assert not torch.allclose(y_full, y_causal)


def test_constructor_validations():
    with pytest.raises(ValueError, match="num_heads must be > 0"):
        MultiLatentAttention(
            device=DEVICE,
            num_heads=0,
            head_dim=16,
            out_feature=64,
            d_c=128,
            d_c1=64,
        )

    with pytest.raises(ValueError, match="head_dim must be > 0"):
        MultiLatentAttention(
            device=DEVICE,
            num_heads=4,
            head_dim=0,
            out_feature=64,
            d_c=128,
            d_c1=64,
        )

    with pytest.raises(ValueError, match="out_feature must be > 0"):
        MultiLatentAttention(
            device=DEVICE,
            num_heads=4,
            head_dim=16,
            out_feature=0,
            d_c=128,
            d_c1=64,
        )

    rope = ROPE(dim=16, base=10000).to(device=DEVICE)
    with pytest.raises(ValueError, match="head_dim must be even when rope is enabled"):
        MultiLatentAttention(
            device=DEVICE,
            num_heads=4,
            head_dim=15,
            out_feature=64,
            d_c=128,
            d_c1=64,
            rope=rope,
        )


def test_backward_pass_has_finite_gradients():
    torch.manual_seed(1)
    mla = _build_mla()
    x = torch.randn(2, 10, mla.d_model, device=DEVICE, requires_grad=True)
    y = mla(x)
    loss = y.sum()
    loss.backward()

    for param in mla.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
