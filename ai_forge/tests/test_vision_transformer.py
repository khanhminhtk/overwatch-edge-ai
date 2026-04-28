import torch
import pytest

from src.modeling.recognizer.vision_transformer import VisionTransformers
from src.modeling.recognizer.backbones import MobileNetBackBone
from src.modeling.recognizer.multi_latent_attention import MultiLatentAttention
from src.modeling.recognizer.deepseek_moe import DeepSeekMOE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL = 256

def _init_vit(num_layers: int) -> VisionTransformers:
    backbone = MobileNetBackBone(
            device=DEVICE,
            in_feature=None,
            out_feature=D_MODEL
    )        
    attentions = torch.nn.ModuleList([
        MultiLatentAttention(
            device=DEVICE,
            num_heads=4,
            head_dim=D_MODEL // 4,
            out_feature=D_MODEL,
            d_c=64,
            d_c1=64,
            attn_drop=0.1,
            proj_drop=0.1,
        ) for _ in range(num_layers)
    ])
    moes = torch.nn.ModuleList([
        DeepSeekMOE(
            device=DEVICE,
            n_shared_experts=2,
            n_routed_experts=4,
            top_k=2,
            expert_hidden_dim=128,
            d_model=D_MODEL,
            dropout=0.1
        ) for _ in range(num_layers)
    ])
    vit = VisionTransformers(
        device=DEVICE,
        d_model=D_MODEL,
        backbone=backbone,
        attentions=attentions,
        moes=moes
    )
    return vit

def test_vision_transformer():
    vit = _init_vit(num_layers=2).to(DEVICE)
    batch_size = 2
    patch = 16
    channels = 3
    height = 32
    width = 32
    x = torch.randn(batch_size, patch, channels, height, width).to(DEVICE)
    out, aux_loss = vit(x)
    assert out.shape == (batch_size, patch, D_MODEL)
    assert aux_loss is not None

def test_vision_transformer_with_attn_mask():
    vit = _init_vit(num_layers=2).to(DEVICE)
    batch_size = 2
    patch = 16
    channels = 3
    height = 32
    width = 32
    x = torch.randn(batch_size, patch, channels, height, width).to(DEVICE)
    attn_mask = torch.ones(patch, patch).to(DEVICE)
    out, aux_loss = vit(x, attn_mask=attn_mask)
    assert out.shape == (batch_size, patch, D_MODEL)
    assert aux_loss is not None

def test_vision_transformer_with_cls_token():
    vit = _init_vit(num_layers=2).to(DEVICE)
    batch_size = 2
    patch = 16
    channels = 3
    height = 32
    width = 32
    x = torch.randn(batch_size, patch, channels, height, width).to(DEVICE)
    out, aux_loss = vit(x, has_cls_token=True)
    assert out.shape == (batch_size, patch, D_MODEL)
    assert aux_loss is not None

def test_vision_transformer_invalid_input():
    vit = _init_vit(num_layers=2).to(DEVICE)
    x = torch.randn(2, 3, 32, 32).to(DEVICE)
    try:
        vit(x)
        assert False, "Expected an error due to invalid input shape"
    except Exception as e:
        assert "expected" in str(e).lower()

def test_vision_transformer_invalid_attn_mask():
    vit = _init_vit(num_layers=2).to(DEVICE)
    batch_size = 2
    patch = 16
    channels = 3
    height = 32
    width = 32
    x = torch.randn(batch_size, patch, channels, height, width).to(DEVICE)
    attn_mask = torch.ones(patch + 1, patch + 1).to(DEVICE)  # Invalid shape
    with pytest.raises(ValueError, match="attn_mask last two dims must be"):
        vit(x, attn_mask=attn_mask)

def test_vision_transformer_attn_mask_invalid_dim():
    vit = _init_vit(num_layers=2).to(DEVICE)
    batch_size = 2
    patch = 16
    channels = 3
    height = 32
    width = 32
    x = torch.randn(batch_size, patch, channels, height, width).to(DEVICE)
    attn_mask = torch.ones(patch, patch, patch).to(DEVICE)  # Invalid shape
    with pytest.raises(ValueError, match="attn_mask batch dim must be 1 or"):
        vit(x, attn_mask=attn_mask)
