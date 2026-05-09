import torch
import torchinfo
import torch.nn as nn

from src.modeling.recognizer.vision_transformer import VisionTransformers
from src.modeling.recognizer.backbones import MobileNetBackBone
from src.modeling.recognizer.multi_latent_attention import MultiLatentAttention
from src.modeling.recognizer.deepseek_moe import DeepSeekMOE
from src.modeling.common.activations import SwiGLU

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

def _initialize_lazy_modules(model: nn.Module, device: str) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, SwiGLU):
                dummy = torch.randn(1, module.out_feature, device=device)
                _ = module(dummy)

model = _init_vit(num_layers=2).to(DEVICE)
input_tensor = torch.randn(2, 16, 3, 224, 224).to(DEVICE)
_initialize_lazy_modules(model, DEVICE)
torchinfo.summary(
    model,
    input_data=input_tensor,
    depth=3,
    col_names=("input_size", "output_size", "num_params", "trainable"),
)
