import torch
import torch.nn as nn

from src.modeling.recognizer.backbones.mobinet_backbone import MobileNetBackBone
from src.modeling.recognizer.deepseek_moe import DeepSeekMOE
from src.modeling.recognizer.vision_transformer import VisionTransformers
from src.modeling.recognizer.multi_latent_attention import MultiLatentAttention
from src.modeling.common.rope import ROPE
from src.utils.config_loader import ConfigLoader
from src.domain.value_objet.config import RecognizerConfig

def load_model_vit_ctc(
    config: ConfigLoader,
    device: torch.device | str 
) -> VisionTransformers:
    config_recog: RecognizerConfig = config.load_recognizer()
    train_cfg = dict(config_recog.config_training)
    backbone_cfg = dict(train_cfg.get("backbone", {}))
    dataset_cfg = dict(train_cfg.get("dataset", {}))
    max_seq_len = int(dataset_cfg.get("num_patches", 512))
    has_cls_token = bool(dataset_cfg.get("has_cls_token", False))
    backbone = MobileNetBackBone(
        device=device,
        in_feature=None,
        out_feature=config_recog.d_model
    )
    backbone.unfreeze_feature_extractor(
        classifier=bool(backbone_cfg.get("unfreeze_classifier", False)),
        num_layers_from_classification=int(backbone_cfg.get("unfreeze_num_layers", 0)),
    )

    rope = ROPE(
        dim=config_recog.dim_rope,
        max_seq_len=max_seq_len,
        base=10_000,
        has_cls_token=has_cls_token,
    )

    attention = MultiLatentAttention(
        device=device,
        num_heads=config_recog.num_heads,
        head_dim=config_recog.d_model // config_recog.num_heads,
        out_feature=config_recog.d_model,
        d_c=config_recog.d_c,
        d_c1=config_recog.d_c1,
        rope=rope,
        attn_drop=config_recog.attn_drop,
        proj_drop=config_recog.proj_drop,
    )

    moe = DeepSeekMOE(
        device=device,
        n_routed_experts=config_recog.n_routed_experts,
        n_shared_experts=config_recog.n_shared_experts,
        d_model=config_recog.d_model,
        expert_hidden_dim=config_recog.expert_hidden_dim,
        top_k=config_recog.top_k,
        dropout=config_recog.moe_dropout,
    )

    vit = VisionTransformers(
        device=device,
        backbone=backbone,
        attentions=nn.ModuleList([attention]),
        moes=nn.ModuleList([moe]),
        d_model=config_recog.d_model
    )

    return vit
