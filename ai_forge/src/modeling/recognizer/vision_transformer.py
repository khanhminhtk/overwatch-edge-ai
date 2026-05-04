from typing import Tuple, List

import torch
import torch.nn as nn

from src.domain.ports.recognizer.vision_transformer import VisionTransformerPort
from src.domain.ports.recognizer.backbone_port import BackBonePort
from src.domain.ports.recognizer.recognizer_component_ports import AttentionPort, MoEPort

class VisionTransformer(nn.Module, VisionTransformerPort):
    def __init__(
        self,
        device: str | torch.device,
        d_model: int,
        attention: AttentionPort,
        moe: MoEPort,
    ) -> None:
        super().__init__()
        self.attention = attention
        self.moe = moe
        self._device = device
        self.d_model = d_model
        self.rms_norm_1 = nn.RMSNorm(d_model)
        self.rms_norm_2 = nn.RMSNorm(d_model)

    def forward(
            self, 
            x: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            has_cls_token: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        # x: (batch_size, patch, d_model)
        attn_input = self.rms_norm_1(x)
        x = x + self.attention(
            attn_input,
            attn_mask=attn_mask,
            has_cls_token=has_cls_token,
        )
        moe_input = self.rms_norm_2(x)
        try:
            moe_result = self.moe(moe_input, return_aux_loss=True)
        except TypeError:
            moe_result = self.moe(moe_input)
        if isinstance(moe_result, tuple):
            moe_out, aux_loss = moe_result
        else:
            moe_out, aux_loss = moe_result, None
        x = x + moe_out
        return x, aux_loss
    

class VisionTransformers(nn.Module, VisionTransformerPort):
    def __init__(
        self,
        device: str | torch.device,
        d_model: int,
        backbone: BackBonePort,
        attentions: nn.ModuleList,
        moes: nn.ModuleList,
    ) -> None:
        super().__init__()
        self._device = device
        self.backbone = backbone
        self._d_model = d_model
        if len(attentions) != len(moes):
            raise ValueError(
                f"attentions and moes must have the same length, got "
                f"{len(attentions)} and {len(moes)}"
            )
        self.layers = nn.ModuleList(
            [
                VisionTransformer(
                    device=device,
                    d_model=d_model,
                    attention=attention,
                    moe=moe
                )
                for attention, moe in zip(attentions, moes)
            ]
        )

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    def forward(
            self, 
            x: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            has_cls_token: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        # x: (batch_size, patch, channels, height, width)
        x = self.backbone(x) # (batch_size, patch, d_model)
        aux_losses: List[torch.Tensor] = []
        for layer in self.layers:
            x, aux_loss = layer(x, attn_mask=attn_mask, has_cls_token=has_cls_token)
            if aux_loss is not None:
                aux_losses.append(aux_loss)

        if not aux_losses:
            return x, None

        mean_aux_loss = torch.stack(aux_losses).mean()
        return x, mean_aux_loss
