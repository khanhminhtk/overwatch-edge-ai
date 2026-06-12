import torch
import torch.nn as nn

from src.modeling.recognizer.vision_transformer import VisionTransformers


class RecognizerCTCModel(nn.Module):
    def __init__(
        self,
        model: VisionTransformers,
        vocab_size: int,
    ):
        super().__init__()
        self.model = model
        self.classifier = nn.Linear(
            in_features=self.model.d_model,
            out_features=vocab_size, # vocab + blank + unknown
            bias=True,
            device=model.device
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        has_cls_token: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x, aux_loss = self.model(x, attn_mask=attn_mask, has_cls_token=has_cls_token)
        logits = self.classifier(x)
        return logits, aux_loss