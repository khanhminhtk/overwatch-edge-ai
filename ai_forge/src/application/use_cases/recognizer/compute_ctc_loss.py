from typing import Any

import torch

from src.application.use_cases.recognizer.recognizerCTCModel import RecognizerCTCModel
from src.infra.data.dataloaders.ctc_collate import CTCBatch


def compute_ctc_loss(
    model: RecognizerCTCModel,
    batch: CTCBatch,
    criterion: torch.nn.CTCLoss,
    device: torch.device,
    aux_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    images = batch.images.to(device, non_blocking=True)
    targets = batch.targets.to(device, non_blocking=True)
    target_lengths = batch.target_lengths.to(device, non_blocking=True)
    attn_mask = getattr(batch, "attn_mask", None)
    has_cls_token = bool(getattr(batch, "has_cls_token", False))
    logits, aux_loss = model(
        x=images,
        attn_mask=attn_mask,
        has_cls_token=has_cls_token,
    )

    input_ctcloss = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)  # (patch, batch_size, vocab_size)
    input_lengths = torch.full(
        (input_ctcloss.shape[1],),
        fill_value=input_ctcloss.shape[0],
        dtype=torch.long,
        device=device,
    )
    ctc_loss = criterion(input_ctcloss, targets, input_lengths, target_lengths)
    moe_aux = aux_loss if aux_loss is not None else torch.zeros((), device=device)
    total_loss = ctc_loss + aux_loss_weight * moe_aux
    metrics: dict[str, Any] = {
        "loss": float(total_loss.detach().item()),
        "ctc_loss": float(ctc_loss.detach().item()),
        "aux_loss": float(moe_aux.detach().item()),
        "logits": logits.detach(),
    }
    return total_loss, metrics
