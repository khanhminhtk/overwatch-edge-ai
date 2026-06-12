from __future__ import annotations

from tests.recognizer_test_helpers import torch, TinyVisionModel
from src.application.use_cases.recognizer.recognizerCTCModel import RecognizerCTCModel


def test_recognizer_ctc_model_forward_with_real_tensors() -> None:
    model_core = TinyVisionModel(d_model=32, device="cpu")
    model = RecognizerCTCModel(model=model_core, vocab_size=20)
    x = torch.rand(2, 10, 3, 16, 16)

    logits, aux_loss = model(x=x, attn_mask=None, has_cls_token=False)

    assert logits.shape == (2, 10, 20)
    assert aux_loss is not None
    assert aux_loss.dim() == 0
