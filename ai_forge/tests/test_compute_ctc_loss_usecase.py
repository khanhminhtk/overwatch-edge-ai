from __future__ import annotations

from tests.recognizer_test_helpers import torch, TinyVisionModel
from src.application.use_cases.recognizer.compute_ctc_loss import compute_ctc_loss
from src.application.use_cases.recognizer.recognizerCTCModel import RecognizerCTCModel
from src.infra.data.dataloaders.ctc_collate import CTCBatch, CTCLabelEncoder


def test_compute_ctc_loss_with_real_batch_data() -> None:
    encoder = CTCLabelEncoder(vocab=list("abc123"))
    model = RecognizerCTCModel(model=TinyVisionModel(d_model=16), vocab_size=encoder.num_classes)
    texts = ["ab1", "c23"]
    encoded = [encoder.encode(t) for t in texts]
    target_lengths = torch.tensor([len(x) for x in encoded], dtype=torch.long)
    targets = torch.tensor([token for seq in encoded for token in seq], dtype=torch.long)
    batch = CTCBatch(
        images=torch.rand(2, 10, 3, 16, 16),
        targets=targets,
        target_lengths=target_lengths,
        texts=texts,
        paths=["/tmp/a.png", "/tmp/b.png"],
    )
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    total_loss, metrics = compute_ctc_loss(
        model=model,
        batch=batch,
        criterion=criterion,
        device=torch.device("cpu"),
        aux_loss_weight=0.0,
    )

    assert torch.is_tensor(total_loss)
    assert torch.isfinite(total_loss).item() is True
    assert set(["loss", "ctc_loss", "aux_loss", "logits"]).issubset(metrics.keys())
    assert metrics["logits"].shape[0] == 2
