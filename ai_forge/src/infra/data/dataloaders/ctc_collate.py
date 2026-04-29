from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

from src.domain.ports.data.dataloader_port import BaseCTCLabelEncoder


@dataclass(frozen=True)
class CTCBatch:
    images: torch.Tensor
    targets: torch.Tensor
    target_lengths: torch.Tensor
    texts: List[str]
    paths: List[str]


class CTCLabelEncoder(BaseCTCLabelEncoder):
    """
    Encode text labels for CTC training.

    Notes:
    - index 0 is reserved for CTC blank token.
    - unknown characters are mapped to `unk_idx`.
    """

    def __init__(self, vocab: Sequence[str], blank_idx: int = 0, unk_token: str = "<unk>"):
        if blank_idx != 0:
            raise ValueError("CTCLabelEncoder.__init__: this implementation expects blank_idx == 0")
        if len(vocab) == 0:
            raise ValueError("CTCLabelEncoder.__init__: vocab must not be empty")

        self.blank_idx = blank_idx
        self.unk_token = unk_token
        self.vocab = list(dict.fromkeys(vocab))
        self.char_to_idx: Dict[str, int] = {ch: i + 1 for i, ch in enumerate(self.vocab)}
        self.unk_idx = len(self.vocab) + 1
        self.char_to_idx[self.unk_token] = self.unk_idx

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx.get(ch, self.unk_idx) for ch in text]
    
    @property
    def num_classes(self) -> int:
        return len(self.vocab) + 2  # vocab + blank + unk


def build_ctc_collate_fn(encoder: BaseCTCLabelEncoder):
    """
    Collate function for RecognizerDataset output:
    {
        "image": Tensor[P, C, H, W],
        "label": str,
        "path": str
    }
    """

    def _collate(batch: List[dict]) -> CTCBatch:
        if len(batch) == 0:
            raise ValueError("build_ctc_collate_fn._collate: batch must not be empty")

        images = torch.stack([item["image"] for item in batch], dim=0)
        texts = [item["label"] for item in batch]
        paths = [item["path"] for item in batch]

        encoded_labels = [encoder.encode(text) for text in texts]
        target_lengths = torch.tensor([len(seq) for seq in encoded_labels], dtype=torch.long)
        flat_targets = [token for seq in encoded_labels for token in seq]
        targets = torch.tensor(flat_targets, dtype=torch.long)

        return CTCBatch(
            images=images,
            targets=targets,
            target_lengths=target_lengths,
            texts=texts,
            paths=paths,
        )

    return _collate
