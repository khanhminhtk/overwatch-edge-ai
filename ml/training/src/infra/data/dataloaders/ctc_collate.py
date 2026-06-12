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

    def __init__(self, vocab: Sequence[str], blank_idx: int = 0, unk_token: str = "<unk>", with_padding: bool = True):
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
        if len(self.char_to_idx) % 8 != 0 and with_padding:
            padding_size = 8 - (len(self.char_to_idx) % 8)
            for i in range(padding_size):
                pad_token = f"<pad_{i}>"
                self.char_to_idx[pad_token] = len(self.char_to_idx)


    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx.get(ch, self.unk_idx) for ch in text]


    def decode(
        self,
        indices: Sequence[int],
        remove_special_tokens: bool = True,
        ctc_decode: bool = True,
    ) -> str:
        idx_to_char: Dict[int, str] = {
            idx: ch for ch, idx in self.char_to_idx.items()
        }

        pad_indices = {
            idx
            for ch, idx in self.char_to_idx.items()
            if ch.startswith("<pad_") and ch.endswith(">")
        }

        special_indices = {self.blank_idx}
        if remove_special_tokens:
            special_indices.update(pad_indices)
            special_indices.add(self.unk_idx)

        chars: List[str] = []
        prev_idx = None

        for raw_idx in indices:
            idx = int(raw_idx)

            # CTC greedy decode: collapse repeated consecutive tokens
            if ctc_decode and idx == prev_idx:
                continue

            prev_idx = idx

            if remove_special_tokens and idx in special_indices:
                continue

            chars.append(idx_to_char.get(idx, self.unk_token))

        return "".join(chars)


    @property
    def num_classes(self) -> int:
        # number of classes must include blank + vocab + unk + padding
        return max([self.blank_idx, *self.char_to_idx.values()]) + 1

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
