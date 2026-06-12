from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
cv2 = pytest.importorskip("cv2")

from src.infra.data.dataloaders.ctc_collate import CTCLabelEncoder
from src.utils.config_loader import ConfigLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TinyVisionModel(nn.Module):
    def __init__(self, d_model: int, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self._device = torch.device(device)
        self._d_model = d_model
        self.patch_proj = nn.Linear(3 * 16 * 16, d_model, device=self._device)

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def device(self) -> torch.device:
        return self._device

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        has_cls_token: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del attn_mask, has_cls_token
        bsz, num_patches = x.shape[:2]
        x = x.reshape(bsz, num_patches, -1)
        hidden = self.patch_proj(x)
        aux_loss = hidden.abs().mean() * 0.0
        return hidden, aux_loss


def create_real_dataset(
    root_dir: Path,
    num_samples: int = 20,
    num_patches: int = 10,
    image_height: int = 16,
    patch_width: int = 16,
) -> tuple[Path, list[str]]:
    data_root = root_dir / "recognizer_data"
    train_dir = data_root / "train"
    batch_dir = train_dir / "batch_01"
    batch_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[str, str] = {}
    words: list[str] = []
    width = num_patches * patch_width

    for idx in range(num_samples):
        text = f"abc{idx % 10}"
        words.append(text)
        file_name = f"img_{idx:03d}.png"
        labels[file_name] = text
        image_path = batch_dir / file_name

        canvas = torch.randint(0, 255, (image_height, width, 3), dtype=torch.uint8).numpy()
        cv2.putText(canvas, text, (2, image_height - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        ok = cv2.imwrite(str(image_path), canvas)
        assert ok, f"Failed to write image: {image_path}"

    (batch_dir / "label.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    return data_root, words


def write_training_yaml(config_path: Path, data_root: Path) -> None:
    yaml_text = f"""
recognizer:
  architecture:
    d_model: 32
    num_heads: 4
    d_c: 8
    d_c1: 8
    dim_rope: 8
    n_shared_experts: 1
    n_routed_experts: 2
    top_k: 1
    expert_hidden_dim: 16
    attn_drop: 0.0
    proj_drop: 0.0
    moe_dropout: 0.0
    config_training:
      dataset:
        data_root: {data_root}
        train_split: train
        val_split: val
        val_ratio: 0.2
        split_seed: 42
        num_patches: 10
        patch_size: [16, 16]
      vocab:
        mode: full
        full_charset: "abc0123456789"
      dataloader:
        batch_size: 4
        shuffle: true
        num_workers: 0
        pin_memory: false
        drop_last: false
        persistent_workers: false
      optimizer:
        lr: 0.001
        weight_decay: 0.0
      loss:
        blank: 0
        zero_infinity: true
        aux_loss_weight: 0.0
      loop:
        epochs: 2
        grad_clip_norm: 1.0
        seed: 42
      hooks:
        tensorboard:
          log_every_n_steps: 1
          log_lr_every_n_steps: 0
          flush_every_n_steps: 1
          log_histograms_every_n_steps: 0
          max_modules: 10
          leaf_only: true
        nan_guard:
          enabled: false
          check_inputs: false
          check_outputs: false
        grad_norm:
          enabled: false
          module_names: []
      io:
        save_dir: {config_path.parent / "checkpoints"}
        tensorboard_dir: {config_path.parent / "tb"}
        best_loss_checkpoint_name: best_loss.pt
        best_cer_checkpoint_name: best_cer.pt
        best_checkpoint_name: best_loss.pt
        last_checkpoint_name: last_checkpoint.pt
"""
    config_path.write_text(yaml_text.strip() + "\n", encoding="utf-8")


def build_loader(tmp_path: Path) -> tuple[ConfigLoader, CTCLabelEncoder]:
    data_root, words = create_real_dataset(root_dir=tmp_path, num_samples=20)
    yaml_path = tmp_path / "recognizer_ctc_test.yaml"
    write_training_yaml(config_path=yaml_path, data_root=data_root)

    loader = ConfigLoader(
        yaml_relative_paths=[yaml_path.name],
        project_root=tmp_path,
        env_relative_path="not_used.env",
    )
    vocab = sorted({ch for w in words for ch in w})
    encoder = CTCLabelEncoder(vocab=vocab)
    return loader, encoder
