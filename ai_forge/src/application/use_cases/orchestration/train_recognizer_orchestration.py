from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Protocol

import torch

from src.application.use_cases.recognizer.build_dataloader import load_dataloader_vit_ctc
from src.application.use_cases.recognizer.load_model_vit_ctc import load_model_vit_ctc
from src.application.use_cases.recognizer.recognizerCTCModel import RecognizerCTCModel
from src.application.use_cases.recognizer.trainer import RecognizerTrainer
from src.infra.data.dataloaders.ctc_collate import CTCLabelEncoder
from src.infra.data.datasets.recognizer_dataset import RecognizerDataset
from src.utils.config_loader import ConfigLoader
from src.modeling.common.custom_weight_init import custom_weight_init

DEFAULT_CONFIG_RELATIVE_PATH = "config/training/recognizer_ctc.yaml"
DEFAULT_ENV_RELATIVE_PATH = "config/.env"
LOGGER = logging.getLogger(__name__)


class TrainRecognizerUseCasePort(Protocol):
    def run(self) -> dict[str, Any]:
        """Run recognizer training end-to-end."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _build_vocab_from_dataset(data_root: Path, train_split: str) -> list[str]:
    dataset = RecognizerDataset(
        root_dir=data_root.parent,
        data_path=f"{data_root.name}/{train_split}",
    )
    vocab = sorted({ch for sample in dataset.samples for ch in str(sample.label)})
    if not vocab:
        raise ValueError("train_recognizer_orchestration: empty vocabulary from dataset labels")
    return vocab


def resolve_vocab(config_training: dict[str, Any], data_root: Path, train_split: str) -> list[str]:
    vocab_cfg = config_training.get("vocab", {})
    mode = str(vocab_cfg.get("mode", "full")).strip().lower()
    full_charset = str(vocab_cfg.get("full_charset", "")).strip()
    observed_vocab = _build_vocab_from_dataset(data_root=data_root, train_split=train_split)

    if mode == "auto":
        return observed_vocab
    if mode != "full":
        raise ValueError(f"resolve_vocab: unsupported vocab.mode '{mode}', expected 'full' or 'auto'")
    if not full_charset:
        return observed_vocab

    resolved_vocab = list(dict.fromkeys(full_charset))
    if len(observed_vocab) > 0:
        ratio = len(resolved_vocab) / len(observed_vocab)
        if ratio >= 1.5:
            LOGGER.warning(
                "Configured full vocab is much larger than observed train charset (%s vs %s, ratio=%.2f). "
                "This can increase CTC blank-collapse risk.",
                len(resolved_vocab),
                len(observed_vocab),
                ratio,
            )
    return resolved_vocab


@dataclass
class TrainRecognizerOrchestration(TrainRecognizerUseCasePort):
    config_relative_path: str = DEFAULT_CONFIG_RELATIVE_PATH
    env_relative_path: str = DEFAULT_ENV_RELATIVE_PATH
    project_root: Path | None = None
    device: str | torch.device | None = None
    val_ratio: float | None = None
    split_seed: int | None = None

    def run(self) -> dict[str, Any]:
        root = self.project_root or _project_root()
        config_loader = ConfigLoader(
            yaml_relative_paths=[self.config_relative_path],
            project_root=root,
            env_relative_path=self.env_relative_path,
        )
        recog_cfg = config_loader.load_recognizer()
        config_training = dict(recog_cfg.config_training)

        dataset_cfg = config_training["dataset"]
        data_root = Path(dataset_cfg["data_root"])
        train_split = str(dataset_cfg["train_split"])
        vocab = resolve_vocab(config_training=config_training, data_root=data_root, train_split=train_split)
        encoder = CTCLabelEncoder(vocab=vocab)

        train_dataset, train_loader, val_dataset, val_loader = load_dataloader_vit_ctc(
            config=config_loader,
            encoder=encoder,
            val_ratio=self.val_ratio,
            split_seed=self.split_seed,
        )

        device = torch.device(self.device) if self.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        recognizer_model = load_model_vit_ctc(config=config_loader, device=device)
        model = RecognizerCTCModel(model=recognizer_model, vocab_size=encoder.num_classes).to(device)
        model.apply(custom_weight_init)

        optimizer_cfg = config_training["optimizer"]
        loss_cfg = config_training["loss"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_cfg["lr"]),
            weight_decay=float(optimizer_cfg["weight_decay"]),
        )
        criterion = torch.nn.CTCLoss(
            blank=int(loss_cfg["blank"]),
            zero_infinity=bool(loss_cfg["zero_infinity"]),
        )

        trainer = RecognizerTrainer(
            config=recog_cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            aux_loss_weight=float(loss_cfg.get("aux_loss_weight", 0.0)),
        )
        result = trainer.train()
        result["train_dataset_size"] = len(train_dataset)
        result["val_dataset_size"] = len(val_dataset)
        result["device"] = str(device)
        return result


def run_train_recognizer(use_case: TrainRecognizerUseCasePort) -> dict[str, Any]:
    return use_case.run()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train recognizer end-to-end.")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_RELATIVE_PATH,
        help="Config YAML path relative to project root.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=DEFAULT_ENV_RELATIVE_PATH,
        help="Dotenv path relative to project root.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device, e.g. cpu/cuda/cuda:0.",
    )
    parser.add_argument("--val-ratio", type=float, default=None, help="Override validation ratio.")
    parser.add_argument("--split-seed", type=int, default=None, help="Override split seed.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    use_case = TrainRecognizerOrchestration(
        config_relative_path=args.config,
        env_relative_path=args.env,
        device=args.device,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
    )
    result = run_train_recognizer(use_case)
    print("Training finished.")
    print(f"best_checkpoint={result.get('best_checkpoint')}")
    print(f"last_checkpoint={result.get('last_checkpoint')}")
    print(f"best_val_loss={result.get('best_val_loss')}")
    print(f"best_val_cer={result.get('best_val_cer')}")


if __name__ == "__main__":
    main()
