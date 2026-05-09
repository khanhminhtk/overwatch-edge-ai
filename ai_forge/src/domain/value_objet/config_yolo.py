from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml


DEFAULT_YOLO_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "training" / "yolo" / "config.yaml"
)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    device: str
    deterministic: bool
    output_dir: str


@dataclass(frozen=True)
class ModelConfig:
    weights: str
    task: str
    pretrained: bool


@dataclass(frozen=True)
class DataConfig:
    config: str
    cache: bool


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    imgsz: int
    batch: int
    workers: int
    patience: int
    amp: bool
    val: bool
    resume: bool
    close_mosaic: int


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr0: float
    lrf: float
    momentum: float
    weight_decay: float
    warmup_epochs: float
    warmup_momentum: float
    warmup_bias_lr: float
    cos_lr: bool


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool
    hsv_h: float
    hsv_s: float
    hsv_v: float
    degrees: float
    translate: float
    scale: float
    shear: float
    perspective: float
    flipud: float
    fliplr: float
    mosaic: float
    mixup: float
    copy_paste: float


@dataclass(frozen=True)
class SaveConfig:
    project: str
    run_name: str
    exist_ok: bool
    save: bool
    save_period: int
    plots: bool


@dataclass(frozen=True)
class EvaluationConfig:
    conf: float
    iou: float
    max_det: int
    half: bool


@dataclass(frozen=True)
class TrackingConfig:
    tensorboard: bool
    wandb: bool
    mlflow: bool

@dataclass(frozen=True)
class YoloConfig:
    experiment: ExperimentConfig
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    augmentation: AugmentationConfig
    save: SaveConfig
    evaluation: EvaluationConfig
    tracking: TrackingConfig

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "YoloConfig":
        return cls(
            experiment=ExperimentConfig(**raw["experiment"]),
            model=ModelConfig(**raw["model"]),
            data=DataConfig(**raw["data"]),
            training=TrainingConfig(**raw["training"]),
            optimizer=OptimizerConfig(**raw["optimizer"]),
            augmentation=AugmentationConfig(**raw["augmentation"]),
            save=SaveConfig(**raw["save"]),
            evaluation=EvaluationConfig(**raw["evaluation"]),
            tracking=TrackingConfig(**raw["tracking"]),
        )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "YoloConfig":
        config_path = Path(config_path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid YOLO config format: {config_path}")
        return cls.from_dict(raw)


def load_yolo_config(config_path: str | Path = DEFAULT_YOLO_CONFIG_PATH) -> YoloConfig:
    return YoloConfig.from_yaml(config_path)


class YoloArgsAdapter:
    @staticmethod
    def _resolve_project_path(project: str) -> str:
        return str(Path(project).expanduser().resolve())

    @staticmethod
    def _resolve_device(device: str) -> str:
        normalized = device.strip().lower()
        if normalized != "auto":
            return device

        try:
            import torch
        except Exception:
            return "cpu"

        return "0" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def to_train_kwargs(config: YoloConfig, **overrides: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "data": config.data.config,
            "epochs": config.training.epochs,
            "imgsz": config.training.imgsz,
            "batch": config.training.batch,
            "workers": config.training.workers,
            "patience": config.training.patience,
            "amp": config.training.amp,
            "val": config.training.val,
            "resume": config.training.resume,
            "close_mosaic": config.training.close_mosaic,
            "optimizer": config.optimizer.name,
            "lr0": config.optimizer.lr0,
            "lrf": config.optimizer.lrf,
            "momentum": config.optimizer.momentum,
            "weight_decay": config.optimizer.weight_decay,
            "warmup_epochs": config.optimizer.warmup_epochs,
            "warmup_momentum": config.optimizer.warmup_momentum,
            "warmup_bias_lr": config.optimizer.warmup_bias_lr,
            "cos_lr": config.optimizer.cos_lr,
            "project": YoloArgsAdapter._resolve_project_path(config.save.project),
            "name": config.save.run_name,
            "exist_ok": config.save.exist_ok,
            "save": config.save.save,
            "save_period": config.save.save_period,
            "plots": config.save.plots,
            "cache": config.data.cache,
            "device": YoloArgsAdapter._resolve_device(config.experiment.device),
            "seed": config.experiment.seed,
            "deterministic": config.experiment.deterministic,
        }
        if config.augmentation.enabled:
            kwargs.update(
                {
                    "hsv_h": config.augmentation.hsv_h,
                    "hsv_s": config.augmentation.hsv_s,
                    "hsv_v": config.augmentation.hsv_v,
                    "degrees": config.augmentation.degrees,
                    "translate": config.augmentation.translate,
                    "scale": config.augmentation.scale,
                    "shear": config.augmentation.shear,
                    "perspective": config.augmentation.perspective,
                    "flipud": config.augmentation.flipud,
                    "fliplr": config.augmentation.fliplr,
                    "mosaic": config.augmentation.mosaic,
                    "mixup": config.augmentation.mixup,
                    "copy_paste": config.augmentation.copy_paste,
                }
            )
        kwargs.update(overrides)
        return kwargs

    @staticmethod
    def to_val_kwargs(config: YoloConfig, **overrides: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "data": config.data.config,
            "project": YoloArgsAdapter._resolve_project_path(config.save.project),
            "name": config.save.run_name,
            "exist_ok": config.save.exist_ok,
            "imgsz": config.training.imgsz,
            "batch": config.training.batch,
            "workers": config.training.workers,
            "device": YoloArgsAdapter._resolve_device(config.experiment.device),
            "conf": config.evaluation.conf,
            "iou": config.evaluation.iou,
            "max_det": config.evaluation.max_det,
            "half": config.evaluation.half,
            "plots": config.save.plots,
        }
        kwargs.update(overrides)
        return kwargs
