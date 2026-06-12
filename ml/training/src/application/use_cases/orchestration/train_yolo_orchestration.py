from src.utils.config_loader import ConfigLoader
from src.domain.value_objet.config_yolo import YoloConfig
from src.infra.modeling.detection.yolo import YoloTrainer

DEFAULT_YOLO_CONFIG_RELATIVE_PATH = "config/training/yolo/config.yaml"
DEFAULT_ENV_RELATIVE_PATH = "config/.env"


def _build_yolo_config(config_loader: ConfigLoader) -> YoloConfig:
    return config_loader.load_yolo_config(DEFAULT_YOLO_CONFIG_RELATIVE_PATH)


def train_yolo(config_loader: ConfigLoader) -> str:
    yolo_config = _build_yolo_config(config_loader)
    trainer = YoloTrainer(config=yolo_config)
    trainer.load(finetune=True)
    trainer.train()
    return trainer.best_weight_path

if __name__ == "__main__":
    config_loader = ConfigLoader(
        yaml_relative_paths=[DEFAULT_YOLO_CONFIG_RELATIVE_PATH],
        env_relative_path=DEFAULT_ENV_RELATIVE_PATH,
    )
    best_weight_path = train_yolo(config_loader)
    print(f"Training finished. best_checkpoint={best_weight_path}")
