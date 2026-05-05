from pathlib import Path

import torch
from ultralytics import YOLO

from src.application.use_cases.recognizer.load_model_vit_ctc import load_model_vit_ctc
from src.application.use_cases.recognizer.recognizerCTCModel import RecognizerCTCModel
from src.application.use_cases.orchestration.train_recognizer_orchestration import resolve_vocab
from src.infra.modeling.detection.yolo import YoloTrainer 
from src.infra.data.dataloaders.ctc_collate import CTCLabelEncoder
from src.utils.config_loader import ConfigLoader

def _extract_model_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("load_recognizer_pt: checkpoint must be a dict-like object.")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if not isinstance(state_dict, dict):
            raise ValueError("load_recognizer_pt: checkpoint['model_state_dict'] must be a state_dict mapping.")
        return state_dict

    return checkpoint


def load_yolo_pt(
    config_loader: ConfigLoader,
    path: str,
    yolo_config_relative_path: str = "config/training/yolo/config.yaml",
) -> YoloTrainer:
    config_yolo = config_loader.load_yolo_config(config_relative_path=yolo_config_relative_path)
    trainer = YoloTrainer(config=config_yolo)
    trainer.load_model_best_weights(path)
    return trainer


def load_recognizer_pt(config_loader: ConfigLoader, path: str) -> RecognizerCTCModel:
    recog_cfg = config_loader.load_recognizer()
    config_training = dict(recog_cfg.config_training)
    dataset_cfg = config_training["dataset"]
    data_root = Path(dataset_cfg["data_root"])
    train_split = str(dataset_cfg["train_split"])

    vocab = resolve_vocab(config_training=config_training, data_root=data_root, train_split=train_split)
    ctc_encoder = CTCLabelEncoder(vocab=vocab)
    recognizer_model = load_model_vit_ctc(config=config_loader, device="cpu")
    model = RecognizerCTCModel(model=recognizer_model, vocab_size=ctc_encoder.num_classes).to("cpu")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = _extract_model_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_pt(
    config_loader: ConfigLoader,
    path: str,
    yolo_config_relative_path: str = "config/training/yolo/config.yaml",
    mode_ultralytics: bool = True,
) -> RecognizerCTCModel | YoloTrainer:
    if mode_ultralytics:
        return load_yolo_pt(
            config_loader=config_loader,
            path=path,
            yolo_config_relative_path=yolo_config_relative_path,
        )
    return load_recognizer_pt(config_loader=config_loader, path=path)
