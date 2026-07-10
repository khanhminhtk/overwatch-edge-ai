from __future__ import annotations

from pydantic.dataclasses import dataclass

from .hardware_info import HardwareInfo


@dataclass(frozen=True, kw_only=True, slots=True)
class DetectionConfig:
    deployment_stage: str = "training"
    framework: str = "unknown_framework"
    hardware: HardwareInfo = HardwareInfo()
    model_type: str = "detector"
    task_name: str = "detection"
    model_name: str = "yolo_detector"
    docker_image: str = ""
    dataset_zip_path: str = ""
    manifest_path: str = ""
    results_csv_path: str = ""
    checkpoint_dir: str = ""
    best_checkpoint_name: str = "best.pt"
    last_checkpoint_name: str = "last.pt"
    dataset_name: str = "unknown_dataset"
    pipeline_name: str = "unknown_pipeline"
    pipeline_run_id: str = "unknown_pipeline_run_id"
    model_alias: str = ""
    training_config_path: str = "ml/training/config/training/yolo/config.yaml"
    training_env_path: str = ""
