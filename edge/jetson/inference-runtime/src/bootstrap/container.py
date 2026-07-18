from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.platform.config import ConfigLoader
from src.platform.capture import LowQualityCaptureConfig
from src.modules.inference import DetectorSettings, RecognizerSettings
from src.platform.triton import TritonConfig
from src.platform.video import VideoSourceConfig


SERVICE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = SERVICE_ROOT / "config"
ENV_FILE = CONFIG_DIR / ".env"
CONFIG_FILE = CONFIG_DIR / "inference_runtime_config.yaml"


@dataclass(frozen=True)
class InferenceRuntimeContainer:
    detector_config: TritonConfig
    recognizer_config: TritonConfig
    detector_settings: DetectorSettings
    recognizer_settings: RecognizerSettings
    video_config: VideoSourceConfig
    capture_config: LowQualityCaptureConfig


@dataclass(frozen=True)
class _TritonServerConfig:
    url: str


@dataclass(frozen=True)
class _TritonModelConfig:
    model_name: str
    model_version: str = "1"


def validate_runtime_inputs() -> None:
    missing_paths = [path for path in (CONFIG_FILE, ENV_FILE) if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Required config files not found: {missing}")


def load_triton_config(model_section: str) -> TritonConfig:
    server = ConfigLoader.load(
        _TritonServerConfig,
        yaml_files=[CONFIG_FILE],
        env_files=[ENV_FILE],
        section="triton",
    )
    model = ConfigLoader.load(
        _TritonModelConfig,
        yaml_files=[CONFIG_FILE],
        env_files=[ENV_FILE],
        section=f"triton.{model_section}",
    )
    return TritonConfig(
        url=server.url,
        model_name=model.model_name,
        model_version=model.model_version,
    )


def load_video_config() -> VideoSourceConfig:
    return ConfigLoader.load(
        VideoSourceConfig,
        yaml_files=[CONFIG_FILE],
        env_files=[ENV_FILE],
        section="video",
    )


def load_detector_settings() -> DetectorSettings:
    return ConfigLoader.load(
        DetectorSettings,
        yaml_files=[CONFIG_FILE],
        env_files=[ENV_FILE],
        section="inference.detector",
    )


def load_recognizer_settings() -> RecognizerSettings:
    return ConfigLoader.load(
        RecognizerSettings,
        yaml_files=[CONFIG_FILE],
        env_files=[ENV_FILE],
        section="inference.recognizer",
    )


def load_capture_config() -> LowQualityCaptureConfig:
    return ConfigLoader.load(
        LowQualityCaptureConfig,
        yaml_files=[CONFIG_FILE],
        env_files=[ENV_FILE],
        section="capture",
    )


def load_inference_config() -> InferenceRuntimeContainer:
    return InferenceRuntimeContainer(
        detector_config=load_triton_config("detector"),
        recognizer_config=load_triton_config("recognizer"),
        detector_settings=load_detector_settings(),
        recognizer_settings=load_recognizer_settings(),
        video_config=load_video_config(),
        capture_config=load_capture_config(),
    )
