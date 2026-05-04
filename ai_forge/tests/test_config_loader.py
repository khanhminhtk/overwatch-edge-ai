from __future__ import annotations

from pathlib import Path

import pytest

from src.domain.value_objet.config import DomainConfig, RecognizerConfig
from src.utils.config_loader import ConfigLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_RELATIVE_PATH = "config/.env"
RECOGNIZER_YAML = "config/training/recognizer_ctc.yaml"


def _loader() -> ConfigLoader:
    return ConfigLoader(
        yaml_relative_paths=[RECOGNIZER_YAML],
        project_root=PROJECT_ROOT,
        env_relative_path=ENV_RELATIVE_PATH,
    )


def test_load_domain_config_from_real_files() -> None:
    cfg = _loader().load_domain_config()

    assert isinstance(cfg, DomainConfig)
    assert cfg.get_required("recognizer", "architecture", "d_model") > 0
    assert cfg.get_required("recognizer", "architecture", "config_training", "dataset", "data_root")


def test_load_domain_config_raises_on_empty_paths() -> None:
    with pytest.raises(ValueError, match="yaml_relative_paths must not be empty"):
        ConfigLoader(yaml_relative_paths=[], project_root=PROJECT_ROOT, env_relative_path=ENV_RELATIVE_PATH)


def test_load_recognizer_from_real_files_returns_dataclass() -> None:
    rec_cfg = _loader().load_recognizer()

    assert isinstance(rec_cfg, RecognizerConfig)
    assert rec_cfg.d_model > 0
    assert rec_cfg.num_heads > 0
    assert "dataset" in rec_cfg.config_training
    assert "loop" in rec_cfg.config_training
    transforms_cfg = rec_cfg.config_training["dataset"]["transforms"]
    assert isinstance(transforms_cfg["enabled"], bool)
    assert isinstance(transforms_cfg["train"], dict)
    assert isinstance(transforms_cfg["val"], dict)


def test_load_recognizer_raises_for_wrong_arch_path() -> None:
    wrong_loader = ConfigLoader(
        yaml_relative_paths=[RECOGNIZER_YAML],
        project_root=PROJECT_ROOT,
        env_relative_path=ENV_RELATIVE_PATH,
        recognizer_arch_path=("recognizer", "not_exists"),
    )
    with pytest.raises(ValueError, match="Cannot find recognizer architecture config"):
        wrong_loader.load_recognizer()
