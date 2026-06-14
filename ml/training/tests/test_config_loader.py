from __future__ import annotations

from pathlib import Path

import pytest

from src.domain.value_objet.config import DomainConfig, RecognizerConfig
from src.domain.value_objet.config_yolo import DEFAULT_YOLO_CONFIG_PATH, YoloConfig
from src.utils.config_loader import ConfigLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_RELATIVE_PATH = "config/.env"
RECOGNIZER_YAML = "config/training/recognizer_ctc.yaml"
YOLO_YAML = "config/training/yolo/config.yaml"


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


def test_load_domain_config_resolves_pwd_placeholder_relative_to_project_root(tmp_path: Path) -> None:
    config_path = tmp_path / "recognizer_pwd.yaml"
    config_path.write_text(
        "\n".join(
            [
                "recognizer:",
                "  architecture:",
                "    d_model: 32",
                "    num_heads: 4",
                "    d_c: 8",
                "    d_c1: 8",
                "    dim_rope: 8",
                "    n_shared_experts: 1",
                "    n_routed_experts: 2",
                "    top_k: 1",
                "    expert_hidden_dim: 16",
                "    attn_drop: 0.0",
                "    proj_drop: 0.0",
                "    moe_dropout: 0.0",
                "    config_training:",
                "      dataset:",
                "        data_root: ${pwd}/data",
                "        train_split: train",
                "        val_split: val",
                "        val_ratio: 0.2",
                "        split_seed: 42",
                "        num_patches: 10",
                "        patch_size: [16, 16]",
                "      vocab:",
                '        full_charset: "abc"',
                "      dataloader:",
                "        batch_size: 2",
                "        shuffle: true",
                "        num_workers: 0",
                "        pin_memory: false",
                "        drop_last: false",
                "        persistent_workers: false",
                "      optimizer:",
                "        lr: 0.001",
                "        weight_decay: 0.0",
                "      loss:",
                "        blank: 0",
                "        zero_infinity: true",
                "        aux_loss_weight: 0.0",
                "      loop:",
                "        epochs: 1",
                "        eval_every_n_epochs: 1",
                "        grad_clip_norm: 1.0",
                "      hooks:",
                "        fast_mode: false",
                "        tensorboard:",
                "          log_every_n_steps: 1",
                "          log_lr_every_n_steps: 0",
                "          flush_every_n_steps: 1",
                "          log_histograms_every_n_steps: 0",
                "          max_modules: 1",
                "          leaf_only: true",
                "        nan_guard:",
                "          enabled: false",
                "          check_inputs: false",
                "          check_outputs: false",
                "        grad_norm:",
                "          enabled: false",
                "          module_names: []",
                "      io:",
                "        save_dir: ${pwd}/${TEST_SAVE_DIR}",
                "        tensorboard_dir: ${pwd}/${TEST_TB_DIR}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_path = tmp_path / "test.env"
    env_path.write_text("TEST_SAVE_DIR=artifacts/checkpoints\nTEST_TB_DIR=artifacts/tensorboard\n", encoding="utf-8")

    cfg = ConfigLoader(
        yaml_relative_paths=[config_path.name],
        project_root=tmp_path,
        env_relative_path=env_path.name,
    ).load_domain_config()

    assert cfg.get_required("recognizer", "architecture", "config_training", "dataset", "data_root") == str(tmp_path / "data")
    assert cfg.get_required("recognizer", "architecture", "config_training", "io", "save_dir") == str(
        tmp_path / "artifacts" / "checkpoints"
    )
    assert cfg.get_required("recognizer", "architecture", "config_training", "io", "tensorboard_dir") == str(
        tmp_path / "artifacts" / "tensorboard"
    )


def test_load_domain_config_accepts_absolute_config_and_env_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "recognizer_absolute.yaml"
    env_path = tmp_path / "absolute.env"
    config_path.write_text(
        "\n".join(
            [
                "recognizer:",
                "  architecture:",
                "    d_model: 32",
                "    num_heads: 4",
                "    d_c: 8",
                "    d_c1: 8",
                "    dim_rope: 8",
                "    n_shared_experts: 1",
                "    n_routed_experts: 2",
                "    top_k: 1",
                "    expert_hidden_dim: 16",
                "    attn_drop: 0.0",
                "    proj_drop: 0.0",
                "    moe_dropout: 0.0",
                "    config_training:",
                "      dataset:",
                "        data_root: ${DATA_ROOT}",
                "        train_split: train",
                "        val_split: val",
                "        val_ratio: 0.2",
                "        split_seed: 42",
                "        num_patches: 10",
                "        patch_size: [16, 16]",
                "      vocab:",
                '        full_charset: "abc"',
                "      dataloader:",
                "        batch_size: 2",
                "        shuffle: true",
                "        num_workers: 0",
                "        pin_memory: false",
                "        drop_last: false",
                "        persistent_workers: false",
                "      optimizer:",
                "        lr: 0.001",
                "        weight_decay: 0.0",
                "      loss:",
                "        blank: 0",
                "        zero_infinity: true",
                "        aux_loss_weight: 0.0",
                "      loop:",
                "        epochs: 1",
                "        eval_every_n_epochs: 1",
                "        grad_clip_norm: 1.0",
                "      hooks:",
                "        fast_mode: false",
                "        tensorboard:",
                "          log_every_n_steps: 1",
                "          log_lr_every_n_steps: 0",
                "          flush_every_n_steps: 1",
                "          log_histograms_every_n_steps: 0",
                "          max_modules: 1",
                "          leaf_only: true",
                "        nan_guard:",
                "          enabled: false",
                "          check_inputs: false",
                "          check_outputs: false",
                "        grad_norm:",
                "          enabled: false",
                "          module_names: []",
                "      io:",
                "        save_dir: ${pwd}/artifacts/checkpoints",
                "        tensorboard_dir: ${pwd}/artifacts/tensorboard",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_path.write_text(f"DATA_ROOT={tmp_path / 'data'}\n", encoding="utf-8")

    cfg = ConfigLoader(
        yaml_relative_paths=[str(config_path)],
        project_root=Path("/unused/project/root"),
        env_relative_path=str(env_path),
    ).load_domain_config()

    assert cfg.get_required("recognizer", "architecture", "config_training", "dataset", "data_root") == str(tmp_path / "data")


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


def test_load_yolo_config_from_real_file_returns_dataclass() -> None:
    yolo_cfg = _loader().load_yolo_config(YOLO_YAML)

    assert isinstance(yolo_cfg, YoloConfig)
    assert yolo_cfg.model.weights
    assert yolo_cfg.training.epochs > 0
    assert yolo_cfg.data.config


def test_load_yolo_config_with_default_path() -> None:
    yolo_cfg = YoloConfig.from_yaml(DEFAULT_YOLO_CONFIG_PATH)

    assert isinstance(yolo_cfg, YoloConfig)
    assert yolo_cfg.save.project
