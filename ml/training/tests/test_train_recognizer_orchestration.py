from __future__ import annotations

from pathlib import Path

import pytest

from tests.recognizer_test_helpers import TinyVisionModel, build_loader, write_training_yaml
from src.application.use_cases.orchestration.train_recognizer_orchestration import (
    TrainRecognizerOrchestration,
    TrainRecognizerUseCasePort,
    run_train_recognizer,
)


class _DummyUseCase(TrainRecognizerUseCasePort):
    def run(self) -> dict[str, object]:
        return {"ok": True, "value": 123}


def test_run_train_recognizer_uses_interface() -> None:
    result = run_train_recognizer(_DummyUseCase())
    assert result["ok"] is True
    assert result["value"] == 123


def test_train_orchestration_end_to_end_with_real_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("tensorboard")

    loader, _ = build_loader(tmp_path=tmp_path)
    cfg = loader.load_recognizer()

    # Keep test fast: run one epoch only.
    cfg_training = dict(cfg.config_training)
    cfg_training["loop"] = dict(cfg_training["loop"])
    cfg_training["loop"]["epochs"] = 1

    data_root = Path(cfg_training["dataset"]["data_root"])
    config_path = tmp_path / "recognizer_ctc_orch_test.yaml"
    write_training_yaml(config_path=config_path, data_root=data_root)

    # Force tiny model to avoid downloading backbone weights during tests.
    monkeypatch.setattr(
        "src.application.use_cases.orchestration.train_recognizer_orchestration.load_model_vit_ctc",
        lambda config, device: TinyVisionModel(d_model=32, device=device),
    )

    use_case = TrainRecognizerOrchestration(
        config_relative_path=config_path.name,
        env_relative_path="not_used.env",
        project_root=tmp_path,
        device="cpu",
        val_ratio=0.2,
        split_seed=42,
    )

    result = use_case.run()

    assert "best_checkpoint" in result
    assert "last_checkpoint" in result
    assert Path(str(result["last_checkpoint"])).is_file()
    assert isinstance(result["train_dataset_size"], int)
    assert isinstance(result["val_dataset_size"], int)
    assert result["device"] == "cpu"


def test_train_orchestration_cli_help_runs() -> None:
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "src.application.use_cases.orchestration.train_recognizer_orchestration",
        "--help",
    ]
    completed = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, check=False)
    assert completed.returncode == 0
    assert "Train recognizer end-to-end" in completed.stdout


def test_resolve_vocab_mode_auto_uses_observed_charset(tmp_path: Path) -> None:
    from tests.recognizer_test_helpers import create_real_dataset
    from src.application.use_cases.orchestration.train_recognizer_orchestration import _resolve_vocab

    data_root, words = create_real_dataset(root_dir=tmp_path, num_samples=12)
    observed = sorted({ch for w in words for ch in w})
    resolved = _resolve_vocab(
        config_training={"vocab": {"mode": "auto", "full_charset": "xyzXYZ"}},
        data_root=data_root,
        train_split="train",
    )
    assert sorted(resolved) == observed
