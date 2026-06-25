from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.applications.use_cases.mlflow.recognizer.mlflow_model_registry import (  # noqa: E402
    MlflowModelRegistryRecognizer,
    resolve_run_id,
)


class MlflowModelRegistryRecognizerTest(unittest.TestCase):
    def test_execute_registers_model_sets_tags_alias_and_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "MLFLOW_MODEL_NAME_RECOGNIZER=VitCTCDeepSeek",
                        "MLFLOW_DEPLOYMENT_STAGE=staging",
                        "RECOG_BEST_CHECKPOINT_NAME=best_cer.pt",
                    ]
                ),
                encoding="utf-8",
            )

            registry = Mock()
            registry.log_pyfunc_model.return_value = "models:/m-recognizer-123"
            registry.register_model.return_value = "7"
            artifact = Mock()
            logger = Mock()
            recognizer = MlflowModelRegistryRecognizer(
                mlflow_model_registry=registry,
                mlflow_artifact=artifact,
                logger=logger,
                path_env_ml_traning=str(env_path),
            )

            with patch.object(
                MlflowModelRegistryRecognizer,
                "_load_checkpoint_metadata",
                return_value={
                    "epoch": 12,
                    "version": "0.9",
                    "epoch_idx": 11,
                    "best_val_loss": 0.14,
                    "best_val_cer": 0.02,
                    "encoder_num_classes": 3,
                    "encoder_vocab_size": 3,
                    "model_total_params": 10,
                },
            ):
                checkpoint_path = Path(temp_dir) / "best_cer.pt"
                checkpoint_path.write_bytes(b"checkpoint")
                env_path.write_text(
                    "\n".join(
                        [
                            f"RECOG_SAVE_DIR={temp_dir}",
                            "MLFLOW_MODEL_NAME_RECOGNIZER=VitCTCDeepSeek",
                            "MLFLOW_DEPLOYMENT_STAGE=staging",
                            "RECOG_BEST_CHECKPOINT_NAME=best_cer.pt",
                            "GPU_HOST_NAME=edge-host",
                            "GPU_NAME=RTX-4090",
                            "GPU_MEMORY_GB=24",
                            "CUDA_VERSION=12.4",
                            "RECOG_DATASET_NAME=recognizer_dataset",
                            "GPU_DRIVER_VERSION=550.54",
                            "FRAMEWORK_NAME=pytorch",
                            "PIPELINE_NAME_RECOGNIZER=recognizer_ctc",
                            "PIPELINE_RUN_ID_RECOGNIZER=pipeline-123",
                        ]
                    ),
                    encoding="utf-8",
                )
                recognizer = MlflowModelRegistryRecognizer(
                    mlflow_model_registry=registry,
                    mlflow_artifact=artifact,
                    logger=logger,
                    path_env_ml_traning=str(env_path),
                )
                with patch.object(
                    MlflowModelRegistryRecognizer,
                    "_log_pyfunc_model",
                    return_value="models:/m-recognizer-123",
                ) as log_pyfunc_model:
                    model_version = recognizer.execute(
                        run_id="run-123",
                        git_commit="abc123",
                        framework="pytorch",
                    )

        self.assertEqual(model_version, "7")
        log_pyfunc_model.assert_called_once_with(
            run_id="run-123",
            framework="pytorch",
            checkpoint_metadata={
                "epoch": 12,
                "version": "0.9",
                "epoch_idx": 11,
                "best_val_loss": 0.14,
                "best_val_cer": 0.02,
                "encoder_num_classes": 3,
                "encoder_vocab_size": 3,
                "model_total_params": 10,
            },
            pwd="",
        )
        registry.register_model.assert_called_once_with(
            model_uri="models:/m-recognizer-123",
            registered_model_name="VitCTCDeepSeek",
        )
        registry.set_model_version_tags.assert_called_once_with(
            registered_model_name="VitCTCDeepSeek",
            model_version="7",
            tags={
                "git_commit": "abc123",
                "framework": "pytorch",
                "model_type": "recognizer",
                "deployment_stage": "staging",
                "source_run_id": "run-123",
                "host_name": "edge-host",
                "gpu_name": "RTX-4090",
                "dataset_name": "recognizer_dataset",
                "pipeline_name": "recognizer_ctc",
                "pipeline_run_id": "pipeline-123",
                "checkpoint_epoch": "12",
                "checkpoint_version": "0.9",
                "best_val_loss": "0.14",
                "best_val_cer": "0.02",
                "encoder_num_classes": "3",
                "encoder_vocab_size": "3",
                "model_total_params": "10",
            },
        )
        registry.set_model_alias.assert_called_once_with(
            registered_model_name="VitCTCDeepSeek",
            alias="candidate",
            model_version="7",
        )
        registry.transition_model_stage.assert_called_once_with(
            registered_model_name="VitCTCDeepSeek",
            model_version="7",
            stage="staging",
        )
        artifact.log_json.assert_called_once_with(
            "run-123",
            {
                "registered_model_name": "VitCTCDeepSeek",
                "model_version": "7",
                "model_uri": "models:/m-recognizer-123",
                "source_checkpoint_uri": "runs:/run-123/checkpoints/best_cer.pt",
                "deployment_stage": "staging",
                "alias": "candidate",
                "git_commit": "abc123",
                "framework": "pytorch",
                "host_name": "edge-host",
                "gpu_name": "RTX-4090",
                "dataset_name": "recognizer_dataset",
                "pipeline_name": "recognizer_ctc",
                "pipeline_run_id": "pipeline-123",
                "checkpoint": {
                    "epoch": 12,
                    "version": "0.9",
                    "epoch_idx": 11,
                    "best_val_loss": 0.14,
                    "best_val_cer": 0.02,
                    "encoder_num_classes": 3,
                    "encoder_vocab_size": 3,
                    "model_total_params": 10,
                },
            },
            "production/registry_summary.json",
        )

    def test_execute_uses_alias_override_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "MLFLOW_MODEL_NAME_RECOGNIZER=VitCTCDeepSeek",
                        "MLFLOW_DEPLOYMENT_STAGE=production",
                        "MLFLOW_MODEL_ALIAS_RECOGNIZER=golden",
                        "RECOG_BEST_CHECKPOINT_NAME=best_cer.pt",
                    ]
                ),
                encoding="utf-8",
            )

            registry = Mock()
            registry.log_pyfunc_model.return_value = "models:/m-recognizer-456"
            registry.register_model.return_value = "11"
            artifact = Mock()
            recognizer = MlflowModelRegistryRecognizer(
                mlflow_model_registry=registry,
                mlflow_artifact=artifact,
                logger=Mock(),
                path_env_ml_traning=str(env_path),
            )

            with patch.object(
                MlflowModelRegistryRecognizer,
                "_load_checkpoint_metadata",
                return_value={
                    "epoch": 12,
                    "version": "0.9",
                    "epoch_idx": 11,
                    "best_val_loss": 0.14,
                    "best_val_cer": 0.02,
                    "encoder_num_classes": 3,
                    "encoder_vocab_size": 3,
                    "model_total_params": 10,
                },
            ):
                with patch.object(
                    MlflowModelRegistryRecognizer,
                    "_log_pyfunc_model",
                    return_value="models:/m-recognizer-456",
                ):
                    recognizer.execute(
                        run_id="run-456",
                        git_commit="def456",
                        framework="pytorch",
                    )

        registry.set_model_alias.assert_called_once_with(
            registered_model_name="VitCTCDeepSeek",
            alias="golden",
            model_version="11",
        )

    def test_execute_rejects_missing_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "MLFLOW_MODEL_NAME_RECOGNIZER=VitCTCDeepSeek",
                        "MLFLOW_DEPLOYMENT_STAGE=staging",
                        "RECOG_BEST_CHECKPOINT_NAME=best_cer.pt",
                    ]
                ),
                encoding="utf-8",
            )

            recognizer = MlflowModelRegistryRecognizer(
                mlflow_model_registry=Mock(),
                mlflow_artifact=Mock(),
                logger=Mock(),
                path_env_ml_traning=str(env_path),
            )

            with self.assertRaisesRegex(ValueError, "run_id"):
                recognizer.execute(
                    run_id="",
                    git_commit="abc123",
                )

    def test_export_model_download_url_delegates_to_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "MLFLOW_MODEL_NAME_RECOGNIZER=VitCTCDeepSeek",
                        "MLFLOW_DEPLOYMENT_STAGE=staging",
                        "RECOG_BEST_CHECKPOINT_NAME=best_cer.pt",
                    ]
                ),
                encoding="utf-8",
            )

            registry = Mock()
            registry.export_model_download_url.return_value = (
                "http://localhost:5000/api/2.0/mlflow-artifacts/artifacts/"
                "6/run-123/artifacts/checkpoints/best_cer.pt"
            )
            recognizer = MlflowModelRegistryRecognizer(
                mlflow_model_registry=registry,
                mlflow_artifact=Mock(),
                logger=Mock(),
                path_env_ml_traning=str(env_path),
            )

            download_url = recognizer.export_model_download_url("VitCTCDeepSeek", "6")

        self.assertEqual(
            download_url,
            "http://localhost:5000/api/2.0/mlflow-artifacts/artifacts/"
            "6/run-123/artifacts/checkpoints/best_cer.pt",
        )
        registry.export_model_download_url.assert_called_once_with("VitCTCDeepSeek", "6")

    def test_resolve_run_id_strips_value(self) -> None:
        self.assertEqual(resolve_run_id("  run-123  "), "run-123")

    def test_resolve_run_id_rejects_missing_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "MLFLOW_RUN_ID_RECOGNIZER_TEST"):
            resolve_run_id("")

if __name__ == "__main__":
    unittest.main()
