import os
import sys
from pathlib import Path

from mlflow.tracking import MlflowClient

SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.applications.use_cases.mlflow.base.artifacts import BaseMlflowArtifactsUseCase
from src.infra.mlflow.mlflow_tracking import MlflowTracking
from src.utils.logger import Logger

class MlflowArtifactsRecognizer(BaseMlflowArtifactsUseCase):
    def __init__(
        self,
        mlflow_tracking: MlflowTracking,
        logger: Logger,
        path_env_ml_traning: str
    ):
        super().__init__(
            mlflow_tracking=mlflow_tracking,
            logger=logger,
            path_env_ml_traning=path_env_ml_traning,
            checkpoint_dir_env_key="RECOG_SAVE_DIR",
            model_name_env_key="MLFLOW_MODEL_NAME_RECOGNIZER",
            docker_image_env_key="RECOG_DOCKER_IMAGE",
            dataset_zip_path_env_key="RECOG_DATASET_ZIP_PATH",
            manifest_path_env_key="RECOG_MANIFEST_PATH",
            dataset_name_env_key="RECOG_DATASET_NAME",
            pipeline_name_env_key="PIPELINE_NAME_RECOGNIZER",
            pipeline_run_id_env_key="PIPELINE_RUN_ID_RECOGNIZER",
        )
        self._tensorboard_log_dir = self._env("RECOG_TENSORBOARD_DIR")
        self._recog_best_checkpoint_name = self._env("RECOG_BEST_CHECKPOINT_NAME", "best_cer.pt")
        self._recog_last_checkpoint_name = self._env("RECOG_LAST_CHECKPOINT_NAME", "last_checkpoint.pt")

    def _upload_checkpoint_artifacts(self, run_id: str, checkpoint_name: list[str], pwd: str) -> None:
        for name in checkpoint_name:
            self.log_artifact(
                run_id,
                self._resolve_path(pwd, self._checkpoint_dir, name),
                artifact_path="checkpoints",
            )
            checkpoint = self._load_checkpoint(self._resolve_path(pwd, self._checkpoint_dir), name)
            if name == self._recog_last_checkpoint_name:
                self._mlflow_tracking.log_params(run_id, checkpoint)
            if name == self._recog_best_checkpoint_name:
                self._mlflow_tracking.log_params(
                    run_id,
                    {
                        "epoch_best_val_cer": checkpoint.get("epoch"),
                    }
                )

    def _upload_training_artifacts(self, run_id: str, pwd: str) -> None:
        training_artifact_dir = self._resolve_path(pwd, self._tensorboard_log_dir)
        for name in os.listdir(training_artifact_dir):
            self.log_artifact(
                run_id,
                self._resolve_path(pwd, self._tensorboard_log_dir, name),
                artifact_path="tensorboard",
            )

    def _load_checkpoint(self, checkpoint_dir: str, file_checkpoint: str) -> None:
        import torch

        checkpoint_path = os.path.join(checkpoint_dir, file_checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return {
            "epoch": checkpoint.get("epoch"),
            "version": checkpoint.get("version"),
            "epoch_idx": checkpoint.get("epoch_idx"),
            "best_val_loss": checkpoint.get("best_val_loss"),
            "best_val_cer": checkpoint.get("best_val_cer"),
            "encoder_vocab": checkpoint.get("encoder_vocab"),
            "encoder_num_classes": checkpoint.get("encoder_num_classes"),
            "num_encoder_vocab": len(checkpoint.get("encoder_vocab")) if checkpoint.get("encoder_vocab") else 0,
            "num_parameters": sum(p.numel() for p in checkpoint.get("encoder_state_dict").values()) if checkpoint.get("encoder_state_dict") else 0,
        }

# def load_checkpoint(checkpoint_dir: str, file_checkpoint: str) -> None:
#     checkpoint_path = os.path.join(checkpoint_dir, file_checkpoint)
#     if not os.path.exists(checkpoint_path):
#         raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
#     checkpoint = torch.load(checkpoint_path, map_location="cpu")
#     return checkpoint

# if __name__ == "__main__":
#     checkpoint_dir = "/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus/data/checkpoint_recognizer"
#     file_checkpoint = "best_cer.pt"

#     mltracking = MlflowTracking("http://localhost:5000")

#     logger = Logger()
#     recognizer = MlflowArtifactsRecognizer(
#         mlflow_tracking=mltracking,
#         logger=logger,
#         path_env_ml_traning="/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus/ml/training/config/.env.example"
#     )

#     checkpoint_name = ["best_cer.pt", "last_checkpoint.pt"]

#     recognizer.execute(
#         git_commit="abc123",
#         checkpoint_name=checkpoint_name,
#         version="0.1"
#     )

# # if __name__ == "__main__":
# #     import subprocess
# #     git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
# #     print(f"Current git commit: {git_commit}")
