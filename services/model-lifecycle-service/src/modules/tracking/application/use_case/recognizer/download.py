from __future__ import annotations

import argparse
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[6]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from mlflow import MlflowClient

from src.modules.tracking.application.use_case.tracking_common import log_exceptions
from src.modules.tracking.domain.entity_objects import RecognizerRun
from src.modules.tracking.domain.value_objects import RecognizerConfig
from src.modules.tracking.domain.value_objects import TrackingArtifactDownloadSpec
from src.platform.config import ConfigLoader
from src.platform.logger import Logger
from src.platform.tracking.mlflow import MlflowRegistry, ModelRegistry
from src.platform.tracking.mlflow.config import MlflowConfig


class RecognizerMlflowModelDownload:
    def __init__(
        self,
        run: RecognizerRun,
        registry: ModelRegistry,
        logger: Logger,
    ) -> None:
        self._run = run
        self._registry = registry
        self._logger = logger

    @log_exceptions("[RECOGNIZER_DOWNLOAD_CHAMPION_ERROR]")
    def download_champion_url(self, artifact_path: str) -> str:
        version = self._resolve_alias_version("champion")
        spec = TrackingArtifactDownloadSpec.for_recognizer(artifact_path)
        return self._registry.export_model_download_url(
            model_name=self._run.config.model_name,
            version=version,
            artifact_path=spec.artifact_path,
        )

    @log_exceptions("[RECOGNIZER_DOWNLOAD_VERSION_ERROR]")
    def download_version_url(self, version: str, artifact_path: str) -> str:
        spec = TrackingArtifactDownloadSpec.for_recognizer(artifact_path)
        return self._registry.export_model_download_url(
            model_name=self._run.config.model_name,
            version=version,
            artifact_path=spec.artifact_path,
        )

    @log_exceptions("[RECOGNIZER_DOWNLOAD_CHAMPION_TO_FILE_ERROR]")
    def download_champion_to_file(self, artifact_path: str, output_path: str) -> str:
        version = self._resolve_alias_version("champion")
        return self.download_version_to_file(
            version=version,
            artifact_path=artifact_path,
            output_path=output_path,
        )

    @log_exceptions("[RECOGNIZER_DOWNLOAD_VERSION_TO_FILE_ERROR]")
    def download_version_to_file(
        self,
        version: str,
        artifact_path: str,
        output_path: str,
    ) -> str:
        spec = TrackingArtifactDownloadSpec.for_recognizer(artifact_path)
        return self._registry.download_model_artifact(
            model_name=self._run.config.model_name,
            version=version,
            artifact_path=spec.artifact_path,
            output_path=output_path,
        )

    @log_exceptions("[RECOGNIZER_DOWNLOAD_RESOLVE_ALIAS_ERROR]")
    def _resolve_alias_version(self, alias: str) -> str:
        model_version = self._registry.get_model_version_by_alias(
            self._run.config.model_name,
            alias,
        )
        if model_version is None:
            raise ValueError(f"Alias not found: {alias}")
        return str(getattr(model_version, "version"))


# def build_use_case(repo_root: str | Path) -> RecognizerMlflowModelDownload:
#     resolved_repo_root = Path(repo_root).resolve()
#     recognizer_config = ConfigLoader.load(
#         RecognizerConfig,
#         yaml_files=[str(resolved_repo_root / "services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml")],
#         env_files=[str(resolved_repo_root / "services/model-lifecycle-service/config/.env")],
#         section={
#             "mlflow": None,
#             "mlflow.recognizer": None,
#         },
#     )
#     mlflow_config = ConfigLoader.load(
#         MlflowConfig,
#         yaml_files=[str(resolved_repo_root / "services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml")],
#         env_files=[str(resolved_repo_root / "services/model-lifecycle-service/config/.env")],
#         section="mlflow",
#     )
#     run = RecognizerRun(
#         experiment_name="test_experiment",
#         run_name="test_run",
#         config=recognizer_config,
#     )
#     client = MlflowClient(tracking_uri=mlflow_config.tracking_uri)
#     registry = MlflowRegistry(client=client, tracking_url=mlflow_config.tracking_uri)
#     return RecognizerMlflowModelDownload(
#         run=run,
#         registry=registry,
#         logger=Logger("RecognizerMlflowModelDownload"),
#     )

# def main(
#     *,
#     alias: str | None = "champion",
#     version: str | None = None,
#     artifact_path: str = "checkpoints/best_cer.pt",
#     output_path: str | None = None,
# ) -> str:
#     use_case = build_use_case(REPO_ROOT)
#     if version is not None:
#         if output_path is not None:
#             return use_case.download_version_to_file(
#                 version=version,
#                 artifact_path=artifact_path,
#                 output_path=output_path,
#             )
#         download_url = use_case.download_version_url(
#             version=version,
#             artifact_path=artifact_path,
#         )
#     else:
#         if alias != "champion":
#             raise ValueError("Only alias='champion' is supported in recognizer main()")
#         if output_path is not None:
#             return use_case.download_champion_to_file(
#                 artifact_path=artifact_path,
#                 output_path=output_path,
#             )
#         download_url = use_case.download_champion_url(artifact_path)

#     return download_url


# def _parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Download recognizer artifacts from MLflow.")
#     parser.add_argument("--alias", default="champion")
#     parser.add_argument("--version")
#     parser.add_argument("--artifact-path", default="checkpoints/best_cer.pt")
#     parser.add_argument("--output-path")
#     return parser.parse_args()

REPO_ROOT = SERVICE_ROOT.parent.parent
CONFIG_YAML = REPO_ROOT / "services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml"
CONFIG_ENV = REPO_ROOT / "services/model-lifecycle-service/config/.env"

# if __name__ == "__main__":
#     args = _parse_args()
#     print(
#         main(
#             alias=args.alias,
#             version=args.version,
#             artifact_path=args.artifact_path,
#             output_path=args.output_path,
#         )
#     )
