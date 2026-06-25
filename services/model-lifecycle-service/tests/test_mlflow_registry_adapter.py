from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS


SERVICE_ROOT = Path(__file__).resolve().parents[1]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.infra.mlflow.mlflow_registry import MlflowRegistry  # noqa: E402


class MlflowRegistryAdapterTest(unittest.TestCase):
    @patch("src.infra.mlflow.mlflow_registry.MlflowClient")
    @patch("src.infra.mlflow.mlflow_registry.mlflow.set_tracking_uri")
    def test_register_model_uses_create_model_version_for_runs_uri(
        self,
        _set_tracking_uri: Mock,
        client_cls: Mock,
    ) -> None:
        client = Mock()
        client_cls.return_value = client
        client.create_registered_model.side_effect = MlflowException(
            message="already exists",
            error_code=RESOURCE_ALREADY_EXISTS,
        )
        client.create_model_version.return_value = SimpleNamespace(version="9")

        registry = MlflowRegistry("http://localhost:5000")

        with patch("src.infra.mlflow.mlflow_registry.mlflow.register_model") as register_model_mock:
            version = registry.register_model(
                model_uri="runs:/run-123/checkpoints/best_cer.pt",
                registered_model_name="VitCTCDeepSeek",
            )

        self.assertEqual(version, "9")
        register_model_mock.assert_not_called()
        client.create_model_version.assert_called_once_with(
            name="VitCTCDeepSeek",
            source="runs:/run-123/checkpoints/best_cer.pt",
            run_id="run-123",
        )

    @patch("src.infra.mlflow.mlflow_registry.MlflowArtifactsRepository.resolve_uri")
    @patch("src.infra.mlflow.mlflow_registry.RunsArtifactRepository.get_underlying_uri")
    @patch("src.infra.mlflow.mlflow_registry.MlflowClient")
    @patch("src.infra.mlflow.mlflow_registry.mlflow.set_tracking_uri")
    def test_export_model_download_url_resolves_runs_uri_to_http_url(
        self,
        _set_tracking_uri: Mock,
        client_cls: Mock,
        get_underlying_uri: Mock,
        resolve_uri: Mock,
    ) -> None:
        client = Mock()
        client_cls.return_value = client
        client.get_model_version_download_uri.return_value = "runs:/run-123/checkpoints/best_cer.pt"
        get_underlying_uri.return_value = "mlflow-artifacts:/6/run-123/artifacts/checkpoints/best_cer.pt"
        resolve_uri.return_value = (
            "http://localhost:5000/api/2.0/mlflow-artifacts/artifacts/"
            "6/run-123/artifacts/checkpoints/best_cer.pt"
        )

        registry = MlflowRegistry("http://localhost:5000")

        download_url = registry.export_model_download_url("VitCTCDeepSeek", "6")

        self.assertEqual(
            download_url,
            "http://localhost:5000/api/2.0/mlflow-artifacts/artifacts/"
            "6/run-123/artifacts/checkpoints/best_cer.pt",
        )
        client.get_model_version_download_uri.assert_called_once_with("VitCTCDeepSeek", "6")
        get_underlying_uri.assert_called_once_with(
            "runs:/run-123/checkpoints/best_cer.pt",
            tracking_uri="http://localhost:5000",
        )
        resolve_uri.assert_called_once_with(
            "mlflow-artifacts:/6/run-123/artifacts/checkpoints/best_cer.pt",
            "http://localhost:5000",
        )


if __name__ == "__main__":
    unittest.main()
