from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlflow.exceptions import MlflowException


SERVICE_ROOT = Path(__file__).resolve().parents[6]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.platform.tracking.mlflow.registry import MlflowRegistry


class MlflowRegistryUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher_client_mlflow = patch(
            "src.platform.tracking.mlflow.client.mlflow"
        )
        self.mock_mlflow = patcher_client_mlflow.start()
        self.addCleanup(patcher_client_mlflow.stop)

        patcher_registry_mlflow = patch(
            "src.platform.tracking.mlflow.registry.mlflow",
            self.mock_mlflow,
        )
        patcher_registry_mlflow.start()
        self.addCleanup(patcher_registry_mlflow.stop)

        self.mock_client = MagicMock()
        self.mock_client.tracking_uri = "http://mock-tracking:5000"

        self.registry = MlflowRegistry(client=self.mock_client)

    def test_init_sets_tracking_uri_globally(self) -> None:
        self.mock_mlflow.set_tracking_uri.assert_called_once_with(
            "http://mock-tracking:5000"
        )

    def test_init_stores_tracking_uri(self) -> None:
        self.assertEqual(
            self.registry._tracking_uri, "http://mock-tracking:5000"
        )

    def test_init_creates_default_download_resolver(self) -> None:
        self.assertIsNotNone(self.registry._download_resolver)

    def test_export_model_download_url_uses_default_resolver(self) -> None:
        self.mock_client.get_model_version_download_uri.return_value = (
            "https://artifact.example/model.bin"
        )

        result = self.registry.export_model_download_url(
            model_name="my-model",
            version="7",
            artifact_path="weights/best.pt",
        )

        self.assertEqual(result, "https://artifact.example/model.bin")
        self.mock_client.get_model_version_download_uri.assert_called_once_with(
            "my-model",
            "7",
        )

    def test_download_model_artifact_uses_default_resolver(self) -> None:
        self.registry._download_resolver = MagicMock()
        self.registry._download_resolver.download_model_artifact.return_value = "/tmp/best.pt"

        result = self.registry.download_model_artifact(
            model_name="my-model",
            version="7",
            artifact_path="weights/best.pt",
            output_path="/tmp/best.pt",
        )

        self.assertEqual(result, "/tmp/best.pt")
        self.registry._download_resolver.download_model_artifact.assert_called_once_with(
            client=self.mock_client,
            model_name="my-model",
            version="7",
            artifact_path="weights/best.pt",
            output_path="/tmp/best.pt",
        )

    def test_log_pyfunc_model(self) -> None:
        mock_model_info = MagicMock()
        mock_model_info.model_uri = "runs:/run-123/model"
        self.mock_mlflow.pyfunc.log_model.return_value = mock_model_info

        python_model = MagicMock()
        result = self.registry.log_pyfunc_model(
            run_id="run-123",
            model_name="my_model",
            python_model=python_model,
            input_example={"x": 1},
            signature="signature",
            registered_model_name="reg-model",
            tags={"env": "prod"},
            params={"lr": 0.01},
        )

        self.assertEqual(result, "runs:/run-123/model")
        self.mock_mlflow.start_run.assert_called_once_with(run_id="run-123")
        self.mock_mlflow.pyfunc.log_model.assert_called_once_with(
            name="my_model",
            python_model=python_model,
            input_example={"x": 1},
            signature="signature",
            registered_model_name="reg-model",
            tags={"env": "prod"},
            params={"lr": 0.01},
            model_type="pyfunc",
        )

    def test_log_pyfunc_model_reuses_matching_active_run(self) -> None:
        mock_model_info = MagicMock()
        mock_model_info.model_uri = "runs:/run-123/model"
        self.mock_mlflow.pyfunc.log_model.return_value = mock_model_info
        active_run = MagicMock()
        active_run.info.run_id = "run-123"
        self.mock_mlflow.active_run.return_value = active_run

        result = self.registry.log_pyfunc_model(
            run_id="run-123",
            model_name="my_model",
            python_model=MagicMock(),
            input_example={"x": 1},
            signature="signature",
            registered_model_name="reg-model",
            tags={"env": "prod"},
            params={"lr": 0.01},
        )

        self.assertEqual(result, "runs:/run-123/model")
        self.mock_mlflow.start_run.assert_not_called()
        self.mock_mlflow.pyfunc.log_model.assert_called_once()

    def test_register_model_with_runs_uri(self) -> None:
        result = self.registry.register_model(
            "runs:/abc123/model", "my-model", "abc123"
        )
        self.mock_client.create_registered_model.assert_called_once_with(
            "my-model"
        )
        self.mock_client.create_model_version.assert_called_once_with(
            name="my-model",
            source="runs:/abc123/model",
            run_id="abc123",
        )

    def test_register_model_handles_already_exists(self) -> None:
        exc = MlflowException("already exists")
        exc.error_code = "RESOURCE_ALREADY_EXISTS"
        self.mock_client.create_registered_model.side_effect = exc

        result = self.registry.register_model(
            "runs:/abc123/model", "my-model", "abc123"
        )
        self.mock_client.create_model_version.assert_called_once()

    def test_register_model_raises_on_unexpected_error(self) -> None:
        exc = MlflowException("internal error")
        exc.error_code = "INTERNAL_ERROR"
        self.mock_client.create_registered_model.side_effect = exc

        with self.assertRaises(MlflowException):
            self.registry.register_model(
                "runs:/abc123/model", "my-model", "abc123"
            )

    def test_register_model_with_non_runs_uri(self) -> None:
        mock_version = MagicMock()
        mock_version.version = "5"
        self.mock_mlflow.register_model.return_value = mock_version

        result = self.registry.register_model(
            "models:/source/1", "my-model", "run-123"
        )

        self.assertEqual(result, "5")
        self.mock_mlflow.register_model.assert_called_once_with(
            model_uri="models:/source/1", name="my-model"
        )

    def test_set_model_version_tags(self) -> None:
        self.registry.set_model_version_tags(
            "model", "2", {"env": "prod"}
        )
        self.mock_client.set_model_version_tag.assert_called_once_with(
            "model", "2", "env", "prod"
        )

    def test_set_model_alias(self) -> None:
        self.registry.set_model_alias("model", "champion", "3")
        self.mock_client.set_registered_model_alias.assert_called_once_with(
            "model", "champion", "3"
        )

    def test_transition_model_stage(self) -> None:
        self.registry.transition_model_stage("model", "2", "Production")
        self.mock_client.transition_model_version_stage.assert_called_once_with(
            "model", "2", "Production"
        )

    def test_client_property(self) -> None:
        self.assertIs(self.registry.client, self.mock_client)


if __name__ == "__main__":
    unittest.main()
