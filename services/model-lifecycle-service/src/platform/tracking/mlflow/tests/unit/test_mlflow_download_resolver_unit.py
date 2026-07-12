from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


SERVICE_ROOT = Path(__file__).resolve().parents[6]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.platform.tracking.mlflow.download_resolver import (
    MlflowArtifactDownloadResolver,
)


class MlflowArtifactDownloadResolverUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = MlflowArtifactDownloadResolver("http://mlflow.local:5000")
        self.mock_client = MagicMock()

    def test_download_model_artifact_uses_direct_download_when_url_is_http(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "best.pt"
            response = MagicMock()
            response.__enter__.return_value = response
            response.__exit__.return_value = False
            response.read.side_effect = [b"weights", b""]
            model_version = MagicMock()
            model_version.run_id = "run-123"
            self.mock_client.get_model_version.return_value = model_version
            run = MagicMock()
            run.info.artifact_uri = "mlflow-artifacts:/1/run-123/artifacts"
            self.mock_client.get_run.return_value = run

            with patch(
                "src.platform.tracking.mlflow.download_resolver.MlflowArtifactsRepository.resolve_uri",
                return_value="http://artifact.local/root",
            ) as mock_resolve, patch(
                "src.platform.tracking.mlflow.download_resolver.urllib.request.urlopen",
                return_value=response,
            ) as mock_urlopen:
                result = self.resolver.download_model_artifact(
                    client=self.mock_client,
                    model_name="detector",
                    version="7",
                    artifact_path="checkpoints/best.pt",
                    output_path=str(destination),
                )

            self.assertEqual(result, str(destination))
            self.assertEqual(destination.read_bytes(), b"weights")
            self.mock_client.get_model_version.assert_called_once_with("detector", "7")
            self.mock_client.get_run.assert_called_once_with("run-123")
            mock_resolve.assert_called_once_with(
                "mlflow-artifacts:/1/run-123/artifacts",
                "http://mlflow.local:5000",
            )
            mock_urlopen.assert_called_once()
            self.mock_client.download_artifacts.assert_not_called()

    def test_download_model_artifact_falls_back_to_mlflow_client_for_non_direct_uri(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "best.pt"
            downloaded = Path(tmpdir) / "downloaded.pt"
            downloaded.write_bytes(b"weights")

            model_version = MagicMock()
            model_version.run_id = "run-123"
            self.mock_client.get_model_version.return_value = model_version
            run = MagicMock()
            run.info.artifact_uri = "s3://bucket/root"
            self.mock_client.get_run.return_value = run
            self.mock_client.download_artifacts.return_value = str(downloaded)

            result = self.resolver.download_model_artifact(
                client=self.mock_client,
                model_name="detector",
                version="7",
                artifact_path="checkpoints/best.pt",
                output_path=str(destination),
            )

            self.assertEqual(result, str(destination))
            self.assertEqual(destination.read_bytes(), b"weights")
            self.mock_client.get_model_version.assert_called_once_with("detector", "7")
            self.mock_client.get_run.assert_called_once_with("run-123")
            self.mock_client.download_artifacts.assert_called_once_with(
                run_id="run-123",
                path="checkpoints/best.pt",
                dst_path=str(destination.parent),
            )


if __name__ == "__main__":
    unittest.main()
