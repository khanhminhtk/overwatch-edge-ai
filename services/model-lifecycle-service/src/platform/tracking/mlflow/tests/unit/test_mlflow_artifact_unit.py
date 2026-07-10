from __future__ import annotations

import unittest
from unittest.mock import MagicMock, call, mock_open, patch

from mlflow.tracking import MlflowClient

from src.platform.tracking.mlflow.artifact import MlflowArtifact


class MlflowArtifactUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_client = MagicMock(spec=MlflowClient)
        self.artifact = MlflowArtifact(client=self.mock_client)

    def test_log_artifact(self) -> None:
        self.artifact.log_artifact("run-1", "/tmp/model.pkl", "models")
        self.mock_client.log_artifact.assert_called_once_with(
            "run-1", "/tmp/model.pkl", "models"
        )

    def test_log_artifact_default_path(self) -> None:
        self.artifact.log_artifact("run-1", "/tmp/model.pkl", None)
        self.mock_client.log_artifact.assert_called_once_with(
            "run-1", "/tmp/model.pkl", None
        )

    def test_log_artifacts(self) -> None:
        self.artifact.log_artifacts(
            "run-1", "/tmp/dir", "artifacts"
        )
        self.mock_client.log_artifacts.assert_called_once_with(
            "run-1", "/tmp/dir", "artifacts"
        )

    def test_log_text_writes_temp_file_and_logs(self) -> None:
        with patch(
            "src.platform.tracking.mlflow.artifact.tempfile.TemporaryDirectory"
        ) as mock_tmpdir:
            mock_tmpdir.return_value.__enter__.return_value = "/tmp/tmpdir"

            m = mock_open()
            with patch(
                "src.platform.tracking.mlflow.artifact.Path.open", m
            ):
                self.artifact.log_text("run-1", "hello", "subdir/note.txt")

        m.assert_called_once()
        handle = m()
        handle.write.assert_called_once_with("hello")
        self.mock_client.log_artifact.assert_called_once_with(
            "run-1",
            "/tmp/tmpdir/note.txt",
            "subdir",
        )

    def test_log_json(self) -> None:
        with patch.object(
            self.artifact, "log_text"
        ) as mock_log_text:
            self.artifact.log_json(
                "run-1", {"key": "val"}, "data/file.json"
            )
            mock_log_text.assert_called_once_with(
                "run-1",
                '{\n  "key": "val"\n}',
                "data/file.json",
            )

    def test_log_external_artifact_uri(self) -> None:
        with patch.object(
            self.artifact, "log_json"
        ) as mock_log_json:
            self.artifact.log_external_artifact_uri(
                "run-1", "weights", "s3://bucket/model"
            )
            self.mock_client.set_tag.assert_called_once_with(
                "run-1", "artifact_uri.weights", "s3://bucket/model"
            )
            mock_log_json.assert_called_once_with(
                "run-1",
                {"key": "weights", "uri": "s3://bucket/model"},
                "external_artifacts/weights.json",
            )


if __name__ == "__main__":
    unittest.main()
