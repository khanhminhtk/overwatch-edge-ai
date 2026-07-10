from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.tracking.application.use_case.detection.download import (  # noqa: E402
    DetectionMlflowModelDownload,
    main,
)
from src.modules.tracking.domain.entity_objects import DetectionRun  # noqa: E402
from src.modules.tracking.domain.value_objects import DetectionConfig  # noqa: E402


class DetectionMlflowModelDownloadUnitTest(unittest.TestCase):
    def test_download_champion_url_resolves_alias_then_exports_artifact_url(self) -> None:
        registry = Mock()
        registry.get_model_version_by_alias.return_value = SimpleNamespace(version="12")
        registry.export_model_download_url.return_value = "http://mlflow.local/model/12/checkpoints/best.pt"
        use_case = DetectionMlflowModelDownload(
            run=DetectionRun(
                experiment_name="exp",
                run_name="run",
                config=DetectionConfig(model_name="yolo_detector"),
            ),
            registry=registry,
            logger=Mock(),
        )

        result = use_case.download_champion_url("checkpoints/best.pt")

        self.assertEqual(result, "http://mlflow.local/model/12/checkpoints/best.pt")
        registry.get_model_version_by_alias.assert_called_once_with("yolo_detector", "champion")
        registry.export_model_download_url.assert_called_once_with(
            model_name="yolo_detector",
            version="12",
            artifact_path="checkpoints/best.pt",
        )

    def test_download_version_url_exports_artifact_url_without_alias_lookup(self) -> None:
        registry = Mock()
        registry.export_model_download_url.return_value = "http://mlflow.local/model/7/dataset/manifest.json"
        use_case = DetectionMlflowModelDownload(
            run=DetectionRun(
                experiment_name="exp",
                run_name="run",
                config=DetectionConfig(model_name="yolo_detector"),
            ),
            registry=registry,
            logger=Mock(),
        )

        result = use_case.download_version_url(version="7", artifact_path="dataset/manifest.json")

        self.assertEqual(result, "http://mlflow.local/model/7/dataset/manifest.json")
        registry.get_model_version_by_alias.assert_not_called()
        registry.export_model_download_url.assert_called_once_with(
            model_name="yolo_detector",
            version="7",
            artifact_path="dataset/manifest.json",
        )

    def test_download_champion_url_raises_when_alias_does_not_exist(self) -> None:
        registry = Mock()
        registry.get_model_version_by_alias.return_value = None
        use_case = DetectionMlflowModelDownload(
            run=DetectionRun(
                experiment_name="exp",
                run_name="run",
                config=DetectionConfig(model_name="yolo_detector"),
            ),
            registry=registry,
            logger=Mock(),
        )

        with self.assertRaisesRegex(ValueError, "Alias not found"):
            use_case.download_champion_url("checkpoints/best.pt")

    def test_download_version_to_file_uses_registry_download(self) -> None:
        registry = Mock()
        registry.download_model_artifact.return_value = "/tmp/best.pt"
        use_case = DetectionMlflowModelDownload(
            run=DetectionRun(
                experiment_name="exp",
                run_name="run",
                config=DetectionConfig(model_name="yolo_detector"),
            ),
            registry=registry,
            logger=Mock(),
        )

        result = use_case.download_version_to_file(
            version="7",
            artifact_path="checkpoints/best.pt",
            output_path="/tmp/best.pt",
        )

        self.assertEqual(result, "/tmp/best.pt")
        registry.download_model_artifact.assert_called_once_with(
            model_name="yolo_detector",
            version="7",
            artifact_path="checkpoints/best.pt",
            output_path="/tmp/best.pt",
        )

    def test_main_uses_champion_flow_by_default(self) -> None:
        use_case = Mock()
        use_case.download_champion_url.return_value = "http://mlflow.local/champion"

        with patch(
            "src.modules.tracking.application.use_case.detection.download.build_use_case",
            return_value=use_case,
        ):
            result = main()

        self.assertEqual(result, "http://mlflow.local/champion")
        use_case.download_champion_url.assert_called_once_with("checkpoints/best.pt")

    def test_main_uses_explicit_version_when_provided(self) -> None:
        use_case = Mock()
        use_case.download_version_url.return_value = "http://mlflow.local/version/8"

        with patch(
            "src.modules.tracking.application.use_case.detection.download.build_use_case",
            return_value=use_case,
        ):
            result = main(version="8", artifact_path="dataset/manifest.json")

        self.assertEqual(result, "http://mlflow.local/version/8")
        use_case.download_version_url.assert_called_once_with(
            version="8",
            artifact_path="dataset/manifest.json",
        )

    def test_main_downloads_artifact_when_output_path_is_provided(self) -> None:
        use_case = Mock()
        use_case.download_champion_to_file.return_value = "/tmp/best.pt"

        with patch(
            "src.modules.tracking.application.use_case.detection.download.build_use_case",
            return_value=use_case,
        ):
            result = main(output_path="/tmp/best.pt")

        self.assertEqual(result, "/tmp/best.pt")
        use_case.download_champion_to_file.assert_called_once_with(
            artifact_path="checkpoints/best.pt",
            output_path="/tmp/best.pt",
        )


if __name__ == "__main__":
    unittest.main()
