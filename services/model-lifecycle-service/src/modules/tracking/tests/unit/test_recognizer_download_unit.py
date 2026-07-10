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

from src.modules.tracking.application.use_case.recognizer.download import (  # noqa: E402
    RecognizerMlflowModelDownload,
    main,
)
from src.modules.tracking.domain.entity_objects import RecognizerRun  # noqa: E402
from src.modules.tracking.domain.value_objects import RecognizerConfig  # noqa: E402


class RecognizerMlflowModelDownloadUnitTest(unittest.TestCase):
    def test_download_champion_url_resolves_alias_then_exports_tensorboard_url(self) -> None:
        registry = Mock()
        registry.get_model_version_by_alias.return_value = SimpleNamespace(version="21")
        registry.export_model_download_url.return_value = (
            "http://mlflow.local/model/21/tensorboard/events.out.tfevents"
        )
        use_case = RecognizerMlflowModelDownload(
            run=RecognizerRun(
                experiment_name="exp",
                run_name="run",
                config=RecognizerConfig(model_name="vit_ctc_deepseek"),
            ),
            registry=registry,
            logger=Mock(),
        )

        result = use_case.download_champion_url("tensorboard/events.out.tfevents")

        self.assertEqual(
            result,
            "http://mlflow.local/model/21/tensorboard/events.out.tfevents",
        )
        registry.get_model_version_by_alias.assert_called_once_with(
            "vit_ctc_deepseek",
            "champion",
        )
        registry.export_model_download_url.assert_called_once_with(
            model_name="vit_ctc_deepseek",
            version="21",
            artifact_path="tensorboard/events.out.tfevents",
        )

    def test_download_version_url_rejects_artifact_root_that_was_never_uploaded(self) -> None:
        use_case = RecognizerMlflowModelDownload(
            run=RecognizerRun(
                experiment_name="exp",
                run_name="run",
                config=RecognizerConfig(model_name="vit_ctc_deepseek"),
            ),
            registry=Mock(),
            logger=Mock(),
        )

        with self.assertRaisesRegex(ValueError, "Unsupported artifact root"):
            use_case.download_version_url(version="9", artifact_path="training/results.csv")

    def test_main_uses_champion_flow_by_default(self) -> None:
        use_case = Mock()
        use_case.download_champion_url.return_value = "http://mlflow.local/champion"

        with patch(
            "src.modules.tracking.application.use_case.recognizer.download.build_use_case",
            return_value=use_case,
        ), patch(
            "src.modules.tracking.application.use_case.recognizer.download.subprocess.run",
            return_value=SimpleNamespace(stdout="/repo\n"),
        ):
            result = main()

        self.assertEqual(result, "http://mlflow.local/champion")
        use_case.download_champion_url.assert_called_once_with("checkpoints/best_cer.pt")

    def test_main_uses_explicit_version_when_provided(self) -> None:
        use_case = Mock()
        use_case.download_version_url.return_value = "http://mlflow.local/version/9"

        with patch(
            "src.modules.tracking.application.use_case.recognizer.download.build_use_case",
            return_value=use_case,
        ), patch(
            "src.modules.tracking.application.use_case.recognizer.download.subprocess.run",
            return_value=SimpleNamespace(stdout="/repo\n"),
        ):
            result = main(version="9", artifact_path="tensorboard/events.out.tfevents")

        self.assertEqual(result, "http://mlflow.local/version/9")
        use_case.download_version_url.assert_called_once_with(
            version="9",
            artifact_path="tensorboard/events.out.tfevents",
        )


if __name__ == "__main__":
    unittest.main()
