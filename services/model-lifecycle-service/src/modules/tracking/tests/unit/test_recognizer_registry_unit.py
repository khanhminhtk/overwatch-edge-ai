from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from mlflow.pyfunc import PythonModel


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.tracking.application.use_case.recognizer.registry import (  # noqa: E402
    RecognizerMlflowModelRegistry,
)
from src.modules.tracking.domain.entity_objects import RecognizerRun  # noqa: E402
from src.modules.tracking.domain.value_objects import RecognizerConfig  # noqa: E402


class RecognizerMlflowModelRegistryUnitTest(unittest.TestCase):
    def test_execute_promotes_when_best_val_cer_is_lower_than_champion(self) -> None:
        registry = Mock()
        registry.log_pyfunc_model.return_value = "runs:/run-123/vit_ctc_deepseek"
        registry.register_model.return_value = "12"
        registry.get_model_version_by_alias.return_value = Mock(tags={"best_val_cer": "0.08"})
        tracker = Mock()
        use_case = RecognizerMlflowModelRegistry(
            run=RecognizerRun(
                experiment_name="exp",
                run_name="run",
                config=RecognizerConfig(
                    model_name="vit_ctc_deepseek",
                    checkpoint_dir="data/checkpoint/recognizer",
                    best_checkpoint_name="best_cer.pt",
                    last_checkpoint_name="last_checkpoint.pt",
                ),
            ),
            registry=registry,
            tracker=tracker,
            logger=Mock(),
        )

        with patch.object(
            use_case,
            "_load_checkpoint_bundle",
            return_value={
                "epoch_best": 7,
                "epoch_last": 11,
                "version": "1.0",
                "best_val_loss": 0.2,
                "best_val_cer": 0.05,
                "encoder_num_classes": 4,
                "encoder_vocab_size": 4,
                "model_total_params": 100,
            },
        ):
            version = use_case.execute(run_id="run-123", pwd="/repo")

        self.assertEqual(version, "12")
        registry.set_model_version_tags.assert_called_once_with(
            registered_model_name="vit_ctc_deepseek",
            model_version="12",
            tags={"best_val_cer": "0.05"},
        )
        registry.set_model_alias.assert_called_once_with("vit_ctc_deepseek", "champion", "12")
        registry.transition_model_stage.assert_called_once_with("vit_ctc_deepseek", "12", "Production")

    def test_execute_does_not_replace_champion_on_tie(self) -> None:
        registry = Mock()
        registry.log_pyfunc_model.return_value = "runs:/run-123/vit_ctc_deepseek"
        registry.register_model.return_value = "12"
        registry.get_model_version_by_alias.return_value = Mock(tags={"best_val_cer": "0.05"})
        tracker = Mock()
        use_case = RecognizerMlflowModelRegistry(
            run=RecognizerRun(
                experiment_name="exp",
                run_name="run",
                config=RecognizerConfig(
                    model_name="vit_ctc_deepseek",
                    checkpoint_dir="data/checkpoint/recognizer",
                    best_checkpoint_name="best_cer.pt",
                    last_checkpoint_name="last_checkpoint.pt",
                ),
            ),
            registry=registry,
            tracker=tracker,
            logger=Mock(),
        )

        with patch.object(
            use_case,
            "_load_checkpoint_bundle",
            return_value={
                "epoch_best": 7,
                "epoch_last": 11,
                "version": "1.0",
                "best_val_loss": 0.2,
                "best_val_cer": 0.05,
                "encoder_num_classes": 4,
                "encoder_vocab_size": 4,
                "model_total_params": 100,
            },
        ):
            version = use_case.execute(run_id="run-123", pwd="/repo")

        self.assertEqual(version, "12")
        registry.set_model_alias.assert_not_called()
        registry.transition_model_stage.assert_not_called()

    def test_log_model_version_info_logs_input_signature_and_run_metadata(self) -> None:
        registry = Mock()
        tracker = Mock()
        use_case = RecognizerMlflowModelRegistry(
            run=RecognizerRun(
                experiment_name="exp",
                run_name="run",
                config=RecognizerConfig(
                    model_name="vit_ctc_deepseek",
                    checkpoint_dir="data/checkpoint/recognizer",
                    best_checkpoint_name="best_cer.pt",
                    last_checkpoint_name="last_checkpoint.pt",
                ),
            ),
            registry=registry,
            tracker=tracker,
            logger=Mock(),
        )

        with patch.object(
            use_case,
            "_load_checkpoint_bundle",
            return_value={
                "epoch_best": 7,
                "epoch_last": 11,
                "version": "1.0",
                "best_val_loss": 0.2,
                "best_val_cer": 0.05,
                "encoder_num_classes": 4,
                "encoder_vocab_size": 4,
                "model_total_params": 100,
            },
        ):
            use_case.execute(run_id="run-123", pwd="/repo")

        kwargs = registry.log_pyfunc_model.call_args.kwargs
        self.assertIsInstance(kwargs["input_example"], pd.DataFrame)
        self.assertIsNotNone(kwargs["signature"])
        self.assertIsInstance(kwargs["python_model"], PythonModel)
        tracker.log_params.assert_called_once_with(
            run_id="run-123",
            params={"epoch_best": 7, "epoch_last": 11},
        )
        tracker.log_metrics.assert_called_once_with(
            run_id="run-123",
            metrics={"best_val_loss": 0.2, "best_val_cer": 0.05},
        )


if __name__ == "__main__":
    unittest.main()
