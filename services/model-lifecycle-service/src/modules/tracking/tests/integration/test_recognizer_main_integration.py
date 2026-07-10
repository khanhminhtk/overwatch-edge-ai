from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from mlflow import MlflowClient


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.tracking.tests.integration.helpers import (  # noqa: E402
    assert_run_created,
    delete_experiment,
)
from src.platform.config import ConfigLoader  # noqa: E402
from src.platform.tracking.mlflow.config import MlflowConfig  # noqa: E402


class RecognizerMainIntegrationTest(unittest.TestCase):
    def test_main_runs_recognizer_workflow(self) -> None:
        config_yaml = SERVICE_ROOT / "config/model_lifecycle_orchestrator_config.yaml"
        config_env = SERVICE_ROOT / "config/.env"
        script_path = SERVICE_ROOT / "src/modules/tracking/tests/integration/recognizer_main.py"
        mlflow_config = ConfigLoader.load(
            MlflowConfig,
            yaml_files=[str(config_yaml)],
            env_files=[str(config_env)],
            section="mlflow",
        )
        client = MlflowClient(tracking_uri=mlflow_config.tracking_uri)
        payload = None
        try:
            completed = subprocess.run(
                ["uv", "run", "python", str(script_path)],
                cwd=SERVICE_ROOT.parent.parent,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            run = assert_run_created(
                client=client,
                experiment_name=payload["experiment_name"],
                run_id=payload["run_id"],
            )
            self.assertEqual(run.info.run_id, payload["run_id"])
            self.assertTrue(payload["model_version"])
        finally:
            if payload is not None:
                delete_experiment(
                    client=client,
                    experiment_name=payload["experiment_name"],
                )


if __name__ == "__main__":
    unittest.main()
