from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.tracking.tests.integration.helpers import (  # noqa: E402
    assert_run_created,
    delete_experiment,
)


class IntegrationHelpersUnitTest(unittest.TestCase):
    def test_assert_run_created_returns_run_when_experiment_matches(self) -> None:
        client = Mock()
        client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="7")
        run = SimpleNamespace(info=SimpleNamespace(experiment_id="7"))
        client.get_run.return_value = run

        result = assert_run_created(
            client=client,
            experiment_name="tracking-exp",
            run_id="run-123",
        )

        self.assertIs(result, run)

    def test_delete_experiment_returns_false_when_experiment_is_missing(self) -> None:
        client = Mock()
        client.get_experiment_by_name.return_value = None

        deleted = delete_experiment(client=client, experiment_name="missing-exp")

        self.assertFalse(deleted)
        client.delete_experiment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
