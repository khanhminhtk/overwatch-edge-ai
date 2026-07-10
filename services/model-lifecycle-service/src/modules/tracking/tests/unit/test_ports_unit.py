from __future__ import annotations

import sys
import unittest
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.tracking.application.ports.artifacts import ArtifactPort  # noqa: E402
from src.modules.tracking.application.ports.registry import RegistryPort  # noqa: E402
from src.modules.tracking.application.ports.tracking import TrackingPort  # noqa: E402


class _TrackingStep:
    def start_run(self) -> str:
        return "run-123"

    def execute(self, run_id: str, pwd: str) -> None:
        return None

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        return None


class _ArtifactStep:
    def execute(
        self,
        checkpoint_name: list[str],
        run_id: str | None = None,
        pwd: str | None = None,
    ) -> None:
        return None


class _RegistryStep:
    def execute(self, run_id: str, pwd: str) -> str:
        return "1"


class PortsUnitTest(unittest.TestCase):
    def test_tracking_port_matches_workflow_contract(self) -> None:
        self.assertIsInstance(_TrackingStep(), TrackingPort)

    def test_artifact_port_matches_workflow_contract(self) -> None:
        self.assertIsInstance(_ArtifactStep(), ArtifactPort)

    def test_registry_port_matches_workflow_contract(self) -> None:
        self.assertIsInstance(_RegistryStep(), RegistryPort)


if __name__ == "__main__":
    unittest.main()
