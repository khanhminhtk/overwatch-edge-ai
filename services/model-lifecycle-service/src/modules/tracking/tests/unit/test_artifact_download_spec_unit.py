from __future__ import annotations

import sys
import unittest
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.tracking.domain.value_objects import TrackingArtifactDownloadSpec  # noqa: E402


class TrackingArtifactDownloadSpecUnitTest(unittest.TestCase):
    def test_detection_spec_accepts_uploaded_training_artifact(self) -> None:
        spec = TrackingArtifactDownloadSpec.for_detection("training/results.csv")

        self.assertEqual(spec.artifact_path, "training/results.csv")
        self.assertEqual(spec.root_dir, "training")

    def test_recognizer_spec_accepts_uploaded_tensorboard_artifact(self) -> None:
        spec = TrackingArtifactDownloadSpec.for_recognizer("tensorboard/events.out.tfevents")

        self.assertEqual(spec.artifact_path, "tensorboard/events.out.tfevents")
        self.assertEqual(spec.root_dir, "tensorboard")

    def test_detection_spec_rejects_untracked_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported artifact root"):
            TrackingArtifactDownloadSpec.for_detection("tensorboard/events.out.tfevents")


if __name__ == "__main__":
    unittest.main()
