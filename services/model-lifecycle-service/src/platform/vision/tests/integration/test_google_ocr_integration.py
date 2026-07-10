from __future__ import annotations

import os
import unittest

from src.platform.config import ConfigLoader
from src.platform.vision.config import GoogleVisionConfig
from src.platform.vision.google_ocr import GoogleVisionOCR

# When a valid service-account JSON exists under this path in the project,
# the integration tests will exercise the real Google Vision API.
# Otherwise they are skipped.
_CONFIG_SECTION = "GoogleVision"
_KNOWN_SERVICE_ACCOUNT_PATH: str | None = None


def _resolve_service_account_path() -> str | None:
    pwd = os.path.normpath(
        os.path.join(os.path.dirname(__file__), *([".."] * 8))
    )
    try:
        config = ConfigLoader.load(
            GoogleVisionConfig,
            yaml_files=[
                f"{pwd}/services/model-lifecycle-service/config/"
                "model_lifecycle_orchestrator_config.yaml"
            ],
            env_files=[
                f"{pwd}/services/model-lifecycle-service/config/.env"
            ],
            section=_CONFIG_SECTION,
        )
        path = config.certificate_path
        if os.path.isfile(path):
            return path
        # Try resolving relative to project root
        abs_path = os.path.join(pwd, path)
        if os.path.isfile(abs_path):
            return abs_path
    except Exception:
        pass
    return None


def _find_test_image() -> str | None:
    pwd = os.path.normpath(
        os.path.join(os.path.dirname(__file__), *([".."] * 8))
    )
    candidates = [
        f"{pwd}/services/model-lifecycle-service/tests/fixtures/test_image.png",
        f"{pwd}/data/runtime/image_raw_inferences/sample.png",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


@unittest.skipIf(
    _resolve_service_account_path() is None,
    "No Google Vision service account JSON available",
)
class GoogleVisionOCRIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sa_path = _resolve_service_account_path()
        assert sa_path is not None
        config = GoogleVisionConfig(certificate_path=sa_path)
        cls.ocr = GoogleVisionOCR(config=config)

    def test_constructor_with_real_config(self) -> None:
        self.assertIsNotNone(self.ocr)

    def test_execute_returns_expected_shape(self) -> None:
        img_path = _find_test_image()
        if img_path is None:
            self.skipTest("No test image found")
        result = self.ocr.execute(img_path)
        self.assertIn("image_path", result)
        self.assertIn("success", result)
        self.assertIn("error", result)
        self.assertIn("words", result)
        self.assertIsInstance(result["words"], list)

    def test_execute_with_nonexistent_path(self) -> None:
        result = self.ocr.execute("/tmp/nonexistent_image_xyz.png")
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])


if __name__ == "__main__":
    unittest.main()
