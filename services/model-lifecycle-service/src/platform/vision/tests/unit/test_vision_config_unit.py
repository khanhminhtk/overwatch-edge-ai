from __future__ import annotations

import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from src.platform.vision.config import GoogleVisionConfig


class GoogleVisionConfigTest(unittest.TestCase):
    def test_creates_with_valid_path(self) -> None:
        config = GoogleVisionConfig(
            certificate_path="/path/to/key.json"
        )
        self.assertEqual(config.certificate_path, "/path/to/key.json")

    def test_empty_path_raises(self) -> None:
        for path in ("", "  ", "\t"):
            with self.subTest(path=repr(path)):
                with self.assertRaises(ValueError):
                    GoogleVisionConfig(certificate_path=path)

    def test_is_frozen(self) -> None:
        config = GoogleVisionConfig(certificate_path="/a/b.json")
        with self.assertRaises(FrozenInstanceError):
            config.certificate_path = "/other"  # type: ignore[misc]

    def test_equal_when_fields_match(self) -> None:
        c1 = GoogleVisionConfig(certificate_path="/a/b.json")
        c2 = GoogleVisionConfig(certificate_path="/a/b.json")
        self.assertEqual(c1, c2)

    def test_equal_with_image_test_path(self) -> None:
        c1 = GoogleVisionConfig(
            certificate_path="/a.json", image_test_path="/img.png"
        )
        c2 = GoogleVisionConfig(
            certificate_path="/a.json", image_test_path="/img.png"
        )
        self.assertEqual(c1, c2)

    def test_not_equal_when_fields_differ(self) -> None:
        c1 = GoogleVisionConfig(certificate_path="/a/b.json")
        c2 = GoogleVisionConfig(certificate_path="/c/d.json")
        self.assertNotEqual(c1, c2)

    def test_convert_path_to_absolute_updates_certificate(self) -> None:
        config = GoogleVisionConfig(certificate_path="rel/ative.json")
        config.convert_path_to_absolute("/base")
        self.assertEqual(config.certificate_path, "/base/rel/ative.json")

    def test_convert_path_to_absolute_updates_image_test_path(self) -> None:
        config = GoogleVisionConfig(
            certificate_path="c.json", image_test_path="i.png"
        )
        config.convert_path_to_absolute("/base")
        self.assertEqual(config.image_test_path, "/base/i.png")

    def test_convert_path_to_absolute_skips_none_image_test_path(self) -> None:
        config = GoogleVisionConfig(certificate_path="c.json")
        config.convert_path_to_absolute("/base")
        self.assertIsNone(config.image_test_path)

    def test_convert_path_to_absolute_keeps_absolute_path(self) -> None:
        config = GoogleVisionConfig(certificate_path="/already/absolute.json")
        config.convert_path_to_absolute("/base")
        self.assertEqual(config.certificate_path, "/already/absolute.json")

    def test_convert_path_to_absolute_prefers_existing_repo_root_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            service_root = repo_root / "services" / "model-lifecycle-service"
            service_root.mkdir(parents=True)
            target = repo_root / "services" / "model-lifecycle-service" / "config" / "key.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}", encoding="utf-8")

            config = GoogleVisionConfig(
                certificate_path="services/model-lifecycle-service/config/key.json"
            )
            config.convert_path_to_absolute(str(service_root))

            self.assertEqual(config.certificate_path, str(target))

    def test_convert_path_to_absolute_prefers_existing_repo_root_bare_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            service_root = repo_root / "services" / "model-lifecycle-service"
            service_root.mkdir(parents=True)
            target = repo_root / "image.jpg"
            target.write_text("x", encoding="utf-8")

            config = GoogleVisionConfig(
                certificate_path="c.json", image_test_path="image.jpg"
            )
            config.convert_path_to_absolute(str(service_root))

            self.assertEqual(config.image_test_path, str(target))


if __name__ == "__main__":
    unittest.main()
