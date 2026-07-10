from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

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
        object.__setattr__(config, "certificate_path", "rel/ative.json")
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


if __name__ == "__main__":
    unittest.main()
