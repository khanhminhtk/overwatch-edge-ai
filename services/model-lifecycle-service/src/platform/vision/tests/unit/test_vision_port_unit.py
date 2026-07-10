from __future__ import annotations

import unittest

from src.platform.vision.protocols import VisionModelPort


class VisionModelPortTest(unittest.TestCase):
    def test_cannot_instantiate_abc(self) -> None:
        with self.assertRaises(TypeError):
            VisionModelPort()  # type: ignore[abstract]

    def test_subclass_with_all_methods_works(self) -> None:
        class Concrete(VisionModelPort):
            def execute(self, image_path: str) -> str:
                return f"executed {image_path}"

            def execute_batch(self, images_paths: list[str]) -> list[str]:
                return [f"executed {p}" for p in images_paths]

        impl = Concrete()
        self.assertEqual(impl.execute("/img.png"), "executed /img.png")
        self.assertEqual(
            impl.execute_batch(["/a.png", "/b.png"]),
            ["executed /a.png", "executed /b.png"],
        )

    def test_subclass_missing_execute_raises(self) -> None:
        class Incomplete(VisionModelPort):  # type: ignore[abstract]
            def execute_batch(self, images_paths: list[str]) -> list[str]:
                return []

        with self.assertRaises(TypeError):
            Incomplete()

    def test_subclass_missing_execute_batch_raises(self) -> None:
        class Incomplete(VisionModelPort):  # type: ignore[abstract]
            def execute(self, image_path: str) -> str:
                return image_path

        with self.assertRaises(TypeError):
            Incomplete()


if __name__ == "__main__":
    unittest.main()
