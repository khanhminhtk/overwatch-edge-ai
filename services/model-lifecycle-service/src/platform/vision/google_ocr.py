from typing import Any

from google.cloud import vision

from src.platform.vision.config import GoogleVisionConfig
from src.platform.vision.protocols import VisionModelPort
from src.platform.vision.predictions import BoundingBox, TextRegionPrediction


class GoogleVisionOCR(VisionModelPort):
    def __init__(self, config: GoogleVisionConfig) -> None:
        self._client = vision.ImageAnnotatorClient.from_service_account_json(
            config.certificate_path
        )

    def execute(self, image_path: str) -> Any:
        with open(image_path, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        response = self._client.text_detection(image=image)

        return self._extract_words_from_response(response, image_path)

    def execute_batch(self, images_paths: list[str]) -> list[Any]:
        requests = [
            vision.AnnotateImageRequest(
                image=vision.Image(content=open(p, "rb").read())
            )
            for p in images_paths
        ]
        batch_response = self._client.batch_annotate_images(requests=requests)
        return [
            self._extract_words_from_response(response, image_path)
            for response, image_path in zip(
                batch_response.responses, images_paths
            )
        ]

    def _extract_words_from_response(
        self, response: Any, image_path: str
    ) -> TextRegionPrediction:
        if response.error.message:
            return TextRegionPrediction(
                bounding_box=BoundingBox(x_min=0, y_min=0, x_max=0, y_max=0),
                text="",
                confidence=None
            )

        words: list[TextRegionPrediction] = []

        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        text = "".join(
                            symbol.text for symbol in word.symbols
                        )

                        bbox = [
                            {"x": vertex.x, "y": vertex.y}
                            for vertex in word.bounding_box.vertices
                        ]

                        words.append(
                            TextRegionPrediction(
                                bounding_box=BoundingBox.from_polygon(bbox),
                                text=text,
                                confidence=word.confidence
                            )
                        )

        return {
            "image_path": image_path,
            "success": True,
            "error": None,
            "words": words,
        }

# if __name__ == "__main__":
#     import os
#     import sys

#     _root = os.path.normpath(
#         os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..")
#     )
#     if _root not in sys.path:
#         sys.path.insert(0, _root)

#     from subprocess import run

#     from src.platform.config import ConfigLoader
#     from src.platform.vision.config import GoogleVisionConfig

#     pwd = run(["pwd"], capture_output=True, text=True).stdout.strip()

#     config = ConfigLoader.load(
#         GoogleVisionConfig,
#         yaml_files=[f"{pwd}/services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml"],
#         env_files=[f"{pwd}/services/model-lifecycle-service/config/.env"],
#         section="GoogleVision"
#     )

#     config.convert_path_to_absolute(pwd)

#     if not os.path.isfile(config.certificate_path):
#         print(
#             f"[ERROR] certificate_path not found: {config.certificate_path}\n"
#             "Set google_vision_certificate_path in config/.env to a valid "
#             "Google service account JSON file."
#         )
#         sys.exit(1)

#     if config.image_test_path is None or not os.path.isfile(config.image_test_path):
#         print(
#             f"[ERROR] image_test_path not found: {config.image_test_path}\n"
#             "Set google_vision_image_test_path in config/.env to a valid image file."
#         )
#         sys.exit(1)

#     try:
#         ocr = GoogleVisionOCR(config=config)
#     except Exception as e:
#         print(
#             f"[ERROR] Failed to initialize GoogleVisionOCR with:\n"
#             f"  certificate_path: {config.certificate_path}\n"
#             f"  error: {e}\n"
#             "Make sure google_vision_certificate_path points to a valid "
#             "Google service account JSON file."
#         )
#         sys.exit(1)

#     try:
#         result = ocr.execute(config.image_test_path)
#     except Exception as e:
#         print(
#             f"[ERROR] Failed to execute OCR on:\n"
#             f"  image: {config.image_test_path}\n"
#             f"  error: {e}"
#         )
#         sys.exit(1)

#     print(result)

