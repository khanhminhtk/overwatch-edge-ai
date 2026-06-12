import os
from typing import Any, List
import warnings

from google.cloud import vision

from src.applications.ports.modelai import IModelAI

warnings.filterwarnings("ignore", category=FutureWarning)

class GoogleVisionOCR(IModelAI):
    def __init__(self, service_account_path: str):
        self.client = vision.ImageAnnotatorClient.from_service_account_json(
            service_account_path
        )

    def execute(self, image_path: str) -> Any:
        with open(image_path, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        response = self.client.text_detection(image=image)

        return self._extract_words_from_response(response, image_path)

    def execute_batch(self, images_paths: List[str]) -> List[Any]:
        batch_response = self.client.batch_annotate_images(
            requests=images_paths
        )
        results = [self._extract_words_from_response(response, image_path) for response, image_path in zip(batch_response.responses, images_paths)]
        return results

    def _extract_words_from_response(self, response, image_path: str):
        if response.error.message:
            return {
                "image_path": image_path,
                "success": False,
                "error": response.error.message,
                "words": [],
            }

        words = []

        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        text = "".join(symbol.text for symbol in word.symbols)

                        bbox = [
                            {"x": vertex.x, "y": vertex.y}
                            for vertex in word.bounding_box.vertices
                        ]

                        words.append(
                            {
                                "text": text,
                                "bbox": bbox,
                                "confidence": word.confidence,
                            }
                        )

        return {
            "image_path": image_path,
            "success": True,
            "error": None,
            "words": words,
        }