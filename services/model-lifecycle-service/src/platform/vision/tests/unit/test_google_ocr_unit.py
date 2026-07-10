from __future__ import annotations

import unittest
from unittest.mock import MagicMock, call, mock_open, patch

from src.platform.vision.config import GoogleVisionConfig
from src.platform.vision.google_ocr import GoogleVisionOCR


class GoogleVisionOCRUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GoogleVisionConfig(
            certificate_path="/fake/key.json"
        )

        self.mock_client = MagicMock()
        self.mock_client.text_detection = MagicMock()
        self.mock_client.batch_annotate_images = MagicMock()

        patcher = patch(
            "src.platform.vision.google_ocr.vision.ImageAnnotatorClient"
        )
        self.mock_client_class = patcher.start()
        self.mock_client_class.from_service_account_json.return_value = (
            self.mock_client
        )
        self.addCleanup(patcher.stop)

        self.ocr = GoogleVisionOCR(config=self.config)

    def _make_word(self, text: str, confidence: float) -> MagicMock:
        word = MagicMock()
        word.symbols = [MagicMock(text=ch) for ch in text]
        word.confidence = confidence

        v1, v2 = MagicMock(), MagicMock()
        v1.x, v1.y = 0, 0
        v2.x, v2.y = 10, 20
        word.bounding_box.vertices = [v1, v1, v2, v2]
        return word

    def _make_ocr_response(
        self, words: list[MagicMock], error_message: str | None = None
    ) -> MagicMock:
        response = MagicMock()
        response.error.message = error_message

        if error_message:
            return response

        page = MagicMock()
        block = MagicMock()
        paragraph = MagicMock()
        paragraph.words = words
        block.paragraphs = [paragraph]
        page.blocks = [block]
        response.full_text_annotation.pages = [page]
        return response

    def test_constructor_creates_client(self) -> None:
        self.mock_client_class.from_service_account_json.assert_called_once_with(
            "/fake/key.json"
        )
        self.assertIsNotNone(self.ocr)

    def test_execute_returns_extracted_words(self) -> None:
        word_a = self._make_word("hello", 0.95)
        word_b = self._make_word("world", 0.88)
        mock_resp = self._make_ocr_response([word_a, word_b])

        self.mock_client.text_detection.return_value = mock_resp

        with patch("builtins.open", mock_open(read_data=b"fake-image")):
            result = self.ocr.execute("/img/test.png")

        self.mock_client.text_detection.assert_called_once()
        self.assertTrue(result["success"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["image_path"], "/img/test.png")
        self.assertEqual(len(result["words"]), 2)
        self.assertEqual(result["words"][0]["text"], "hello")
        self.assertEqual(result["words"][0]["confidence"], 0.95)
        self.assertEqual(result["words"][1]["text"], "world")
        self.assertEqual(result["words"][1]["confidence"], 0.88)

    def test_execute_returns_error_response(self) -> None:
        mock_resp = self._make_ocr_response(
            [], error_message="API failure"
        )
        self.mock_client.text_detection.return_value = mock_resp

        with patch("builtins.open", mock_open(read_data=b"data")):
            result = self.ocr.execute("/img/bad.png")

        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "API failure")
        self.assertEqual(result["words"], [])

    def test_execute_batch_returns_results(self) -> None:
        word_a = self._make_word("one", 0.9)
        word_b = self._make_word("two", 0.8)

        mock_resp1 = self._make_ocr_response([word_a])
        mock_resp2 = self._make_ocr_response([word_b])

        batch_response = MagicMock()
        batch_response.responses = [mock_resp1, mock_resp2]
        self.mock_client.batch_annotate_images.return_value = batch_response

        with patch("builtins.open", mock_open(read_data=b"img-data")):
            results = self.ocr.execute_batch(["/a.png", "/b.png"])

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["success"])
        self.assertTrue(results[1]["success"])
        self.assertEqual(results[0]["words"][0]["text"], "one")
        self.assertEqual(results[1]["words"][0]["text"], "two")

    def test_execute_batch_passes_annotate_image_requests(self) -> None:
        batch_response = MagicMock()
        batch_response.responses = [self._make_ocr_response([])]
        self.mock_client.batch_annotate_images.return_value = batch_response

        with patch("builtins.open", mock_open(read_data=b"data")):
            self.ocr.execute_batch(["/img.png"])

        call_kwargs = self.mock_client.batch_annotate_images.call_args.kwargs
        requests = call_kwargs["requests"]
        self.assertEqual(len(requests), 1)
        self.assertIsInstance(
            requests[0],
            self._get_annotate_image_request_class(),
        )

    def _get_annotate_image_request_class(self):
        from google.cloud import vision
        return vision.AnnotateImageRequest

    def test_execute_batch_handles_error(self) -> None:
        mock_resp = self._make_ocr_response(
            [], error_message="batch error"
        )
        batch_response = MagicMock()
        batch_response.responses = [mock_resp]
        self.mock_client.batch_annotate_images.return_value = batch_response

        with patch("builtins.open", mock_open(read_data=b"data")):
            results = self.ocr.execute_batch(["/fail.png"])

        self.assertFalse(results[0]["success"])
        self.assertEqual(results[0]["error"], "batch error")

    def test_extract_words_from_response_empty_pages(self) -> None:
        response = MagicMock()
        response.error.message = None
        response.full_text_annotation.pages = []

        result = self.ocr._extract_words_from_response(
            response, "/empty.png"
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["words"], [])


if __name__ == "__main__":
    unittest.main()
