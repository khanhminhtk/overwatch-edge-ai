from __future__ import annotations

import numpy as np

from src.modules.inference import (
    Detection,
    RecognizerSettings,
    RecognizerService,
    decode_ctc_logits,
    preprocess_crop,
)
from src.modules.inference import InferenceState

VOCABULARY = "0123456789a"
SETTINGS = RecognizerSettings(
    num_patches=16,
    patch_width=64,
    patch_height=64,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    vocabulary=VOCABULARY,
    blank_index=0,
    num_classes=192,
)


def test_vocab_and_ctc_decode_reserve_blank_and_map_padding_to_unknown() -> None:
    assert len(VOCABULARY) == 11
    logits = np.zeros((1, 7, 192), dtype=np.float32)
    logits[0, np.arange(7), [0, 1, 1, 0, 11, 187, 191]] = 1.0

    tokens, text = decode_ctc_logits(logits, SETTINGS)

    assert tokens == [1, 11, 187, 191]
    assert text == "0a<unk><unk>"


def test_preprocess_crop_creates_16_rgb_patches() -> None:
    crop = np.zeros((8, 32, 3), dtype=np.uint8)
    crop[:, :, 2] = 255

    tensor = preprocess_crop(crop, SETTINGS)

    assert tensor.shape == (1, 16, 3, 64, 64)
    np.testing.assert_allclose(
        tensor[0, 0, :, 0, 0],
        [(1.0 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    )


def test_preprocess_crop_supports_a_crop_narrower_than_16_pixels() -> None:
    crop = np.zeros((8, 4, 3), dtype=np.uint8)
    crop[:, :, 2] = 255

    tensor = preprocess_crop(crop, SETTINGS)

    assert tensor.shape == (1, 16, 3, 64, 64)
    np.testing.assert_allclose(
        tensor[0, 0, :, 0, 0],
        [(1.0 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    )


def test_recognizer_reads_detector_snapshot_and_decodes_each_crop() -> None:
    state = InferenceState()
    state.publish(
        np.zeros((100, 100, 3), dtype=np.uint8), [Detection(0, 0.9, 10, 10, 90, 90)]
    )

    class FakeModel:
        def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            assert inputs["images"].shape == (1, 16, 3, 64, 64)
            logits = np.zeros((1, 16, 192), dtype=np.float32)
            for timestep, class_index in enumerate([1, 1, 0, 11]):
                logits[0, timestep, class_index] = 1.0
            return {"logits": logits}

    results = RecognizerService(FakeModel(), SETTINGS, state=state).recognize_latest()

    assert len(results) == 1
    assert results[0].text == "0a"


def test_recognizer_can_read_a_full_image_without_detector() -> None:
    class FakeModel:
        def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            assert inputs["images"].shape == (1, 16, 3, 64, 64)
            logits = np.zeros((1, 4, 192), dtype=np.float32)
            logits[0, :, 1] = 1.0
            return {"logits": logits}

    results = RecognizerService(FakeModel(), SETTINGS).recognize_frame(
        np.zeros((100, 100, 3), dtype=np.uint8)
    )

    assert results[0].text == "0"
