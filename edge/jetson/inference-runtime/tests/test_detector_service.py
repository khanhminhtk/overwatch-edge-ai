from __future__ import annotations

import numpy as np
import pytest

from src.modules.inference import (
    DetectorService,
    DetectorSettings,
    decode_yolo_output,
    preprocess_bgr,
)

SETTINGS = DetectorSettings(
    input_width=640,
    input_height=640,
    confidence_threshold=0.5,
    nms_threshold=0.45,
)


def test_preprocess_bgr_returns_normalized_nchw_rgb() -> None:
    frame = np.array([[[0, 0, 255]]], dtype=np.uint8)

    tensor = preprocess_bgr(frame, width=2, height=2)

    assert tensor.shape == (1, 3, 2, 2)
    np.testing.assert_allclose(tensor[0, :, 0, 0], [1.0, 0.0, 0.0])


def test_decode_yolo_output_filters_and_runs_nms() -> None:
    output = np.zeros((1, 5, 2), dtype=np.float32)
    output[0, :4, 0] = [320, 320, 200, 200]
    output[0, :4, 1] = [322, 322, 200, 200]
    output[0, 4] = [0.9, 0.8]

    detections = decode_yolo_output(
        output,
        original_width=1280,
        original_height=640,
        input_width=640,
        input_height=640,
        confidence_threshold=0.25,
        nms_threshold=0.45,
    )

    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.9)
    assert (detections[0].x1, detections[0].y1, detections[0].x2, detections[0].y2) == (
        440,
        220,
        840,
        420,
    )


def test_detector_service_passes_images_and_decodes_output() -> None:
    class FakeModel:
        def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            assert inputs["images"].shape == (1, 3, 640, 640)
            output = np.zeros((1, 5, 1), dtype=np.float32)
            output[0, :, 0] = [320, 320, 320, 320, 0.9]
            return {"output0": output}

    result = DetectorService(FakeModel(), SETTINGS).detect(
        np.zeros((480, 640, 3), dtype=np.uint8)
    )

    assert len(result) == 1
    assert result[0].width == 320
    assert result[0].height == 240
