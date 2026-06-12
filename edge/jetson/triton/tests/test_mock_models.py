from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


TRITON_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    module_path = TRITON_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_detector_mock_generates_expected_output_shapes():
    detector_module = _load_module("mock_detector_model", "detector/1/model.py")

    images = np.ones((2, 3, 640, 640), dtype=np.float32)

    boxes, scores, class_ids, detections = detector_module.generate_detector_outputs(
        images,
        max_detections=4,
    )

    assert boxes.shape == (2, 4, 4)
    assert scores.shape == (2, 4)
    assert class_ids.shape == (2, 4)
    assert detections.shape == (2, 1)
    assert boxes.dtype == np.float32
    assert scores.dtype == np.float32
    assert class_ids.dtype == np.int32
    assert detections.dtype == np.int32


def test_detector_mock_changes_detection_count_by_input_intensity():
    detector_module = _load_module("mock_detector_model", "detector/1/model.py")

    dark = np.zeros((1, 3, 640, 640), dtype=np.float32)
    bright = np.ones((1, 3, 640, 640), dtype=np.float32)

    _, _, _, dark_count = detector_module.generate_detector_outputs(dark, max_detections=4)
    _, _, _, bright_count = detector_module.generate_detector_outputs(bright, max_detections=4)

    assert int(bright_count[0, 0]) > int(dark_count[0, 0])


def test_recognizer_mock_generates_text_confidence_and_logits():
    recognizer_module = _load_module("mock_recognizer_model", "recognizer/1/model.py")

    crops = np.ones((3, 3, 48, 160), dtype=np.float32)

    text, confidence, logits = recognizer_module.generate_recognizer_outputs(
        crops,
        sequence_length=8,
    )

    assert text.shape == (3, 1)
    assert confidence.shape == (3, 1)
    assert logits.shape == (3, 8, len(recognizer_module.VOCAB))
    assert confidence.dtype == np.float32
    assert logits.dtype == np.float32
    assert text.dtype == np.object_


def test_recognizer_mock_varies_text_with_input():
    recognizer_module = _load_module("mock_recognizer_model", "recognizer/1/model.py")

    crops = np.stack(
        [
            np.zeros((3, 48, 160), dtype=np.float32),
            np.ones((3, 48, 160), dtype=np.float32),
        ],
        axis=0,
    )

    text, confidence, _ = recognizer_module.generate_recognizer_outputs(
        crops,
        sequence_length=8,
    )

    assert text[0, 0] != text[1, 0]
    assert float(confidence[1, 0]) >= float(confidence[0, 0])
