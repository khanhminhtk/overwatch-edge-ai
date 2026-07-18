from __future__ import annotations

import numpy as np

from src.modules.inference import (
    Detection,
    InferenceState,
    Recognition,
    VisualizerService,
)


def test_visualizer_renders_matching_detection_and_recognition() -> None:
    state = InferenceState()
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    detection = Detection(0, 0.9, 20, 30, 120, 80)
    snapshot = state.publish(frame, [detection])
    state.publish_recognitions(
        snapshot.sequence, [Recognition(detection, "phô mai", (1, 2))]
    )

    rendered = VisualizerService(state).render_latest()

    assert rendered is not None
    assert tuple(rendered[30, 20]) == (0, 255, 0)
    assert np.any(rendered != 0)


def test_visualizer_refuses_recognition_from_another_frame() -> None:
    state = InferenceState()
    detection = Detection(0, 0.9, 1, 1, 10, 10)
    state.publish_recognitions(1, [Recognition(detection, "text", (1,))])
    state.publish(np.zeros((20, 20, 3), dtype=np.uint8), [])
    state.publish(np.zeros((20, 20, 3), dtype=np.uint8), [detection])

    assert VisualizerService(state).render_latest() is None
