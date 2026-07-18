from __future__ import annotations

import numpy as np

from src.platform.capture import LowQualityCaptureConfig, LowQualityFrameStore


def test_low_quality_store_keeps_only_materially_different_frames(tmp_path) -> None:
    store = LowQualityFrameStore(
        LowQualityCaptureConfig(
            success_output_dir=str(tmp_path / "success"),
            fail_output_dir=str(tmp_path / "fail"),
            min_frame_difference=0.1,
            min_save_interval_seconds=0,
        ),
        tmp_path,
    )
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    white = np.full((100, 100, 3), 255, dtype=np.uint8)

    first = store.save_if_distinct("fail", black)

    assert first is not None and first.exists()
    assert store.save_if_distinct("fail", black) is None
    assert store.save_if_distinct("fail", white) is not None
    assert store.save_if_distinct("success", black) is not None
