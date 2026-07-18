from __future__ import annotations

from pathlib import Path

import pytest

from src.entrypoints.inference_daemon.main import (
    FpsMeter,
    _capture_category,
    _image_output_path,
    parse_args,
    source_from_args,
)
from src.modules.inference import Detection, Recognition
from src.platform.capture import LowQualityCaptureConfig
from src.platform.video import VideoSourceConfig, VideoSourceType


def test_source_from_args_supports_all_requested_source_types() -> None:
    assert source_from_args(parse_args(["--camera", "2"])) == 2
    assert source_from_args(parse_args(["--stream-url", "rtsp://camera/live"])) == "rtsp://camera/live"
    assert source_from_args(parse_args(["--image-path", "input.jpg"])) == Path("input.jpg")
    assert source_from_args(parse_args(["--video-path", "input.mp4"])) == Path("input.mp4")


def test_source_from_args_uses_configured_source_when_no_cli_source() -> None:
    assert source_from_args(parse_args([])) is None


def test_model_options_override_configured_model_names() -> None:
    args = parse_args(["--model", "detector_v2", "--recognizer-model", "ocr_v2"])

    assert args.detector_model == "detector_v2"
    assert args.recognizer_model == "ocr_v2"


def test_inference_mode_can_run_recognizer_without_detector() -> None:
    assert parse_args(["--inference-mode", "recognizer"]).inference_mode == "recognizer"


def test_video_source_config_resolves_selected_env_backed_source() -> None:
    config = VideoSourceConfig(
        source_type=VideoSourceType.STREAM_URL,
        camera_index=0,
        stream_url="rtsp://camera/live",
        image_path="",
        video_path="",
        width=640,
        height=480,
        fps=30,
    )

    assert config.configured_source() == "rtsp://camera/live"


def test_fps_meter_tracks_end_to_end_inference_rate() -> None:
    meter = FpsMeter(smoothing=1.0)

    meter.update(inference_seconds=0.05)
    meter.update(inference_seconds=0.025)

    assert meter.inference_fps == pytest.approx(40.0)


def test_image_output_path_creates_annotated_name_for_directory(
    tmp_path: Path,
) -> None:
    assert _image_output_path(tmp_path, Path("input.png")) == tmp_path / "input.annotated.png"


def test_capture_category_separates_success_and_fail_samples() -> None:
    detection = Detection(0, 0.9, 0, 0, 10, 10)
    config = LowQualityCaptureConfig(
        success_min_detection_confidence=0.5,
        success_min_ocr_confidence=0.85,
        fail_max_detection_confidence=0.5,
        fail_max_ocr_confidence=0.85,
    )

    assert _capture_category([], config) is None
    assert _capture_category([Recognition(detection, "", (), 0.0)], config) == "fail"
    assert _capture_category([Recognition(detection, "wrong", (1,), 0.5)], config) == "fail"
    assert _capture_category([Recognition(detection, "good", (1,), 0.9)], config) == "success"
