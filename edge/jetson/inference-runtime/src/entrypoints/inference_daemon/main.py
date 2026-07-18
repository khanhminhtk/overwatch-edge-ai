from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time
from typing import Sequence

import cv2
import numpy as np

from src.bootstrap import SERVICE_ROOT, load_inference_config
from src.modules.inference import (
    DetectorService,
    Recognition,
    RecognizerService,
    TemporalFilterService,
    VisualizerService,
    inference_state,
)
from src.platform.triton import TritonClient
from src.platform.capture import LowQualityCaptureConfig, LowQualityFrameStore
from src.platform.video import VideoSource, VideoSourceType


class FpsMeter:
    """Exponentially smoothed end-to-end frame-processing FPS."""

    def __init__(self, smoothing: float = 0.2) -> None:
        self._smoothing = smoothing
        self.inference_fps = 0.0

    def update(self, inference_seconds: float) -> None:
        self.inference_fps = _smooth_fps(
            self.inference_fps, inference_seconds, self._smoothing
        )


def _smooth_fps(current: float, elapsed: float, smoothing: float) -> float:
    instantaneous = 1.0 / elapsed if elapsed > 0 else 0.0
    if current == 0.0:
        return instantaneous
    return (1 - smoothing) * current + smoothing * instantaneous


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, metavar="INDEX", help="camera index")
    source.add_argument("--stream-url", help="RTSP/HTTP camera stream URL")
    source.add_argument("--image-path", type=Path, help="infer one image")
    source.add_argument("--video-path", type=Path, help="infer a video file")
    parser.add_argument("--triton-url", help="override Triton gRPC endpoint")
    parser.add_argument(
        "--model",
        "--detector-model",
        dest="detector_model",
        help="override Triton detector model name",
    )
    parser.add_argument(
        "--recognizer-model", help="override Triton OCR/recognizer model name"
    )
    parser.add_argument(
        "--inference-mode",
        choices=("pipeline", "detector", "recognizer"),
        default="pipeline",
        help="pipeline: detector then OCR; detector: boxes only; recognizer: OCR full image",
    )
    parser.add_argument("--detector-interval", type=int, default=1)
    parser.add_argument("--max-frames", type=int, help="stop after this many frames")
    parser.add_argument("--output", type=Path, help="write annotated image or MP4")
    parser.add_argument("--display", action="store_true", help="show annotated frames")
    return parser.parse_args(argv)


def source_from_args(args: argparse.Namespace) -> int | str | Path | None:
    if args.camera is not None:
        return args.camera
    if args.stream_url:
        return args.stream_url
    if args.image_path:
        return args.image_path
    if args.video_path:
        return args.video_path
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.detector_interval <= 0:
        raise ValueError("--detector-interval must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")

    runtime = load_inference_config()
    detector_config = runtime.detector_config
    recognizer_config = runtime.recognizer_config
    if args.triton_url:
        detector_config = replace(detector_config, url=args.triton_url)
        recognizer_config = replace(recognizer_config, url=args.triton_url)
    if args.detector_model:
        detector_config = replace(detector_config, model_name=args.detector_model)
    if args.recognizer_model:
        recognizer_config = replace(recognizer_config, model_name=args.recognizer_model)
    detector = DetectorService(TritonClient(detector_config), runtime.detector_settings)
    recognizer = RecognizerService(TritonClient(recognizer_config), runtime.recognizer_settings)
    return _run(
        args,
        runtime.video_config,
        detector,
        recognizer,
        LowQualityFrameStore(runtime.capture_config, SERVICE_ROOT),
    )


def _run(
    args: argparse.Namespace,
    video_config,
    detector: DetectorService,
    recognizer: RecognizerService,
    low_quality_frames: LowQualityFrameStore,
) -> int:
    source = VideoSource(video_config)
    source.open(source_from_args(args))
    temporal_filter = TemporalFilterService()
    visualizer = VisualizerService()
    writer: cv2.VideoWriter | None = None
    processed = 0
    fps_meter = FpsMeter()
    last_printed_texts: tuple[str, ...] = ()
    is_image = args.image_path is not None or (
        source_from_args(args) is None
        and video_config.source_type is VideoSourceType.IMAGE_PATH
    )
    try:
        while args.max_frames is None or processed < args.max_frames:
            frame = source.read()
            if frame is None:
                break
            inference_started_at = time.perf_counter()
            if args.inference_mode == "recognizer":
                filtered_recognitions = recognizer.recognize_frame(frame)
                fresh_recognitions = filtered_recognitions
            elif args.inference_mode == "detector":
                detections = detector.detect(frame)
                filtered_recognitions = [
                    Recognition(detection, "", ()) for detection in detections
                ]
                fresh_recognitions: list[Recognition] = []
            elif (
                processed % args.detector_interval == 0
                or not temporal_filter.tracked_detections()
            ):
                detections = detector.detect(frame)
                recognitions = recognizer.recognize_detections(
                    temporal_filter.start_frame(detections)
                )
                filtered_recognitions = temporal_filter.finish_frame(recognitions)
                fresh_recognitions = recognitions
            else:
                detections = temporal_filter.tracked_detections()
                inference_state.publish(frame, detections)
                recognitions = recognizer.recognize_detections(
                    temporal_filter.start_frame(detections)
                )
                filtered_recognitions = temporal_filter.finish_frame(recognitions)
                fresh_recognitions = recognitions
            capture_category = _capture_category(
                fresh_recognitions,
                low_quality_frames.config,
            )
            if capture_category is not None:
                saved_path = low_quality_frames.save_if_distinct(
                    capture_category, frame
                )
                if saved_path is not None:
                    print(
                        f"{capture_category}_frame={saved_path.resolve()}",
                        flush=True,
                    )
            texts = tuple(item.text for item in filtered_recognitions if item.text)
            if is_image or (texts and texts != last_printed_texts):
                print(f"ocr_text={' | '.join(texts) if texts else '<empty>'}", flush=True)
            last_printed_texts = texts
            rendered = visualizer.render(frame, filtered_recognitions)
            inference_completed_at = time.perf_counter()
            fps_meter.update(inference_completed_at - inference_started_at)
            _draw_fps(rendered, source.fps, fps_meter)
            writer = _write_frame(
                writer,
                args.output,
                rendered,
                source,
                is_image,
                args.image_path,
            )
            if args.display:
                cv2.imshow("Triton inference", rendered)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            processed += 1
            if is_image:
                break
    finally:
        source.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
    print(f"processed_frames={processed}")
    return 0


def _draw_fps(frame: np.ndarray, camera_fps: float, fps_meter: FpsMeter) -> None:
    lines = (
        f"REAL FPS: {camera_fps:.1f}",
        f"INFERENCE FPS: {fps_meter.inference_fps:.1f}",
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    padding = 10
    line_height = 24
    max_width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines)
    left = max(0, frame.shape[1] - max_width - 2 * padding)
    bottom = padding + len(lines) * line_height
    cv2.rectangle(frame, (left, 0), (frame.shape[1], bottom), (0, 0, 0), -1)
    for index, line in enumerate(lines, start=1):
        cv2.putText(
            frame,
            line,
            (left + padding, index * line_height - 6),
            font,
            scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def _capture_category(
    recognitions: list[Recognition], capture_config: LowQualityCaptureConfig
) -> str | None:
    """Classify a frame only when detector and recognizer have an outcome."""
    if not recognitions:
        return None
    if any(
        recognition.detection.confidence < capture_config.fail_max_detection_confidence
        or not recognition.text
        or recognition.confidence < capture_config.fail_max_ocr_confidence
        for recognition in recognitions
    ):
        return "fail"
    if all(
        recognition.detection.confidence
        >= capture_config.success_min_detection_confidence
        and recognition.text
        and recognition.confidence >= capture_config.success_min_ocr_confidence
        for recognition in recognitions
    ):
        return "success"
    return None


def _write_frame(
    writer,
    output: Path | None,
    frame: np.ndarray,
    source: VideoSource,
    is_image: bool,
    image_input: Path | None,
):
    if output is None:
        return writer
    if is_image:
        image_output = _image_output_path(output, image_input)
        image_output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_output), frame):
            raise RuntimeError(f"cannot write output image: {image_output}")
        print(f"output={image_output.resolve()}")
        return writer
    output.parent.mkdir(parents=True, exist_ok=True)
    if writer is None:
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source.fps or 30.0,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot write output video: {output}")
    writer.write(frame)
    return writer


def _image_output_path(output: Path, image_input: Path | None) -> Path:
    if output.is_dir() or not output.suffix:
        if image_input is None:
            raise ValueError("an image output directory requires --image-path")
        extension = image_input.suffix or ".jpg"
        return output / f"{image_input.stem}.annotated{extension}"
    return output


if __name__ == "__main__":
    raise SystemExit(main())
