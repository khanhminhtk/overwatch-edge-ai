#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from src.modules.inference import (  # noqa: E402
    DetectorService,
    RecognizerService,
    TemporalFilterService,
    VisualizerService,
    inference_state,
)
from src.bootstrap import load_inference_config  # noqa: E402
from src.platform.triton import TritonClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--url", default="localhost:8001")
    parser.add_argument("--detector-interval", type=int, default=3)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_config = load_inference_config()
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot read video: {args.input}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start_frame < 0 or args.start_frame >= total_frames:
        raise ValueError("--start-frame must be within the input video")
    end_frame = min(args.end_frame or total_frames, total_frames)
    if end_frame <= args.start_frame:
        raise ValueError("--end-frame must be after --start-frame")
    if args.detector_interval <= 0:
        raise ValueError("--detector-interval must be positive")
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create output video: {args.output}")

    detector = DetectorService(
        TritonClient(replace(runtime_config.detector_config, url=args.url)),
        runtime_config.detector_settings,
    )
    recognizer = RecognizerService(
        TritonClient(replace(runtime_config.recognizer_config, url=args.url)),
        runtime_config.recognizer_settings,
    )
    temporal_filter = TemporalFilterService()
    visualizer = VisualizerService()
    frame_index = args.start_frame
    detector_seconds = 0.0
    end_to_end_seconds = 0.0
    total_boxes = 0
    nonblank_ocr = 0
    ocr_requests = 0
    try:
        while frame_index < end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            started = time.perf_counter()
            run_detector = (
                frame_index - args.start_frame
            ) % args.detector_interval == 0
            if run_detector or not temporal_filter.tracked_detections():
                detections = detector.detect(frame)
            else:
                detections = temporal_filter.tracked_detections()
                inference_state.publish(frame, detections)
            detector_done = time.perf_counter()
            ocr_detections = temporal_filter.start_frame(detections)
            recognitions = recognizer.recognize_detections(ocr_detections)
            filtered_recognitions = temporal_filter.finish_frame(recognitions)
            rendered = visualizer.render(frame, filtered_recognitions)
            completed = time.perf_counter()
            if rendered is None:
                raise RuntimeError(
                    "visualizer did not receive the current inference snapshots"
                )
            writer.write(rendered)
            detector_seconds += detector_done - started
            end_to_end_seconds += completed - started
            total_boxes += len(detections)
            ocr_requests += len(ocr_detections)
            nonblank_ocr += sum(bool(item.text) for item in filtered_recognitions)
            frame_index += 1
            if frame_index % 100 == 0 or frame_index == end_frame:
                print(f"processed={frame_index}/{end_frame}", flush=True)
    finally:
        capture.release()
        writer.release()

    processed_frames = frame_index - args.start_frame
    print(f"output={args.output.resolve()}")
    print(f"processed_frames={processed_frames} source_fps={fps:.2f}")
    print(f"detector_fps={processed_frames / detector_seconds:.2f}")
    print(f"end_to_end_fps={processed_frames / end_to_end_seconds:.2f}")
    print(
        f"boxes={total_boxes} ocr_requests={ocr_requests} nonblank_ocr={nonblank_ocr}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
