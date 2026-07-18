from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from ..domain.entities import Detection, Recognition


@dataclass(frozen=True)
class TemporalFilterSettings:
    iou_threshold: float = 0.5
    max_missed_frames: int = 4
    text_history_size: int = 5
    ocr_refresh_interval: int = 30
    blank_retry_interval: int = 30
    ocr_refresh_iou_threshold: float = 0.85


@dataclass
class _Track:
    detection: Detection
    missed_frames: int = 0
    texts: deque[str] = field(default_factory=deque)
    token_ids: tuple[int, ...] = ()
    confidence: float = 0.0
    last_ocr_frame: int = -10_000
    last_ocr_detection: Detection | None = None


class TemporalFilterService:
    """Stabilize OCR overlays across adjacent video frames."""

    def __init__(
        self, settings: TemporalFilterSettings = TemporalFilterSettings()
    ) -> None:
        self._settings = settings
        self._tracks: list[_Track] = []
        self._frame_index = 0
        self._current_tracks: dict[Detection, _Track] = {}

    def start_frame(self, detections: list[Detection]) -> list[Detection]:
        """Track current boxes and return only boxes that need a new OCR request."""
        self._frame_index += 1
        unmatched_tracks = set(range(len(self._tracks)))
        self._current_tracks = {}
        to_recognize: list[Detection] = []

        for detection in detections:
            track_index = self._find_track(detection, unmatched_tracks)
            if track_index is None:
                track = _Track(
                    detection=detection,
                    texts=deque(maxlen=self._settings.text_history_size),
                )
                self._tracks.append(track)
            else:
                track = self._tracks[track_index]
                unmatched_tracks.remove(track_index)
                track.detection = detection
                track.missed_frames = 0
            self._current_tracks[detection] = track
            interval = (
                self._settings.ocr_refresh_interval
                if track.texts
                else self._settings.blank_retry_interval
            )
            crop_changed = (
                track.last_ocr_detection is not None
                and _iou(detection, track.last_ocr_detection)
                < self._settings.ocr_refresh_iou_threshold
            )
            if crop_changed or self._frame_index - track.last_ocr_frame >= interval:
                to_recognize.append(detection)

        for index, track in enumerate(self._tracks):
            if index in unmatched_tracks:
                track.missed_frames += 1
        self._tracks = [
            track
            for track in self._tracks
            if track.missed_frames <= self._settings.max_missed_frames
        ]
        return to_recognize

    def finish_frame(self, recognitions: list[Recognition]) -> list[Recognition]:
        """Merge sparse OCR responses into the active tracks for rendering."""
        for recognition in recognitions:
            track = self._current_tracks.get(recognition.detection)
            if track is None:
                continue
            track.last_ocr_frame = self._frame_index
            track.token_ids = recognition.token_ids
            track.confidence = recognition.confidence
            track.last_ocr_detection = recognition.detection
            if recognition.text:
                track.texts.append(recognition.text)
        return [
            Recognition(
                detection,
                _most_common_text(track.texts),
                track.token_ids,
                track.confidence,
            )
            for detection, track in self._current_tracks.items()
        ]

    def tracked_detections(self) -> list[Detection]:
        """Reuse the most recent boxes on detector-skipped video frames."""
        return [track.detection for track in self._tracks if track.missed_frames == 0]

    def apply(
        self, detections: list[Detection], recognitions: list[Recognition]
    ) -> list[Recognition]:
        self.start_frame(detections)
        return self.finish_frame(recognitions)

    def _find_track(self, detection: Detection, candidates: set[int]) -> int | None:
        best_index: int | None = None
        best_iou = self._settings.iou_threshold
        for index in candidates:
            candidate_iou = _iou(detection, self._tracks[index].detection)
            if candidate_iou >= best_iou:
                best_index = index
                best_iou = candidate_iou
        return best_index


def _most_common_text(texts: deque[str]) -> str:
    if not texts:
        return ""
    counts = Counter(texts)
    return max(counts, key=lambda text: (counts[text], text == texts[-1]))


def _iou(left: Detection, right: Detection) -> float:
    width = max(0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = width * height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union else 0.0
