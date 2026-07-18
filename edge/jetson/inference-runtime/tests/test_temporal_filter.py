from src.modules.inference import Detection, Recognition, TemporalFilterService


def test_temporal_filter_keeps_majority_ocr_text_for_matching_boxes() -> None:
    service = TemporalFilterService()
    first = Detection(0, 0.9, 10, 10, 50, 30)
    second = Detection(0, 0.9, 11, 10, 51, 30)
    third = Detection(0, 0.9, 12, 10, 52, 30)

    service.apply([first], [Recognition(first, "KHÁI", (1,))])
    service.apply([second], [Recognition(second, "KHẢI", (2,))])
    filtered = service.apply([third], [Recognition(third, "KHÁI", (1,))])

    assert filtered[0].text == "KHÁI"


def test_temporal_filter_drops_tracks_after_configured_misses() -> None:
    service = TemporalFilterService()
    detection = Detection(0, 0.9, 10, 10, 50, 30)

    service.apply([detection], [Recognition(detection, "KHÁI", (1,))])
    for _ in range(5):
        service.apply([], [])

    next_detection = Detection(0, 0.9, 10, 10, 50, 30)
    filtered = service.apply([next_detection], [Recognition(next_detection, "", ())])

    assert filtered[0].text == ""


def test_temporal_filter_reuses_cached_text_without_another_ocr_request() -> None:
    service = TemporalFilterService()
    detection = Detection(0, 0.9, 10, 10, 50, 30)

    assert service.start_frame([detection]) == [detection]
    service.finish_frame([Recognition(detection, "KHÁI", (1,))])
    moved = Detection(0, 0.9, 11, 10, 51, 30)

    assert service.start_frame([moved]) == []
    filtered = service.finish_frame([])

    assert filtered[0].text == "KHÁI"
