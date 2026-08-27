package domain

import (
	"testing"
	"time"
)

func TestNewInferenceFrameRejectsBlankFrameID(t *testing.T) {
	_, err := NewInferenceFrame("", time.Unix(1, 0).UTC(), "payload", nil)
	if err == nil {
		t.Fatal("expected error for blank frame id")
	}
}

func TestNewInferenceFrameRejectsWhitespaceOnlyFrameID(t *testing.T) {
	_, err := NewInferenceFrame("   ", time.Unix(1, 0).UTC(), "payload", nil)
	if err == nil {
		t.Fatal("expected error for whitespace-only frame id")
	}
}

func TestNewInferenceFrameRejectsBlankImagePayload(t *testing.T) {
	_, err := NewInferenceFrame("frame-1", time.Unix(1, 0).UTC(), "", nil)
	if err == nil {
		t.Fatal("expected error for blank image payload")
	}
}

func TestNewInferenceFrameRejectsZeroTimestamp(t *testing.T) {
	_, err := NewInferenceFrame("frame-1", time.Time{}, "payload", nil)
	if err == nil {
		t.Fatal("expected error for zero timestamp")
	}
}

func TestNewInferenceFrameCreatesValidFrame(t *testing.T) {
	detections := []Detection{
		mustDetection(t, "person", 0.99),
	}

	frameTime := time.Date(2026, time.June, 26, 12, 34, 56, 0, time.FixedZone("UTC+7", 7*60*60))
	frame, err := NewInferenceFrame("frame-1", frameTime, "payload", detections)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if frame.FrameID != "frame-1" {
		t.Fatalf("unexpected FrameID: %q", frame.FrameID)
	}

	if got, want := frame.Timestamp, frameTime.UTC(); !got.Equal(want) {
		t.Fatalf("unexpected Timestamp: got %v want %v", got, want)
	}

	if frame.ImagePayload != "payload" {
		t.Fatalf("unexpected ImagePayload: %q", frame.ImagePayload)
	}

	if len(frame.Detections) != 1 {
		t.Fatalf("unexpected detections len: %d", len(frame.Detections))
	}

	detections[0].Label = "changed"
	if frame.Detections[0].Label != "person" {
		t.Fatalf("expected detections to be copied, got %q", frame.Detections[0].Label)
	}
}

func TestNewBoundingBoxRejectsNegativeCoordinatesAndNonPositiveDimensions(t *testing.T) {
	_, err := NewBoundingBox(-1, 0, 10, 10)
	if err == nil {
		t.Fatal("expected error for negative x")
	}

	_, err = NewBoundingBox(0, -1, 10, 10)
	if err == nil {
		t.Fatal("expected error for negative y")
	}

	_, err = NewBoundingBox(0, 0, 0, 10)
	if err == nil {
		t.Fatal("expected error for non-positive width")
	}

	_, err = NewBoundingBox(0, 0, 10, 0)
	if err == nil {
		t.Fatal("expected error for non-positive height")
	}
}

func TestNewDetectionRejectsBlankLabelAndOutOfRangeScore(t *testing.T) {
	_, err := NewDetection("", 0.5, BoundingBox{})
	if err == nil {
		t.Fatal("expected error for blank label")
	}

	_, err = NewDetection("   ", 0.5, BoundingBox{})
	if err == nil {
		t.Fatal("expected error for whitespace-only label")
	}

	_, err = NewDetection("person", -0.1, BoundingBox{})
	if err == nil {
		t.Fatal("expected error for score below 0")
	}

	_, err = NewDetection("person", 1.1, BoundingBox{})
	if err == nil {
		t.Fatal("expected error for score above 1")
	}
}

func TestNewDetectionRejectsInvalidBoundingBox(t *testing.T) {
	_, err := NewDetection("person", 0.5, BoundingBox{})
	if err == nil {
		t.Fatal("expected error for invalid bounding box")
	}
}

func mustDetection(t *testing.T, label string, score float64) Detection {
	t.Helper()

	box, err := NewBoundingBox(1, 2, 3, 4)
	if err != nil {
		t.Fatalf("unexpected bounding box error: %v", err)
	}

	detection, err := NewDetection(label, score, box)
	if err != nil {
		t.Fatalf("unexpected detection error: %v", err)
	}

	return detection
}
