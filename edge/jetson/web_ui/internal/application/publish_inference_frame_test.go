package application

import (
	"context"
	"testing"
	"time"

	"web_ui/internal/domain"
)

type recordingBroadcaster struct {
	calls int
	frame domain.InferenceFrame
}

func (r *recordingBroadcaster) Broadcast(_ context.Context, frame domain.InferenceFrame) error {
	r.calls++
	r.frame = frame
	return nil
}

func TestPublishInferenceFrameExecuteBroadcastsFrame(t *testing.T) {
	broadcaster := &recordingBroadcaster{}
	useCase := NewPublishInferenceFrame(broadcaster)

	frame := mustInferenceFrame(t, "frame-1", "person")
	if err := useCase.Execute(context.Background(), frame); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	frame.Detections[0].Label = "changed"

	if broadcaster.calls != 1 {
		t.Fatalf("expected broadcaster to be called once, got %d", broadcaster.calls)
	}

	if broadcaster.frame.Detections[0].Label != "person" {
		t.Fatalf("expected broadcast snapshot to be isolated, got %+v", broadcaster.frame)
	}
}

func TestPublishInferenceFrameCurrentReturnsLatestSnapshot(t *testing.T) {
	broadcaster := &recordingBroadcaster{}
	useCase := NewPublishInferenceFrame(broadcaster)

	if _, ok := useCase.Current(); ok {
		t.Fatal("expected no current snapshot before execution")
	}

	frame := mustInferenceFrame(t, "frame-1", "person")
	if err := useCase.Execute(context.Background(), frame); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	got, ok := useCase.Current()
	if !ok {
		t.Fatal("expected current snapshot after execution")
	}

	if got.Detections[0].Label != "person" {
		t.Fatalf("unexpected current snapshot: got %+v want label %q", got, "person")
	}

	got.Detections[0].Label = "changed"

	again, ok := useCase.Current()
	if !ok {
		t.Fatal("expected current snapshot after mutation")
	}

	if again.Detections[0].Label != "person" {
		t.Fatalf("expected current snapshot to be isolated, got %+v", again)
	}
}

func mustInferenceFrame(t *testing.T, frameID string, label string) domain.InferenceFrame {
	t.Helper()

	detection := mustDetection(t, label)
	frame, err := domain.NewInferenceFrame(frameID, time.Unix(1, 0).UTC(), "payload", []domain.Detection{detection})
	if err != nil {
		t.Fatalf("expected valid inference frame, got %v", err)
	}

	return frame
}

func mustDetection(t *testing.T, label string) domain.Detection {
	t.Helper()

	box, err := domain.NewBoundingBox(1, 2, 3, 4)
	if err != nil {
		t.Fatalf("expected valid bounding box, got %v", err)
	}

	detection, err := domain.NewDetection(label, 0.9, box)
	if err != nil {
		t.Fatalf("expected valid detection, got %v", err)
	}

	return detection
}
