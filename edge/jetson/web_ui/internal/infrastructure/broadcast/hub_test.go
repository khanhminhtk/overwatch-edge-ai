package broadcast

import (
	"context"
	"testing"
	"time"

	"web_ui/internal/domain"
)

func TestHubBroadcastDeliversFrameToSubscriber(t *testing.T) {
	hub := NewHub()
	sub, cancel := hub.Subscribe()
	defer cancel()

	frame := mustFrame(t, "frame-1")
	if err := hub.Broadcast(context.Background(), frame); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	select {
	case got := <-sub:
		if got.FrameID != frame.FrameID {
			t.Fatalf("expected frame %q, got %q", frame.FrameID, got.FrameID)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for frame")
	}
}

func TestHubSubscribeDeliversLatestFrameImmediatelyAfterBroadcast(t *testing.T) {
	hub := NewHub()

	want := mustFrame(t, "frame-1")
	if err := hub.Broadcast(context.Background(), want); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	sub, cancel := hub.Subscribe()
	defer cancel()

	select {
	case got := <-sub:
		if got.FrameID != want.FrameID {
			t.Fatalf("expected latest frame %q, got %q", want.FrameID, got.FrameID)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for latest frame")
	}
}

func TestHubBroadcastFanOutsToMultipleSubscribers(t *testing.T) {
	hub := NewHub()
	sub1, cancel1 := hub.Subscribe()
	defer cancel1()
	sub2, cancel2 := hub.Subscribe()
	defer cancel2()

	want := mustFrame(t, "frame-1")
	if err := hub.Broadcast(context.Background(), want); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	for i, sub := range []<-chan domain.InferenceFrame{sub1, sub2} {
		select {
		case got := <-sub:
			if got.FrameID != want.FrameID {
				t.Fatalf("subscriber %d expected frame %q, got %q", i+1, want.FrameID, got.FrameID)
			}
		case <-time.After(time.Second):
			t.Fatalf("subscriber %d timed out waiting for frame", i+1)
		}
	}
}

func TestHubBroadcastKeepsLatestFrameWhenSubscriberIsSlow(t *testing.T) {
	hub := NewHub()
	sub, cancel := hub.Subscribe()
	defer cancel()

	first := mustFrame(t, "frame-1")
	second := mustFrame(t, "frame-2")

	if err := hub.Broadcast(context.Background(), first); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if err := hub.Broadcast(context.Background(), second); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	select {
	case got := <-sub:
		if got.FrameID != second.FrameID {
			t.Fatalf("expected latest frame %q, got %q", second.FrameID, got.FrameID)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for frame")
	}
}

func TestHubBroadcastReturnsContextErrorWhenCanceled(t *testing.T) {
	hub := NewHub()
	sub, cancel := hub.Subscribe()
	defer cancel()

	ctx, cancelCtx := context.WithCancel(context.Background())
	cancelCtx()

	if err := hub.Broadcast(ctx, mustFrame(t, "frame-1")); err != context.Canceled {
		t.Fatalf("expected context.Canceled, got %v", err)
	}

	select {
	case got := <-sub:
		t.Fatalf("expected no delivery on canceled broadcast, got %+v", got)
	case <-time.After(50 * time.Millisecond):
	}
}

func TestHubUnsubscribeIsSafeAndStopsDelivery(t *testing.T) {
	hub := NewHub()
	sub, cancel := hub.Subscribe()

	cancel()

	select {
	case _, ok := <-sub:
		if ok {
			t.Fatal("expected subscriber channel to be closed after unsubscribe")
		}
	default:
		t.Fatal("expected subscriber channel to be closed after unsubscribe")
	}

	if err := hub.Broadcast(context.Background(), mustFrame(t, "frame-1")); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
}

func TestHubBroadcastClonesDeliveredSnapshot(t *testing.T) {
	hub := NewHub()
	sub, cancel := hub.Subscribe()
	defer cancel()

	frame := mustFrame(t, "frame-1")

	if err := hub.Broadcast(context.Background(), frame); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	frame.Detections[0].Label = "caller-mutated-again"

	select {
	case got := <-sub:
		if got.Detections[0].Label != "person" {
			t.Fatalf("expected isolated snapshot label %q, got %q", "person", got.Detections[0].Label)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for frame")
	}
}

func mustFrame(t *testing.T, frameID string) domain.InferenceFrame {
	t.Helper()

	detection := mustDetection(t, "person")
	frame, err := domain.NewInferenceFrame(frameID, time.Unix(1, 0).UTC(), "payload", []domain.Detection{detection})
	if err != nil {
		t.Fatalf("expected valid frame, got %v", err)
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
