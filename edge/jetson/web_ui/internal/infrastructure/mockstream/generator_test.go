package mockstream

import (
	"net/url"
	"strings"
	"testing"
	"time"

	"web_ui/internal/domain"
)

func TestGeneratorNextReturnsValidInferenceFrame(t *testing.T) {
	now := time.Date(2026, time.June, 26, 11, 0, 0, 123000000, time.UTC)
	generator := NewGenerator(func() time.Time { return now })

	frame, err := generator.Next()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if strings.TrimSpace(frame.FrameID) == "" {
		t.Fatal("expected frame id to be populated")
	}

	if !frame.Timestamp.Equal(now) {
		t.Fatalf("unexpected timestamp: got %v want %v", frame.Timestamp, now)
	}

	if !strings.HasPrefix(frame.ImagePayload, "data:image/svg+xml;utf8,") {
		t.Fatalf("expected svg data URI, got %q", frame.ImagePayload)
	}

	svg := mustDecodeSVGPayload(t, frame.ImagePayload)
	if !strings.Contains(svg, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720">`) {
		t.Fatalf("expected svg root and viewBox, got %q", svg)
	}
	if !strings.Contains(svg, `linearGradient id="bg"`) {
		t.Fatalf("expected background gradient in svg, got %q", svg)
	}
	if strings.Contains(svg, `person 0.98`) {
		t.Fatalf("expected mock svg to avoid baked-in label annotations, got %q", svg)
	}
	if strings.Contains(svg, `stroke="#6df7c1"`) {
		t.Fatalf("expected mock svg to avoid baked-in bbox overlays, got %q", svg)
	}

	if len(frame.Detections) == 0 {
		t.Fatal("expected at least one detection")
	}

	detection := frame.Detections[0]
	if detection.Label != "person" {
		t.Fatalf("unexpected detection label: %q", detection.Label)
	}
	if detection.Score <= 0 || detection.Score > 1 {
		t.Fatalf("unexpected detection score: %v", detection.Score)
	}
	if got, want := detection.Box.X, 510; got != want {
		t.Fatalf("unexpected bbox x: got %d want %d", got, want)
	}
	if got, want := detection.Box.Y, 172; got != want {
		t.Fatalf("unexpected bbox y: got %d want %d", got, want)
	}
	if got, want := detection.Box.Width, 220; got != want {
		t.Fatalf("unexpected bbox width: got %d want %d", got, want)
	}
	if got, want := detection.Box.Height, 474; got != want {
		t.Fatalf("unexpected bbox height: got %d want %d", got, want)
	}

	for _, marker := range []string{
		`<rect x="560" y="172" width="120" height="340"`,
		`<rect x="510" y="222" width="220" height="284"`,
		`<rect x="575" y="520" width="54" height="126"`,
		`<rect x="652" y="520" width="54" height="126"`,
	} {
		if !strings.Contains(svg, marker) {
			t.Fatalf("expected svg subject marker %q in %q", marker, svg)
		}
	}
}

func TestGeneratorNextProducesDistinctFrameIDs(t *testing.T) {
	now := time.Date(2026, time.June, 26, 11, 0, 0, 0, time.UTC)
	generator := NewGenerator(func() time.Time { return now })

	first, err := generator.Next()
	if err != nil {
		t.Fatalf("expected no error from first frame, got %v", err)
	}

	second, err := generator.Next()
	if err != nil {
		t.Fatalf("expected no error from second frame, got %v", err)
	}

	if first.FrameID == second.FrameID {
		t.Fatalf("expected distinct frame ids, got %q", first.FrameID)
	}
}

func TestIsMockFrameRecognizesGeneratorFrames(t *testing.T) {
	generator := NewGenerator(func() time.Time {
		return time.Date(2026, time.June, 26, 11, 0, 0, 0, time.UTC)
	})

	frame, err := generator.Next()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if !IsMockFrame(frame) {
		t.Fatalf("expected generated frame to be recognized as mock: %+v", frame)
	}
}

func TestIsMockFrameRejectsNonMockFrames(t *testing.T) {
	frame, err := domain.NewInferenceFrame(
		"camera-1-frame-42",
		time.Date(2026, time.June, 26, 11, 0, 0, 0, time.UTC),
		"data:image/jpeg;base64,ZmFrZQ==",
		nil,
	)
	if err != nil {
		t.Fatalf("expected valid frame, got %v", err)
	}

	if IsMockFrame(frame) {
		t.Fatalf("expected non-mock frame to be rejected: %+v", frame)
	}
}

func mustDecodeSVGPayload(t *testing.T, payload string) string {
	t.Helper()

	const prefix = "data:image/svg+xml;utf8,"
	svg, err := url.PathUnescape(strings.TrimPrefix(payload, prefix))
	if err != nil {
		t.Fatalf("expected valid escaped svg payload, got %v", err)
	}

	return svg
}
