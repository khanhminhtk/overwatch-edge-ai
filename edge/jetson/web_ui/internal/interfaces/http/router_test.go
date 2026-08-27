package httpui

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"web_ui/internal/domain"
)

func TestNewRouterHealthzReturnsOK(t *testing.T) {
	router := NewRouter(nil, nil, nil)

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
}

func TestNewRouterRootUsesInjectedStaticHandler(t *testing.T) {
	staticHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("static ok"))
	})

	router := NewRouter(staticHandler, nil, nil)

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}

	if got := rec.Body.String(); got != "static ok" {
		t.Fatalf("expected body %q, got %q", "static ok", got)
	}
}

func TestNewRouterIngestReturnsBadRequestForInvalidJSON(t *testing.T) {
	router := NewRouter(nil, &recordingFramePublisher{}, nil)

	req := httptest.NewRequest(http.MethodPost, "/ingest", bytes.NewBufferString("{"))
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d", http.StatusBadRequest, rec.Code)
	}
}

func TestNewRouterIngestReturnsBadRequestForUnknownJSONField(t *testing.T) {
	router := NewRouter(nil, &recordingFramePublisher{}, nil)

	req := httptest.NewRequest(
		http.MethodPost,
		"/ingest",
		bytes.NewBufferString(`{"frame_id":"frame-1","timestamp":"1970-01-01T00:00:01Z","image_payload":"payload","unexpected":true}`),
	)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d", http.StatusBadRequest, rec.Code)
	}
}

func TestNewRouterIngestReturnsBadRequestForInvalidDomainPayload(t *testing.T) {
	router := NewRouter(nil, &recordingFramePublisher{}, nil)

	body := ingestRequest{
		FrameID:      "",
		Timestamp:    time.Unix(1, 0).UTC().Format(time.RFC3339Nano),
		ImagePayload: "payload",
	}

	req := httptest.NewRequest(http.MethodPost, "/ingest", mustJSONBody(t, body))
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d", http.StatusBadRequest, rec.Code)
	}
}

func TestNewRouterIngestReturnsServiceUnavailableWhenPublisherUnavailable(t *testing.T) {
	router := NewRouter(nil, nil, nil)

	body := ingestRequest{
		FrameID:      "frame-1",
		Timestamp:    time.Unix(1, 0).UTC().Format(time.RFC3339Nano),
		ImagePayload: "payload",
	}

	req := httptest.NewRequest(http.MethodPost, "/ingest", mustJSONBody(t, body))
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status %d, got %d", http.StatusServiceUnavailable, rec.Code)
	}
}

func TestNewRouterIngestPublishesFrameAndReturnsAccepted(t *testing.T) {
	publisher := &recordingFramePublisher{}
	router := NewRouter(nil, publisher, nil)

	body := ingestRequest{
		FrameID:      "frame-1",
		Timestamp:    time.Unix(1, 0).UTC().Format(time.RFC3339Nano),
		ImagePayload: "payload",
		Detections: []ingestDetectionRequest{
			{
				Label: "person",
				Score: 0.9,
				BBox: ingestBoundingBoxRequest{
					X:      1,
					Y:      2,
					Width:  3,
					Height: 4,
				},
			},
		},
	}

	req := httptest.NewRequest(http.MethodPost, "/ingest", mustJSONBody(t, body))
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusAccepted {
		t.Fatalf("expected status %d, got %d", http.StatusAccepted, rec.Code)
	}

	if publisher.calls != 1 {
		t.Fatalf("expected publisher to be called once, got %d", publisher.calls)
	}

	if publisher.frame.FrameID != "frame-1" {
		t.Fatalf("unexpected frame published: %+v", publisher.frame)
	}
}

func TestNewRouterWebsocketReturnsServiceUnavailableWhenHubUnavailable(t *testing.T) {
	router := NewRouter(nil, nil, nil)

	req := httptest.NewRequest(http.MethodGet, "/ws", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status %d, got %d", http.StatusServiceUnavailable, rec.Code)
	}
}

type recordingFramePublisher struct {
	calls int
	frame domain.InferenceFrame
	err   error
}

func (r *recordingFramePublisher) Execute(_ context.Context, frame domain.InferenceFrame) error {
	r.calls++
	r.frame = frame
	return r.err
}

type stubFrameSubscriptionHub struct {
	ch          <-chan domain.InferenceFrame
	unsubscribe func()
	err         error
}

func (s *stubFrameSubscriptionHub) Subscribe() (<-chan domain.InferenceFrame, func()) {
	if s.unsubscribe == nil {
		s.unsubscribe = func() {}
	}
	return s.ch, s.unsubscribe
}

func mustJSONBody(t *testing.T, payload any) io.Reader {
	t.Helper()

	buf := &bytes.Buffer{}
	if err := json.NewEncoder(buf).Encode(payload); err != nil {
		t.Fatalf("expected JSON encoding to succeed, got %v", err)
	}

	return buf
}

func TestNewRouterIngestReturnsServiceUnavailableWhenPublisherFails(t *testing.T) {
	publisher := &recordingFramePublisher{err: errors.New("publisher unavailable")}
	router := NewRouter(nil, publisher, nil)

	body := ingestRequest{
		FrameID:      "frame-1",
		Timestamp:    time.Unix(1, 0).UTC().Format(time.RFC3339Nano),
		ImagePayload: "payload",
	}

	req := httptest.NewRequest(http.MethodPost, "/ingest", mustJSONBody(t, body))
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status %d, got %d", http.StatusServiceUnavailable, rec.Code)
	}
}

func TestStreamFramesUnsubscribesWhenClientDisconnectsWhileIdle(t *testing.T) {
	frames := make(chan domain.InferenceFrame)
	disconnected := make(chan struct{})
	unsubscribed := make(chan struct{}, 1)

	conn := &stubWebsocketConn{
		readMessageFunc: func() (int, []byte, error) {
			<-disconnected
			return 0, nil, errors.New("client disconnected")
		},
	}

	done := make(chan struct{})
	go func() {
		streamFrames(conn, frames, make(chan time.Time), time.Now, func() {
			unsubscribed <- struct{}{}
		})
		close(done)
	}()

	close(disconnected)

	select {
	case <-unsubscribed:
	case <-time.After(time.Second):
		t.Fatal("expected unsubscribe to be called after client disconnect")
	}

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("expected streamFrames to exit after client disconnect")
	}
}

func TestStreamFramesSendsHeartbeatWhileIdle(t *testing.T) {
	frames := make(chan domain.InferenceFrame)
	heartbeat := make(chan time.Time, 1)

	conn := &stubWebsocketConn{
		readMessageFunc: func() (int, []byte, error) {
			select {}
		},
	}

	done := make(chan struct{})
	go func() {
		streamFrames(conn, frames, heartbeat, time.Now, func() {})
		close(done)
	}()

	heartbeat <- time.Unix(1, 0)

	deadline := time.After(time.Second)
	for {
		if conn.pingCalls.Load() > 0 {
			close(frames)
			break
		}

		select {
		case <-deadline:
			t.Fatal("expected heartbeat ping while idle")
		default:
			time.Sleep(time.Millisecond)
		}
	}

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("expected streamFrames to stop after frames channel closes")
	}
}

func TestStreamFramesUnsubscribesWhenHeartbeatPingFails(t *testing.T) {
	frames := make(chan domain.InferenceFrame)
	heartbeat := make(chan time.Time, 1)
	unsubscribed := make(chan struct{}, 1)

	conn := &stubWebsocketConn{
		readMessageFunc: func() (int, []byte, error) {
			select {}
		},
		writeControlFunc: func(messageType int, data []byte, deadline time.Time) error {
			return errors.New("ping failed")
		},
	}

	done := make(chan struct{})
	go func() {
		streamFrames(conn, frames, heartbeat, time.Now, func() {
			unsubscribed <- struct{}{}
		})
		close(done)
	}()

	heartbeat <- time.Unix(1, 0)

	select {
	case <-unsubscribed:
	case <-time.After(time.Second):
		t.Fatal("expected unsubscribe after heartbeat failure")
	}

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("expected streamFrames to exit after heartbeat failure")
	}
}

func TestStreamFramesConfiguresReadDeadlineAndPongLifecycle(t *testing.T) {
	frames := make(chan domain.InferenceFrame)
	heartbeat := make(chan time.Time)
	base := time.Unix(100, 0)

	conn := &stubWebsocketConn{
		readMessageFunc: func() (int, []byte, error) {
			select {}
		},
	}

	done := make(chan struct{})
	go func() {
		streamFrames(conn, frames, heartbeat, func() time.Time { return base }, func() {})
		close(done)
	}()

	deadline := time.After(time.Second)
	for {
		if got := conn.lastReadDeadline.Load(); got != nil {
			if got.(time.Time).Equal(base.Add(websocketHeartbeatTimeout)) {
				break
			}
		}

		select {
		case <-deadline:
			t.Fatal("expected initial read deadline to be set")
		default:
			time.Sleep(time.Millisecond)
		}
	}

	if conn.pongHandler == nil {
		t.Fatal("expected pong handler to be installed")
	}

	if err := conn.pongHandler("pong"); err != nil {
		t.Fatalf("expected pong handler to succeed, got %v", err)
	}

	if got := conn.lastReadDeadline.Load().(time.Time); !got.Equal(base.Add(websocketHeartbeatTimeout)) {
		t.Fatalf("unexpected read deadline after pong: %v", got)
	}

	close(frames)

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("expected streamFrames to stop after frames channel closes")
	}
}

func TestStreamFramesSetsWriteDeadlineBeforeWritingFrame(t *testing.T) {
	frames := make(chan domain.InferenceFrame, 1)
	heartbeat := make(chan time.Time)
	base := time.Unix(200, 0)
	frame := mustDomainFrame(t)

	conn := &stubWebsocketConn{
		readMessageFunc: func() (int, []byte, error) {
			select {}
		},
		writeJSONFunc: func(v any) error {
			payload, ok := v.(ingestRequest)
			if !ok {
				t.Fatalf("expected ingestRequest payload, got %T", v)
			}
			if payload.FrameID != frame.FrameID {
				t.Fatalf("expected frame id %q, got %q", frame.FrameID, payload.FrameID)
			}
			return nil
		},
	}

	done := make(chan struct{})
	go func() {
		streamFrames(conn, frames, heartbeat, func() time.Time { return base }, func() {})
		close(done)
	}()

	frames <- frame

	deadline := time.After(time.Second)
	for {
		if conn.writeJSONCalls > 0 {
			break
		}

		select {
		case <-deadline:
			t.Fatal("expected frame write to occur")
		default:
			time.Sleep(time.Millisecond)
		}
	}

	if got := conn.lastWriteDeadline.Load(); got == nil {
		t.Fatal("expected write deadline to be set before JSON write")
	} else if !got.(time.Time).Equal(base.Add(websocketWriteTimeout)) {
		t.Fatalf("unexpected write deadline: got %v want %v", got, base.Add(websocketWriteTimeout))
	}

	close(frames)

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("expected streamFrames to stop after frames channel closes")
	}
}

func TestStreamFramesUnsubscribesWhenSettingWriteDeadlineFails(t *testing.T) {
	frames := make(chan domain.InferenceFrame, 1)
	heartbeat := make(chan time.Time)
	unsubscribed := make(chan struct{}, 1)

	conn := &stubWebsocketConn{
		readMessageFunc: func() (int, []byte, error) {
			select {}
		},
		setWriteDeadlineFunc: func(time.Time) error {
			return errors.New("write deadline failed")
		},
	}

	done := make(chan struct{})
	go func() {
		streamFrames(conn, frames, heartbeat, time.Now, func() {
			unsubscribed <- struct{}{}
		})
		close(done)
	}()

	frames <- mustDomainFrame(t)

	select {
	case <-unsubscribed:
	case <-time.After(time.Second):
		t.Fatal("expected unsubscribe after write deadline failure")
	}

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("expected streamFrames to exit after write deadline failure")
	}
}

func TestDecodeIngestRequestRejectsUnknownField(t *testing.T) {
	_, err := decodeIngestRequest(bytes.NewBufferString(`{"frame_id":"frame-1","timestamp":"1970-01-01T00:00:01Z","image_payload":"payload","unknown":"x"}`))
	if err == nil {
		t.Fatal("expected unknown field decode error")
	}
}

func TestNewIngestRequestFromFramePreservesJSONContractShape(t *testing.T) {
	frame := mustDomainFrame(t)

	payload := newIngestRequestFromFrame(frame)

	if payload.FrameID != frame.FrameID {
		t.Fatalf("expected frame id %q, got %q", frame.FrameID, payload.FrameID)
	}
	if payload.Timestamp != frame.Timestamp.UTC().Format(time.RFC3339Nano) {
		t.Fatalf("unexpected timestamp: got %q", payload.Timestamp)
	}
	if payload.ImagePayload != frame.ImagePayload {
		t.Fatalf("unexpected image payload: got %q", payload.ImagePayload)
	}
	if len(payload.Detections) != 1 {
		t.Fatalf("expected 1 detection, got %d", len(payload.Detections))
	}
	if payload.Detections[0].BBox.Width != frame.Detections[0].Box.Width {
		t.Fatalf("unexpected bbox width: got %d", payload.Detections[0].BBox.Width)
	}
}

type stubWebsocketConn struct {
	mu                   sync.Mutex
	writeJSONCalls       int
	pingCalls            atomic.Int32
	lastReadDeadline     atomic.Value
	lastWriteDeadline    atomic.Value
	readMessageFunc      func() (int, []byte, error)
	writeJSONFunc        func(v any) error
	writeControlFunc     func(messageType int, data []byte, deadline time.Time) error
	setWriteDeadlineFunc func(time.Time) error
	pongHandler          func(string) error
}

func (s *stubWebsocketConn) ReadMessage() (int, []byte, error) {
	if s.readMessageFunc != nil {
		return s.readMessageFunc()
	}
	return 0, nil, nil
}

func (s *stubWebsocketConn) WriteJSON(v any) error {
	s.mu.Lock()
	s.writeJSONCalls++
	s.mu.Unlock()
	if s.writeJSONFunc != nil {
		return s.writeJSONFunc(v)
	}
	return nil
}

func (s *stubWebsocketConn) WriteControl(messageType int, data []byte, deadline time.Time) error {
	s.pingCalls.Add(1)
	if s.writeControlFunc != nil {
		return s.writeControlFunc(messageType, data, deadline)
	}
	return nil
}

func (s *stubWebsocketConn) SetReadDeadline(t time.Time) error {
	s.lastReadDeadline.Store(t)
	return nil
}

func (s *stubWebsocketConn) SetWriteDeadline(t time.Time) error {
	s.lastWriteDeadline.Store(t)
	if s.setWriteDeadlineFunc != nil {
		return s.setWriteDeadlineFunc(t)
	}
	return nil
}

func (s *stubWebsocketConn) SetPongHandler(h func(appData string) error) {
	s.pongHandler = h
}

func mustDomainFrame(t *testing.T) domain.InferenceFrame {
	t.Helper()

	box, err := domain.NewBoundingBox(1, 2, 3, 4)
	if err != nil {
		t.Fatalf("expected valid bounding box, got %v", err)
	}

	detection, err := domain.NewDetection("person", 0.9, box)
	if err != nil {
		t.Fatalf("expected valid detection, got %v", err)
	}

	frame, err := domain.NewInferenceFrame("frame-1", time.Unix(1, 0).UTC(), "payload", []domain.Detection{detection})
	if err != nil {
		t.Fatalf("expected valid frame, got %v", err)
	}

	return frame
}
