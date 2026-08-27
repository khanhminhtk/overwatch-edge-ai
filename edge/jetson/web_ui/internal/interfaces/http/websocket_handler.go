package httpui

import (
	"net/http"
	"time"

	"github.com/gorilla/websocket"

	"web_ui/internal/domain"
)

type websocketConn interface {
	ReadMessage() (messageType int, p []byte, err error)
	WriteJSON(v any) error
	WriteControl(messageType int, data []byte, deadline time.Time) error
	SetReadDeadline(t time.Time) error
	SetWriteDeadline(t time.Time) error
	SetPongHandler(h func(appData string) error)
}

type websocketHandler struct {
	hub      frameSubscriptionHub
	upgrader websocket.Upgrader
}

const (
	websocketHeartbeatInterval = 15 * time.Second
	websocketHeartbeatTimeout  = 30 * time.Second
	websocketWriteTimeout      = 5 * time.Second
)

func newWebsocketHandler(hub frameSubscriptionHub) http.Handler {
	return &websocketHandler{
		hub: hub,
		upgrader: websocket.Upgrader{
			CheckOrigin: func(*http.Request) bool {
				return true
			},
		},
	}
}

func (h *websocketHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if h.hub == nil {
		http.Error(w, http.StatusText(http.StatusServiceUnavailable), http.StatusServiceUnavailable)
		return
	}

	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	defer conn.Close()

	frames, unsubscribe := h.hub.Subscribe()
	heartbeatTicker := time.NewTicker(websocketHeartbeatInterval)
	defer heartbeatTicker.Stop()

	streamFrames(conn, frames, heartbeatTicker.C, time.Now, unsubscribe)
}

func streamFrames(conn websocketConn, frames <-chan domain.InferenceFrame, heartbeat <-chan time.Time, now func() time.Time, unsubscribe func()) {
	defer unsubscribe()

	_ = conn.SetReadDeadline(now().Add(websocketHeartbeatTimeout))
	conn.SetPongHandler(func(string) error {
		return conn.SetReadDeadline(now().Add(websocketHeartbeatTimeout))
	})

	readErrCh := make(chan error, 1)
	go func() {
		for {
			_, _, err := conn.ReadMessage()
			if err != nil {
				readErrCh <- err
				return
			}
		}
	}()

	for {
		select {
		case _, ok := <-readErrCh:
			if !ok {
				return
			}
			return
		case tick, ok := <-heartbeat:
			if !ok {
				return
			}
			if err := conn.WriteControl(websocket.PingMessage, nil, tick.Add(websocketWriteTimeout)); err != nil {
				return
			}
		case frame, ok := <-frames:
			if !ok {
				return
			}
			if err := conn.SetWriteDeadline(now().Add(websocketWriteTimeout)); err != nil {
				return
			}
			if err := conn.WriteJSON(newIngestRequestFromFrame(frame)); err != nil {
				return
			}
		}
	}
}
