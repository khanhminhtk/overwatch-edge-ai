package broadcast

import (
	"context"
	"sync"

	"web_ui/internal/domain"
)

type Hub struct {
	mu          sync.Mutex
	nextID      int
	hasLatest   bool
	latest      domain.InferenceFrame
	subscribers map[int]chan domain.InferenceFrame
}

func NewHub() *Hub {
	return &Hub{
		subscribers: make(map[int]chan domain.InferenceFrame),
	}
}

func (h *Hub) Broadcast(ctx context.Context, frame domain.InferenceFrame) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	snapshot := cloneFrame(frame)

	h.mu.Lock()
	defer h.mu.Unlock()

	h.latest = snapshot
	h.hasLatest = true

	for _, subscriber := range h.subscribers {
		deliverLatest(subscriber, snapshot)
	}

	return nil
}

func (h *Hub) Subscribe() (<-chan domain.InferenceFrame, func()) {
	ch := make(chan domain.InferenceFrame, 1)

	h.mu.Lock()
	id := h.nextID
	h.nextID++
	h.subscribers[id] = ch
	if h.hasLatest {
		deliverLatest(ch, h.latest)
	}
	h.mu.Unlock()

	unsubscribe := func() {
		h.mu.Lock()
		subscriber, ok := h.subscribers[id]
		if ok {
			delete(h.subscribers, id)
			close(subscriber)
		}
		h.mu.Unlock()
	}

	return ch, unsubscribe
}

func cloneFrame(frame domain.InferenceFrame) domain.InferenceFrame {
	cloned := frame
	if frame.Detections != nil {
		cloned.Detections = append([]domain.Detection(nil), frame.Detections...)
	}
	return cloned
}

func deliverLatest(subscriber chan domain.InferenceFrame, frame domain.InferenceFrame) {
	payload := cloneFrame(frame)

	select {
	case subscriber <- payload:
		return
	default:
	}

	select {
	case <-subscriber:
	default:
	}

	select {
	case subscriber <- payload:
	default:
	}
}
