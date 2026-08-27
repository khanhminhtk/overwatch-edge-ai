package application

import (
	"context"
	"sync"

	"web_ui/internal/domain"
)

type PublishInferenceFrame struct {
	broadcaster InferenceFrameBroadcaster

	mu         sync.RWMutex
	current    domain.InferenceFrame
	hasCurrent bool
}

func NewPublishInferenceFrame(broadcaster InferenceFrameBroadcaster) *PublishInferenceFrame {
	return &PublishInferenceFrame{
		broadcaster: broadcaster,
	}
}

func (u *PublishInferenceFrame) Execute(ctx context.Context, frame domain.InferenceFrame) error {
	snapshot := cloneInferenceFrame(frame)

	u.mu.Lock()
	u.current = snapshot
	u.hasCurrent = true
	u.mu.Unlock()

	if u.broadcaster == nil {
		return nil
	}

	return u.broadcaster.Broadcast(ctx, snapshot)
}

func (u *PublishInferenceFrame) Current() (domain.InferenceFrame, bool) {
	u.mu.RLock()
	defer u.mu.RUnlock()

	if !u.hasCurrent {
		return domain.InferenceFrame{}, false
	}

	return cloneInferenceFrame(u.current), true
}

func cloneInferenceFrame(frame domain.InferenceFrame) domain.InferenceFrame {
	clonedDetections := make([]domain.Detection, len(frame.Detections))
	copy(clonedDetections, frame.Detections)

	frame.Detections = clonedDetections
	return frame
}
