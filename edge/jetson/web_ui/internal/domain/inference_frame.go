package domain

import (
	"fmt"
	"strings"
	"time"
)

type InferenceFrame struct {
	FrameID      string
	Timestamp    time.Time
	ImagePayload string
	Detections   []Detection
}

func NewInferenceFrame(frameID string, timestamp time.Time, imagePayload string, detections []Detection) (InferenceFrame, error) {
	if strings.TrimSpace(frameID) == "" {
		return InferenceFrame{}, fmt.Errorf("frame id must not be blank")
	}
	if timestamp.IsZero() {
		return InferenceFrame{}, fmt.Errorf("timestamp must not be zero")
	}
	if strings.TrimSpace(imagePayload) == "" {
		return InferenceFrame{}, fmt.Errorf("image payload must not be blank")
	}

	detectionCopy := append([]Detection(nil), detections...)

	return InferenceFrame{
		FrameID:      frameID,
		Timestamp:    timestamp.UTC(),
		ImagePayload: imagePayload,
		Detections:   detectionCopy,
	}, nil
}
