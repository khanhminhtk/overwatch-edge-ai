package domain

import "time"

type DetectionPayload struct {
	CameraID string
	ImageBase64 string //image encode
	Confidence float32
	Text string
	Timestamp int64
}

type ShippingLog struct {
	ID string
	CameraID string
	Text string
	Confidence float32
	ImagePath string
	CreatAt time.Time
	IsHard bool
}