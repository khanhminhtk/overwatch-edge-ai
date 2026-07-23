package domain

import (
	"fmt"
	"strings"
	"time"
)

type Status string

const (
	StatusReceived   Status = "RECEIVED"
	StatusProcessing Status = "PROCESSING"
	StatusProcessed  Status = "PROCESSED"
	StatusFailed     Status = "FAILED"
)

type MessageIdentity struct {
	Topic         string
	PartitionID   int
	MessageOffset int64
	ConsumerGroup string
}

func (m MessageIdentity) Validate() error {
	if strings.TrimSpace(m.Topic) == "" || strings.TrimSpace(m.ConsumerGroup) == "" {
		return fmt.Errorf("message topic and consumer group must not be empty")
	}
	if m.PartitionID < 0 || m.MessageOffset < 0 {
		return fmt.Errorf("message partition and offset must be non-negative")
	}
	return nil
}

type JobRecord struct {
	RequestID       string
	MessageIdentity MessageIdentity
	EventType       string
	Payload         map[string]any
	Status          Status
	ErrorMessage    string
	ProducedAt      *time.Time
	ProcessedAt     *time.Time
	ClaimedBy       string
}

func (j *JobRecord) MarkProcessing(serverID string) error {
	if j.Status != StatusReceived {
		return fmt.Errorf("cannot mark processing from status=%s", j.Status)
	}
	if strings.TrimSpace(serverID) == "" {
		return fmt.Errorf("server ID must not be empty")
	}
	j.Status, j.ClaimedBy, j.ErrorMessage = StatusProcessing, serverID, ""
	return nil
}

func (j *JobRecord) MarkProcessed(now time.Time) error {
	if j.Status != StatusProcessing {
		return fmt.Errorf("cannot mark processed from status=%s", j.Status)
	}
	j.Status, j.ErrorMessage, j.ProcessedAt = StatusProcessed, "", &now
	return nil
}

func (j *JobRecord) MarkFailed(now time.Time, message string) error {
	if j.Status != StatusProcessing {
		return fmt.Errorf("cannot mark failed from status=%s", j.Status)
	}
	j.Status, j.ErrorMessage, j.ProcessedAt = StatusFailed, message, &now
	return nil
}
