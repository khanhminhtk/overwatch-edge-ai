package domain

import (
	"fmt"
	"strings"
)

type Command struct {
	RequestID string         `json:"request_id"`
	EventType string         `json:"event_type"`
	Payload   map[string]any `json:"payload"`
}

func (c Command) Validate() error {
	if strings.TrimSpace(c.RequestID) == "" || strings.TrimSpace(c.EventType) == "" {
		return fmt.Errorf("request_id and event_type are required")
	}
	if c.Payload == nil {
		return fmt.Errorf("payload is required")
	}
	return nil
}

type ResultEvent struct {
	RequestID string         `json:"request_id"`
	JobType   string         `json:"job_type"`
	EventType string         `json:"event_type"`
	Status    string         `json:"status"`
	Payload   map[string]any `json:"payload"`
	Result    map[string]any `json:"result"`
}

type OutgoingCommand struct {
	Topic   string
	Command Command
}
