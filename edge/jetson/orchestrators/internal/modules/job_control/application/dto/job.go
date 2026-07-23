package dto

import "orchestrator/internal/modules/job_control/domain"

type ConsumeEventCommand struct {
	RequestID       string
	EventType       string
	SchemaName      string
	SchemaVersion   string
	MessageKey      *string
	MessageIdentity domain.MessageIdentity
	Payload         map[string]any
}

type ClaimedJob struct {
	RequestID string
	EventType string
	Payload   map[string]any
	Status    domain.Status
}

type JobResult struct {
	RequestID    string
	Success      bool
	ErrorMessage string
	Result       map[string]any
}
