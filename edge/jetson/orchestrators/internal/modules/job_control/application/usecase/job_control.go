package usecase

import (
	"context"
	"fmt"
	"strings"
	"time"

	"orchestrator/internal/modules/job_control/application/dto"
	"orchestrator/internal/modules/job_control/application/ports"
	"orchestrator/internal/modules/job_control/domain"
)

type Logger interface {
	Info(string, ...any)
	Warn(string, ...any)
	Error(string, ...any)
}

type IngestJob struct {
	repository ports.JobRecordRepository
	logger     Logger
	expected   map[string]struct{}
}

func NewIngestJob(repository ports.JobRecordRepository, logger Logger, expectedEventTypes []string) (*IngestJob, error) {
	if repository == nil || logger == nil {
		return nil, fmt.Errorf("repository and logger must not be nil")
	}

	expected := make(map[string]struct{}, len(expectedEventTypes))
	for _, eventType := range expectedEventTypes {
		if eventType = strings.TrimSpace(eventType); eventType != "" {
			expected[eventType] = struct{}{}
		}
	}

	if len(expected) == 0 {
		return nil, fmt.Errorf("at least one expected event type is required")
	}

	return &IngestJob{repository: repository, logger: logger, expected: expected}, nil
}
func (u *IngestJob) Execute(ctx context.Context, command dto.ConsumeEventCommand) (bool, error) {
	if _, ok := u.expected[command.EventType]; !ok {
		return false, fmt.Errorf("unexpected event type %q", command.EventType)
	}

	if strings.TrimSpace(command.RequestID) == "" {
		return false, fmt.Errorf("request ID must not be empty")
	}

	if err := command.MessageIdentity.Validate(); err != nil {
		return false, err
	}

	now := time.Now()
	created, err := u.repository.CreateIfNotExists(ctx, domain.JobRecord{RequestID: command.RequestID, MessageIdentity: command.MessageIdentity, EventType: command.EventType, Payload: command.Payload, Status: domain.StatusReceived, ProducedAt: &now}, command.SchemaName, command.SchemaVersion, command.MessageKey)
	
	if err != nil {
		return false, err
	}
	
	u.logger.Info("[JOB_EVENT_INGESTED]", "request_id", command.RequestID, "event_type", command.EventType, "created", created)
	return created, nil
}

type ProcessNextJob struct {
	claim     ports.JobClaimRepository
	handler   ports.JobHandler
	records   ports.JobRecordRepository
	publisher ports.SuccessEventPublisher
	logger    Logger
	jobName   string
	inflight  *dto.ClaimedJob
}

func NewProcessNextJob(claim ports.JobClaimRepository, handler ports.JobHandler, records ports.JobRecordRepository, publisher ports.SuccessEventPublisher, logger Logger, jobName string) (*ProcessNextJob, error) {
	if claim == nil || handler == nil || records == nil || logger == nil {
		return nil, fmt.Errorf("claim repository, handler, record repository, and logger must not be nil")
	}
	
	return &ProcessNextJob{claim: claim, handler: handler, records: records, publisher: publisher, logger: logger, jobName: jobName}, nil
}
func (u *ProcessNextJob) Execute(ctx context.Context, serverID string) (*dto.JobResult, error) {
	job, err := u.claim.ClaimNextPendingJob(ctx, serverID)
	if err != nil || job == nil {
		return nil, err
	}

	u.inflight = job
	defer func() { u.inflight = nil }()
	started := time.Now()
	result := u.handler.Handle(ctx, *job)
	if result.RequestID == "" {
		result.RequestID = job.RequestID
	}

	u.logger.Info("[PROCESS_NEXT_JOB_HANDLER_COMPLETED]", "request_id", result.RequestID, "success", result.Success, "duration_ms", time.Since(started).Milliseconds())
	if result.Success {
		if err := u.persist(ctx, result.RequestID, "PROCESSED", func() (bool, error) { return u.records.MarkProcessedByRequestID(ctx, result.RequestID) }); err != nil {
			return nil, err
		}
		if u.publisher != nil {
			if err := u.publisher.PublishSuccess(ctx, u.jobName, serverID, *job, result.Result); err != nil {
				u.logger.Error("[JOB_SUCCESS_EVENT_PUBLISH_FAILED]", "request_id", job.RequestID, "error", err)
			}
		}
		return &result, nil
	}

	if result.ErrorMessage == "" {
		result.ErrorMessage = "unknown job error"
	}

	if err := u.persist(ctx, result.RequestID, "FAILED", func() (bool, error) {
		return u.records.MarkFailedByRequestID(ctx, result.RequestID, result.ErrorMessage)
	}); err != nil {
		return nil, err
	}
	return &result, nil
}

func (u *ProcessNextJob) FailInflightJob(ctx context.Context, reason string) error {
	if u.inflight == nil {
		return nil
	}
	requestID := u.inflight.RequestID
	return u.persist(ctx, requestID, "FAILED", func() (bool, error) { return u.records.MarkFailedByRequestID(ctx, requestID, reason) })
}

func (u *ProcessNextJob) persist(ctx context.Context, requestID, status string, action func() (bool, error)) error {
	var err error
	for attempt := 1; attempt <= 3; attempt++ {
		var changed bool
		changed, err = action()
		if err == nil && changed {
			return nil
		}
		if err == nil {
			err = fmt.Errorf("job %s was not updated", requestID)
		}
		if attempt < 3 {
			u.logger.Warn("[PROCESS_NEXT_JOB_STATE_PERSIST_RETRY]", "request_id", requestID, "status", status, "attempt", attempt, "error", err)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(time.Second):
			}
		}
	}
	return fmt.Errorf("persist job %s status %s: %w", requestID, status, err)
}
