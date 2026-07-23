package ports

import (
	"context"

	"orchestrator/internal/modules/job_control/application/dto"
	"orchestrator/internal/modules/job_control/domain"
)

type JobRecordRepository interface {
	CreateIfNotExists(context.Context, domain.JobRecord, string, string, *string) (bool, error)
	MarkProcessedByRequestID(context.Context, string) (bool, error)
	MarkFailedByRequestID(context.Context, string, string) (bool, error)
}
type JobClaimRepository interface {
	ClaimNextPendingJob(context.Context, string) (*dto.ClaimedJob, error)
}
type JobHandler interface {
	Handle(context.Context, dto.ClaimedJob) dto.JobResult
}
type SuccessEventPublisher interface {
	PublishSuccess(context.Context, string, string, dto.ClaimedJob, map[string]any) error
}
