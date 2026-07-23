package scheduler

import (
	"context"
	"time"

	"orchestrator/internal/modules/job_control/application/dto"
	"orchestrator/internal/modules/job_control/application/usecase"
)

type PollingRunner struct {
	process   *usecase.ProcessNextJob
	serverID  string
	idleSleep time.Duration
	logger    usecase.Logger
}

func NewPollingRunner(process *usecase.ProcessNextJob, serverID string, idleSleep time.Duration, logger usecase.Logger) *PollingRunner {
	if idleSleep <= 0 {
		idleSleep = 2 * time.Second
	}
	return &PollingRunner{process: process, serverID: serverID, idleSleep: idleSleep, logger: logger}
}
func (r *PollingRunner) RunOnce(ctx context.Context) (*dto.JobResult, error) {
	return r.process.Execute(ctx, r.serverID)
}
func (r *PollingRunner) Run(ctx context.Context) error {
	for {
		result, err := r.RunOnce(ctx)
		if err != nil {
			r.logger.Error("[JOB_POLLING_ERROR]", "error", err)
		}
		if result == nil || err != nil {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(r.idleSleep):
			}
		}
	}
}
func (r *PollingRunner) Shutdown(ctx context.Context, reason string) error {
	return r.process.FailInflightJob(ctx, reason)
}
