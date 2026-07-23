package usecase

import (
	"context"
	"errors"
	"testing"

	"orchestrator/internal/modules/job_control/application/dto"
	"orchestrator/internal/modules/job_control/domain"
)

func TestIngestJobPersistsExpectedEvent(t *testing.T) {
	repository := &fakeRepository{}
	ingest, err := NewIngestJob(repository, discardLogger{}, []string{"deploy_requested"})
	if err != nil {
		t.Fatal(err)
	}
	created, err := ingest.Execute(context.Background(), dto.ConsumeEventCommand{RequestID: "request-1", EventType: "deploy_requested", MessageIdentity: domain.MessageIdentity{Topic: "jobs", PartitionID: 0, MessageOffset: 1, ConsumerGroup: "edge"}})
	if err != nil || !created {
		t.Fatalf("ingest = %t, %v", created, err)
	}
	if repository.created.RequestID != "request-1" || repository.created.Status != domain.StatusReceived {
		t.Fatalf("persisted job = %#v", repository.created)
	}
}
func TestProcessNextJobMarksFailedHandlerResult(t *testing.T) {
	repository := &fakeRepository{}
	process, err := NewProcessNextJob(&fakeClaimer{job: &dto.ClaimedJob{RequestID: "request-1", EventType: "deploy_requested"}}, fakeHandler{result: dto.JobResult{Success: false, ErrorMessage: "deploy failed"}}, repository, nil, discardLogger{}, "deploy")
	if err != nil {
		t.Fatal(err)
	}
	result, err := process.Execute(context.Background(), "jetson-1")
	if err != nil {
		t.Fatal(err)
	}
	if result.Success || repository.failedRequest != "request-1" || repository.failedMessage != "deploy failed" {
		t.Fatalf("result=%#v repository=%#v", result, repository)
	}
}
func TestProcessNextJobRetriesStatePersistence(t *testing.T) {
	repository := &fakeRepository{processFailures: 2}
	process, err := NewProcessNextJob(&fakeClaimer{job: &dto.ClaimedJob{RequestID: "request-1"}}, fakeHandler{result: dto.JobResult{Success: true}}, repository, nil, discardLogger{}, "deploy")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := process.Execute(context.Background(), "jetson-1"); err != nil {
		t.Fatal(err)
	}
	if repository.processedCalls != 3 {
		t.Fatalf("processed calls = %d, want 3", repository.processedCalls)
	}
}

type discardLogger struct{}

func (discardLogger) Info(string, ...any)  {}
func (discardLogger) Warn(string, ...any)  {}
func (discardLogger) Error(string, ...any) {}

type fakeRepository struct {
	created                         domain.JobRecord
	failedRequest, failedMessage    string
	processedCalls, processFailures int
}

func (r *fakeRepository) CreateIfNotExists(_ context.Context, job domain.JobRecord, _, _ string, _ *string) (bool, error) {
	r.created = job
	return true, nil
}
func (r *fakeRepository) MarkProcessedByRequestID(_ context.Context, _ string) (bool, error) {
	r.processedCalls++
	if r.processedCalls <= r.processFailures {
		return false, errors.New("temporary database failure")
	}
	return true, nil
}
func (r *fakeRepository) MarkFailedByRequestID(_ context.Context, requestID, message string) (bool, error) {
	r.failedRequest, r.failedMessage = requestID, message
	return true, nil
}

type fakeClaimer struct{ job *dto.ClaimedJob }

func (c *fakeClaimer) ClaimNextPendingJob(context.Context, string) (*dto.ClaimedJob, error) {
	return c.job, nil
}

type fakeHandler struct{ result dto.JobResult }

func (h fakeHandler) Handle(context.Context, dto.ClaimedJob) dto.JobResult { return h.result }
