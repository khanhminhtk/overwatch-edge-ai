package application

import (
	"full_pipeline/internal/domain"
	"testing"
)

func TestDeployStartsWithMLflowDownload(t *testing.T) {
	next, err := (Workflow{}).Start(domain.Command{RequestID: "request-1", EventType: "deploy_requested", Payload: map[string]any{"model_type": "detection", "edge_id": "jetson-1"}})
	if err != nil {
		t.Fatal(err)
	}
	if next.Command.EventType != "mlflow_download_requested" {
		t.Fatalf("got %s", next.Command.EventType)
	}
}

func TestFailBatchStartsContinualLearning(t *testing.T) {
	next, err := (Workflow{}).AfterResult(domain.ResultEvent{RequestID: "request-1", EventType: "edge_fail_batch_uploaded", Status: "PROCESSED", Payload: map[string]any{"model_type": "recognizer", "object_name": "edge-fail/a.zip"}})
	if err != nil {
		t.Fatal(err)
	}
	if next == nil || next.Command.EventType != "continual_learning_requested" {
		t.Fatal("expected continual learning command")
	}
}

func TestFineTuningRequiresLabeledDataset(t *testing.T) {
	_, err := (Workflow{}).Start(domain.Command{RequestID: "request-1", EventType: "fine_tuning_requested", Payload: map[string]any{"model_type": "detection", "dataset_version": "v2"}})
	if err == nil {
		t.Fatal("expected validation error")
	}
}
