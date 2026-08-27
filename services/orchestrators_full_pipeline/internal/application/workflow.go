package application

import (
	"fmt"

	"full_pipeline/internal/domain"
)

// Workflow is pure routing logic. Storage and Kafka delivery are adapters.
type Workflow struct{}

func (Workflow) Start(command domain.Command) (domain.OutgoingCommand, error) {
	if err := command.Validate(); err != nil {
		return domain.OutgoingCommand{}, err
	}
	modelType, err := modelType(command.Payload)
	if err != nil {
		return domain.OutgoingCommand{}, err
	}
	switch command.EventType {
	case "train_requested":
		return lifecycle(command.RequestID, "dataset_requested_"+modelType, command.Payload), nil
	case "deploy_requested":
		edgeID, ok := command.Payload["edge_id"].(string)
		if !ok || edgeID == "" {
			return domain.OutgoingCommand{}, fmt.Errorf("deploy_requested requires payload.edge_id")
		}
		return domain.OutgoingCommand{Topic: "edge.model.download.jobs", Command: domain.Command{RequestID: command.RequestID, EventType: "mlflow_download_requested", Payload: command.Payload}}, nil
	case "fine_tuning_requested":
		if _, ok := command.Payload["labeled_dataset_uri"].(string); !ok {
			return domain.OutgoingCommand{}, fmt.Errorf("fine_tuning_requested requires payload.labeled_dataset_uri")
		}
		if _, ok := command.Payload["dataset_version"].(string); !ok {
			return domain.OutgoingCommand{}, fmt.Errorf("fine_tuning_requested requires allocated payload.dataset_version")
		}
		return domain.OutgoingCommand{Topic: "model-lifecycle.lifecycle.jobs", Command: domain.Command{RequestID: command.RequestID, EventType: "fine_tuning_materialize_requested", Payload: command.Payload}}, nil
	default:
		return domain.OutgoingCommand{}, fmt.Errorf("unsupported pipeline command %q", command.EventType)
	}
}

func (Workflow) AfterResult(event domain.ResultEvent) (*domain.OutgoingCommand, error) {
	if event.Status != "PROCESSED" {
		return nil, nil
	}
	payload := clone(event.Payload)
	for key, value := range event.Result {
		payload[key] = value
	}
	modelType, err := modelType(payload)
	if err != nil {
		return nil, err
	}
	var next domain.OutgoingCommand
	switch {
	case event.JobType == "dataset":
		next = lifecycle(event.RequestID, "train_requested_"+modelType, payload)
	case event.JobType == "training":
		next = lifecycle(event.RequestID, modelType, payload)
	case event.JobType == "mlflow_tracking":
		return nil, nil
	case event.EventType == "mlflow_download_requested":
		next = edge(event.RequestID, "edge.model.export.jobs", "export_tensorrt_requested", payload)
	case event.EventType == "export_tensorrt_requested":
		next = edge(event.RequestID, "edge.model.deploy.jobs", "deploy_requested", payload)
	case event.EventType == "deploy_requested":
		next = edge(event.RequestID, "edge.inference.runtime.jobs", "inference_runtime_requested", payload)
	case event.EventType == "edge_fail_batch_uploaded":
		payload["raw_archive_object"] = event.Payload["object_name"]
		next = edge(event.RequestID, "model-lifecycle.continual-learning.jobs", "continual_learning_requested", payload)
	case event.EventType == "fine_tuning_materialize_requested":
		next = lifecycle(event.RequestID, "dataset_requested_"+modelType, payload)
	default:
		return nil, nil
	}
	return &next, nil
}

func lifecycle(requestID, eventType string, payload map[string]any) domain.OutgoingCommand {
	return edge(requestID, "model-lifecycle.lifecycle.jobs", eventType, payload)
}
func edge(requestID, topic, eventType string, payload map[string]any) domain.OutgoingCommand {
	return domain.OutgoingCommand{Topic: topic, Command: domain.Command{RequestID: requestID, EventType: eventType, Payload: payload}}
}
func modelType(payload map[string]any) (string, error) {
	value, ok := payload["model_type"].(string)
	if !ok || (value != "detection" && value != "recognizer") {
		return "", fmt.Errorf("payload.model_type must be detection or recognizer")
	}
	return value, nil
}
func clone(value map[string]any) map[string]any {
	result := make(map[string]any, len(value))
	for key, item := range value {
		result[key] = item
	}
	return result
}
