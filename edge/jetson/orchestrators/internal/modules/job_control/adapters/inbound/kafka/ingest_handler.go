package kafka

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"orchestrator/internal/modules/job_control/application/dto"
	"orchestrator/internal/modules/job_control/application/usecase"
	"orchestrator/internal/modules/job_control/domain"
	platform "orchestrator/internal/platform/messaging/kafka"
)

type IngestHandler struct {
	ingest	*usecase.IngestJob
	consumerGroup, schemaName, schemaVersion string
}

func NewIngestHandler(ingest *usecase.IngestJob, consumerGroup, schemaName, schemaVersion string) (*IngestHandler, error) {
	if ingest == nil {
		return nil, fmt.Errorf("ingest job must not be nil")
	}
	if strings.TrimSpace(consumerGroup) == "" {
		return nil, fmt.Errorf("consumer group must not be empty")
	}
	if schemaName == "" {
		schemaName = "mlflow_tracking_job"
	}
	if schemaVersion == "" {
		schemaVersion = "1.0"
	}
	return &IngestHandler{ingest: ingest, consumerGroup: consumerGroup, schemaName: schemaName, schemaVersion: schemaVersion}, nil
}
func (h *IngestHandler) Handle(ctx context.Context, message platform.Message) error {
	var envelope struct {
		RequestID string          `json:"request_id"`
		EventType string          `json:"event_type"`
		Payload   json.RawMessage `json:"payload"`
	}
	if err := json.Unmarshal(message.Value, &envelope); err != nil {
		return fmt.Errorf("decode Kafka job JSON: %w", err)
	}
	if strings.TrimSpace(envelope.RequestID) == "" || strings.TrimSpace(envelope.EventType) == "" {
		return fmt.Errorf("Kafka job requires non-empty request_id and event_type")
	}
	var payload map[string]any
	if len(envelope.Payload) > 0 && string(envelope.Payload) != "null" {
		if err := json.Unmarshal(envelope.Payload, &payload); err != nil {
			return fmt.Errorf("Kafka job payload must be an object: %w", err)
		}
	}
	var key *string
	if message.Key != nil {
		decoded := string(message.Key)
		key = &decoded
	}
	_, err := h.ingest.Execute(ctx, dto.ConsumeEventCommand{RequestID: envelope.RequestID, EventType: envelope.EventType, SchemaName: h.schemaName, SchemaVersion: h.schemaVersion, MessageKey: key, MessageIdentity: domain.MessageIdentity{Topic: message.Topic, PartitionID: message.Partition, MessageOffset: message.Offset, ConsumerGroup: h.consumerGroup}, Payload: payload})
	return err
}
