package kafka

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"orchestrator/internal/modules/job_control/application/dto"
	platform "orchestrator/internal/platform/messaging/kafka"
)

type producer interface {
	Publish(context.Context, platform.Message) error
}

// SuccessPublisher emits a best-effort completion event after the durable job
// status has been updated. A publishing failure never changes the job result.
type SuccessPublisher struct {
	producer producer
	topic    string
}

func NewSuccessPublisher(producer producer, topic string) (*SuccessPublisher, error) {
	if producer == nil {
		return nil, fmt.Errorf("Kafka producer must not be nil")
	}
	if strings.TrimSpace(topic) == "" {
		return nil, fmt.Errorf("success topic must not be empty")
	}
	return &SuccessPublisher{producer: producer, topic: topic}, nil
}
func (p *SuccessPublisher) PublishSuccess(ctx context.Context, jobName, serverID string, job dto.ClaimedJob, result map[string]any) error {
	payload, err := json.Marshal(map[string]any{"request_id": job.RequestID, "job_type": jobName, "event_type": job.EventType, "status": "PROCESSED", "server_id": serverID, "processed_at": time.Now().Format(time.RFC3339Nano), "payload": job.Payload, "result": result, "error_message": nil})
	if err != nil {
		return fmt.Errorf("encode job success event: %w", err)
	}
	return p.producer.Publish(ctx, platform.Message{Topic: p.topic, Partition: 0, Offset: 0, Key: []byte(job.RequestID), Value: payload, Headers: []platform.Header{{Key: "content-type", Value: []byte("application/json")}}, Timestamp: time.Now()})
}
