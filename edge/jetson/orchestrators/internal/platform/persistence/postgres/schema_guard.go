package postgres

import (
	"context"
	"fmt"
)

const (
	alterTimestampDefaultsSQL = `ALTER TABLE kafka_events
    ALTER COLUMN consumed_at SET DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Ho_Chi_Minh'),
    ALTER COLUMN created_at SET DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Ho_Chi_Minh'),
    ALTER COLUMN updated_at SET DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Ho_Chi_Minh');`
	requestIDUniqueIndexSQL = `CREATE UNIQUE INDEX IF NOT EXISTS idx_kafka_events_request_id_unique ON kafka_events (request_id);`
	claimQueueIndexSQL      = `CREATE INDEX IF NOT EXISTS idx_kafka_events_claim_queue ON kafka_events (event_type, status, created_at);`
	datasetVersionTableSQL  = `CREATE TABLE IF NOT EXISTS model_lifecycle_dataset_versions (model_type TEXT PRIMARY KEY, next_version BIGINT NOT NULL CHECK (next_version > 0), updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP);`
	lifecycleStateTableSQL  = `CREATE TABLE IF NOT EXISTS model_lifecycle_states (lifecycle_id TEXT PRIMARY KEY, model_type TEXT NOT NULL, dataset_version TEXT NOT NULL, current_stage TEXT NOT NULL, status TEXT NOT NULL, context JSONB NOT NULL, retry_count INTEGER NOT NULL DEFAULT 0, error_message TEXT, updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS model_lifecycle_outbox (id BIGSERIAL PRIMARY KEY, idempotency_key TEXT NOT NULL UNIQUE, topic TEXT NOT NULL, message_key TEXT NOT NULL, payload JSONB NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING', attempts INTEGER NOT NULL DEFAULT 0, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, sent_at TIMESTAMPTZ);`
)

// KafkaEventSchemaGuard creates the indexes and tables required by the Kafka
// event queue and model lifecycle workflow.
type KafkaEventSchemaGuard struct{ logger Logger }

func NewKafkaEventSchemaGuard(logger Logger) KafkaEventSchemaGuard {
	return KafkaEventSchemaGuard{logger: logger}
}
func (g KafkaEventSchemaGuard) Ensure(ctx context.Context, executor Executor) error {
	for _, statement := range []string{alterTimestampDefaultsSQL, requestIDUniqueIndexSQL, claimQueueIndexSQL, datasetVersionTableSQL, lifecycleStateTableSQL} {
		if _, err := executor.Exec(ctx, statement); err != nil {
			return fmt.Errorf("ensure Kafka event schema: %w", err)
		}
	}
	if g.logger != nil {
		g.logger.Info("[KAFKA_EVENT_SCHEMA_INDEXES_ENSURED]")
	}
	return nil
}
