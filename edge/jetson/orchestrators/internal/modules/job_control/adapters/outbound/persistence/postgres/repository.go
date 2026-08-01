package postgres

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5"

	"orchestrator/internal/modules/job_control/application/dto"
	"orchestrator/internal/modules/job_control/domain"
	platform "orchestrator/internal/platform/persistence/postgres"
)

const hcmNowSQL = "(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Ho_Chi_Minh')"

const insertJobSQL = `INSERT INTO kafka_events (request_id, topic, partition_id, message_offset, consumer_group, message_key, event_type, schema_name, schema_version, payload, status, produced_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12) ON CONFLICT DO NOTHING RETURNING id`

const claimJobSQL = `WITH candidate AS (
 SELECT e.id FROM kafka_events e
 WHERE e.event_type = $1
 AND (e.status = 'RECEIVED' OR (e.status = 'PROCESSING' AND e.updated_at < $2))
 AND (COALESCE(e.payload->>'target_server_id', '') = '' OR e.payload->>'target_server_id' = $3)
 AND NOT EXISTS (SELECT 1 FROM kafka_events running WHERE running.event_type = $1 AND running.status = 'PROCESSING' AND running.updated_at >= $2)
 ORDER BY e.created_at ASC LIMIT 1 FOR UPDATE SKIP LOCKED
) UPDATE kafka_events e SET status = 'PROCESSING', updated_at = ` + hcmNowSQL + ` FROM candidate WHERE e.id = candidate.id
RETURNING e.request_id, e.event_type, e.payload, e.status`

const markProcessedSQL = `UPDATE kafka_events SET status = 'PROCESSED', error_message = NULL, processed_at = ` + hcmNowSQL + `, updated_at = ` + hcmNowSQL + ` WHERE request_id = $1 RETURNING id`
const markFailedSQL = `UPDATE kafka_events SET status = 'FAILED', error_message = $2, processed_at = ` + hcmNowSQL + `, updated_at = ` + hcmNowSQL + ` WHERE request_id = $1 RETURNING id`


type Repository struct {
	transaction           platform.Transaction
	eventType             string
	reclaimTimeoutSeconds int
}

func NewRepository(
	transaction platform.Transaction, 
	eventType string, 
	reclaimTimeoutSeconds int) (*Repository, error) {
	if strings.TrimSpace(eventType) == "" {
		return nil, fmt.Errorf("event type must not be empty")
	}

	if reclaimTimeoutSeconds <= 0 {
		reclaimTimeoutSeconds = 1800
	}

	return &Repository{
		transaction: transaction, 
		eventType: eventType, 
		reclaimTimeoutSeconds: reclaimTimeoutSeconds,
	}, nil
}

func (r *Repository) CreateIfNotExists(
	ctx context.Context, 
	job domain.JobRecord, 
	schemaName, 
	schemaVersion string, 
	messageKey *string,
	) (bool, error) {
	payload, err := json.Marshal(job.Payload)
	if err != nil {
		return false, fmt.Errorf("marshal job payload: %w", err)
	}

	created := false
	err = r.transaction.Run(ctx, func(tx pgx.Tx) error {
		var id int64
		err := tx.QueryRow(
			ctx, 
			insertJobSQL, 
			job.RequestID, 
			job.MessageIdentity.Topic, 
			job.MessageIdentity.PartitionID, 
			job.MessageIdentity.MessageOffset, 
			job.MessageIdentity.ConsumerGroup, 
			messageKey, 
			job.EventType, 
			nullable(schemaName), 
			nullable(schemaVersion), 
			payload, 
			string(job.Status), 
			job.ProducedAt).Scan(&id)

		if err == pgx.ErrNoRows {
			return nil
		}
		if err != nil {
			return err
		}
		created = true
		return nil
	})
	return created, err
}
func (r *Repository) ClaimNextPendingJob(ctx context.Context, serverID string) (*dto.ClaimedJob, error) {
	if strings.TrimSpace(serverID) == "" {
		return nil, fmt.Errorf("server ID must not be empty")
	}
	
	var result *dto.ClaimedJob
	err := r.transaction.Run(ctx, func(tx pgx.Tx) error {
		var requestID, eventType, status string
		var rawPayload []byte
		err := tx.QueryRow(ctx, claimJobSQL, r.eventType, timeNow().Add(-seconds(r.reclaimTimeoutSeconds)), serverID).Scan(&requestID, &eventType, &rawPayload, &status)
		if err == pgx.ErrNoRows {
			return nil
		}
		if err != nil {
			return err
		}
		payload := map[string]any{}
		if len(rawPayload) > 0 && string(rawPayload) != "null" {
			if err := json.Unmarshal(rawPayload, &payload); err != nil {
				return fmt.Errorf("decode claimed job payload: %w", err)
			}
		}
		result = &dto.ClaimedJob{RequestID: requestID, EventType: eventType, Payload: payload, Status: domain.Status(status)}
		return nil
	})
	return result, err
}

func (r *Repository) MarkProcessedByRequestID(ctx context.Context, requestID string) (bool, error) {
	return r.mark(ctx, markProcessedSQL, requestID)
}

func (r *Repository) MarkFailedByRequestID(ctx context.Context, requestID, message string) (bool, error) {
	return r.mark(ctx, markFailedSQL, requestID, message)
}

func (r *Repository) mark(ctx context.Context, query string, args ...any) (bool, error) {
	changed := false
	err := r.transaction.Run(ctx, func(tx pgx.Tx) error {
		var id int64
		err := tx.QueryRow(ctx, query, args...).Scan(&id)
		if err == pgx.ErrNoRows {
			return nil
		}
		if err != nil {
			return err
		}
		changed = true
		return nil
	})
	return changed, err
}

func nullable(value string) any {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	return value
}
