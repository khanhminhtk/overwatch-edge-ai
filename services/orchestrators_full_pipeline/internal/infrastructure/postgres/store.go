package postgres

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"

	"full_pipeline/internal/domain"
	"github.com/jackc/pgx/v5/pgxpool"
)

//go:embed schema.sql
var schema string

type Store struct{ pool *pgxpool.Pool }

func New(pool *pgxpool.Pool) *Store { return &Store{pool: pool} }
func (s *Store) EnsureSchema(ctx context.Context) error {
	_, err := s.pool.Exec(ctx, schema)
	return err
}
func (s *Store) AllocateDatasetVersion(ctx context.Context, modelType string) (string, error) {
	var version int64
	err := s.pool.QueryRow(ctx, `INSERT INTO pipeline_dataset_versions(model_type,next_version) VALUES($1,2)
ON CONFLICT(model_type) DO UPDATE SET next_version=pipeline_dataset_versions.next_version+1
RETURNING next_version-1`, modelType).Scan(&version)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("v%d", version), nil
}
func (s *Store) Start(ctx context.Context, command domain.Command, next domain.OutgoingCommand) error {
	payload, _ := json.Marshal(command.Payload)
	outgoing, _ := json.Marshal(next.Command.Payload)
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	_, err = tx.Exec(ctx, `INSERT INTO pipeline_runs(request_id,pipeline_type,status,payload) VALUES($1,$2,'PROCESSING',$3::jsonb) ON CONFLICT(request_id) DO NOTHING`, command.RequestID, command.EventType, payload)
	if err != nil {
		return err
	}
	_, err = tx.Exec(ctx, `INSERT INTO pipeline_outbox(request_id,topic,event_type,payload) VALUES($1,$2,$3,$4::jsonb)`, command.RequestID, next.Topic, next.Command.EventType, outgoing)
	if err != nil {
		return err
	}
	return tx.Commit(ctx)
}
func (s *Store) Result(ctx context.Context, event domain.ResultEvent, next *domain.OutgoingCommand) error {
	if event.Status != "PROCESSED" {
		_, err := s.pool.Exec(ctx, `UPDATE pipeline_runs SET status='FAILED',updated_at=now() WHERE request_id=$1`, event.RequestID)
		return err
	}
	if next == nil {
		_, err := s.pool.Exec(ctx, `UPDATE pipeline_runs SET status='COMPLETED',updated_at=now() WHERE request_id=$1`, event.RequestID)
		return err
	}
	payload, _ := json.Marshal(next.Command.Payload)
	_, err := s.pool.Exec(ctx, `INSERT INTO pipeline_outbox(request_id,topic,event_type,payload) VALUES($1,$2,$3,$4::jsonb)`, event.RequestID, next.Topic, next.Command.EventType, payload)
	return err
}

type OutboxRecord struct {
	ID                          int64
	RequestID, Topic, EventType string
	Payload                     map[string]any
}

func (s *Store) Pending(ctx context.Context) ([]OutboxRecord, error) {
	rows, err := s.pool.Query(ctx, `SELECT id,request_id,topic,event_type,payload FROM pipeline_outbox WHERE published_at IS NULL ORDER BY id LIMIT 100`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := []OutboxRecord{}
	for rows.Next() {
		var record OutboxRecord
		var raw []byte
		if err := rows.Scan(&record.ID, &record.RequestID, &record.Topic, &record.EventType, &raw); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(raw, &record.Payload); err != nil {
			return nil, fmt.Errorf("decode outbox payload: %w", err)
		}
		result = append(result, record)
	}
	return result, rows.Err()
}
func (s *Store) MarkPublished(ctx context.Context, id int64) error {
	_, err := s.pool.Exec(ctx, `UPDATE pipeline_outbox SET published_at=now() WHERE id=$1 AND published_at IS NULL`, id)
	return err
}
