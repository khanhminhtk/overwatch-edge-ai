CREATE TABLE IF NOT EXISTS pipeline_runs (
 request_id VARCHAR(100) PRIMARY KEY, pipeline_type VARCHAR(64) NOT NULL, status VARCHAR(30) NOT NULL,
 payload JSONB NOT NULL, error_message TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS pipeline_outbox (
 id BIGSERIAL PRIMARY KEY, request_id VARCHAR(100) NOT NULL REFERENCES pipeline_runs(request_id), topic VARCHAR(255) NOT NULL,
 event_type VARCHAR(255) NOT NULL, payload JSONB NOT NULL, published_at TIMESTAMPTZ, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE INDEX IF NOT EXISTS idx_pipeline_outbox_pending ON pipeline_outbox(id) WHERE published_at IS NULL;
CREATE TABLE IF NOT EXISTS pipeline_dataset_versions (
 model_type VARCHAR(32) PRIMARY KEY, next_version BIGINT NOT NULL DEFAULT 1);
