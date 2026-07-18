from __future__ import annotations

from src.platform.logger import Logger

HCM_NOW_SQL = "(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Ho_Chi_Minh')"


REQUEST_ID_UNIQUE_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_kafka_events_request_id_unique
ON kafka_events (request_id);
"""


CLAIM_QUEUE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_kafka_events_claim_queue
ON kafka_events (event_type, status, created_at);
"""

DATASET_VERSION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS model_lifecycle_dataset_versions (
    model_type TEXT PRIMARY KEY,
    next_version BIGINT NOT NULL CHECK (next_version > 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

LIFECYCLE_STATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS model_lifecycle_states (
    lifecycle_id TEXT PRIMARY KEY,
    model_type TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    current_stage TEXT NOT NULL,
    status TEXT NOT NULL,
    context JSONB NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS model_lifecycle_outbox (
    id BIGSERIAL PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    topic TEXT NOT NULL,
    message_key TEXT NOT NULL,
    payload JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    sent_at TIMESTAMPTZ
);
"""


ALTER_TIMESTAMP_DEFAULTS_SQL = """
ALTER TABLE kafka_events
    ALTER COLUMN consumed_at SET DEFAULT """ + HCM_NOW_SQL + """,
    ALTER COLUMN created_at SET DEFAULT """ + HCM_NOW_SQL + """,
    ALTER COLUMN updated_at SET DEFAULT """ + HCM_NOW_SQL + """;
"""


class KafkaEventSchemaGuard:
    def __init__(self, *, logger: Logger) -> None:
        self._logger = logger

    async def ensure_indexes(self, connection) -> None:
        await connection.execute(ALTER_TIMESTAMP_DEFAULTS_SQL)
        await connection.execute(REQUEST_ID_UNIQUE_INDEX_SQL)
        await connection.execute(CLAIM_QUEUE_INDEX_SQL)
        await connection.execute(DATASET_VERSION_TABLE_SQL)
        await connection.execute(LIFECYCLE_STATE_TABLE_SQL)
        self._logger.info("[KAFKA_EVENT_SCHEMA_INDEXES_ENSURED]")
