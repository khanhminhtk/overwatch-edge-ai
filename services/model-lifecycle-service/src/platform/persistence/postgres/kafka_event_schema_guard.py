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
        self._logger.info("[KAFKA_EVENT_SCHEMA_INDEXES_ENSURED]")
