from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from src.platform.persistence.postgres.kafka_event_schema_guard import (
    ALTER_TIMESTAMP_DEFAULTS_SQL,
    CLAIM_QUEUE_INDEX_SQL,
    DATASET_VERSION_TABLE_SQL,
    REQUEST_ID_UNIQUE_INDEX_SQL,
    KafkaEventSchemaGuard,
)


class KafkaEventSchemaGuardUnitTest(unittest.IsolatedAsyncioTestCase):
    async def test_ensure_indexes_executes_required_ddl(self) -> None:
        connection = MagicMock()
        connection.execute = AsyncMock()
        logger = MagicMock()
        guard = KafkaEventSchemaGuard(logger=logger)

        await guard.ensure_indexes(connection)

        connection.execute.assert_any_await(ALTER_TIMESTAMP_DEFAULTS_SQL)
        connection.execute.assert_any_await(REQUEST_ID_UNIQUE_INDEX_SQL)
        connection.execute.assert_any_await(CLAIM_QUEUE_INDEX_SQL)
        connection.execute.assert_any_await(DATASET_VERSION_TABLE_SQL)
        self.assertEqual(connection.execute.await_count, 4)
        logger.info.assert_called()

    def test_sql_constants_target_kafka_events(self) -> None:
        self.assertIn("ALTER TABLE kafka_events", ALTER_TIMESTAMP_DEFAULTS_SQL)
        self.assertIn("Asia/Ho_Chi_Minh", ALTER_TIMESTAMP_DEFAULTS_SQL)
        self.assertIn("CREATE UNIQUE INDEX IF NOT EXISTS", REQUEST_ID_UNIQUE_INDEX_SQL)
        self.assertIn("kafka_events", REQUEST_ID_UNIQUE_INDEX_SQL)
        self.assertIn("request_id", REQUEST_ID_UNIQUE_INDEX_SQL)
        self.assertIn("CREATE INDEX IF NOT EXISTS", CLAIM_QUEUE_INDEX_SQL)
        self.assertIn("event_type", CLAIM_QUEUE_INDEX_SQL)
        self.assertIn("status", CLAIM_QUEUE_INDEX_SQL)
        self.assertIn("created_at", CLAIM_QUEUE_INDEX_SQL)


if __name__ == "__main__":
    unittest.main()
