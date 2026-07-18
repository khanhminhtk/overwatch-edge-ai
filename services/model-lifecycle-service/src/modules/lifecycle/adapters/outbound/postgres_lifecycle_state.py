from __future__ import annotations

import json
from typing import Any

from src.modules.lifecycle.application import LifecycleCommand
from src.modules.lifecycle.domain import LifecycleContext
from src.platform.persistence.postgres.transaction import PostgresTransaction


class PostgresLifecycleStateRepository:
    """Persists state and next Kafka command in one transaction (transactional outbox)."""

    def __init__(self, transaction: PostgresTransaction) -> None:
        self._transaction = transaction

    async def save_and_enqueue(self, context: LifecycleContext, command: LifecycleCommand, topic: str) -> None:
        async with self._transaction.transaction() as connection:
            await connection.execute(
                """INSERT INTO model_lifecycle_states (lifecycle_id, model_type, dataset_version, current_stage, status, context)
                VALUES ($1,$2,$3,$4,'RUNNING',$5::jsonb)
                ON CONFLICT (lifecycle_id) DO UPDATE SET current_stage=EXCLUDED.current_stage,
                    status='RUNNING', context=EXCLUDED.context, updated_at=CURRENT_TIMESTAMP""",
                context.lifecycle_id, context.model_type.value, context.dataset_version,
                context.stage.value, json.dumps(context.to_payload()),
            )
            await connection.execute(
                """INSERT INTO model_lifecycle_outbox (idempotency_key, topic, message_key, payload)
                VALUES ($1,$2,$3,$4::jsonb) ON CONFLICT (idempotency_key) DO NOTHING""",
                command.request_id, topic, command.request_id,
                json.dumps({"request_id": command.request_id, "event_type": command.event_type, "payload": command.payload}),
            )

    async def claim_outbox(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self._transaction.transaction() as connection:
            rows = await connection.fetch(
                """UPDATE model_lifecycle_outbox SET status='PROCESSING', attempts=attempts+1
                WHERE id IN (SELECT id FROM model_lifecycle_outbox WHERE status='PENDING' ORDER BY id FOR UPDATE SKIP LOCKED LIMIT $1)
                RETURNING id, topic, message_key, payload""", limit)
        return [dict(row) for row in rows]

    async def mark_outbox_sent(self, outbox_id: int) -> None:
        async with self._transaction.transaction() as connection:
            await connection.execute("UPDATE model_lifecycle_outbox SET status='SENT', sent_at=CURRENT_TIMESTAMP WHERE id=$1", outbox_id)
