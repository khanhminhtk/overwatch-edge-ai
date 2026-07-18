from __future__ import annotations

import json

from src.modules.lifecycle.adapters.outbound.postgres_lifecycle_state import PostgresLifecycleStateRepository
from src.platform.messaging.kafka.protocols import MessageProducer


class LifecycleOutboxDispatcher:
    def __init__(self, repository: PostgresLifecycleStateRepository, producer: MessageProducer) -> None:
        self._repository = repository
        self._producer = producer

    async def dispatch_once(self) -> int:
        rows = await self._repository.claim_outbox()
        for row in rows:
            self._producer.publish(
                topic=row["topic"], key=str(row["message_key"]).encode(),
                value=json.dumps(row["payload"]).encode(), headers=[("content-type", b"application/json")],
            )
            await self._repository.mark_outbox_sent(int(row["id"]))
        return len(rows)
