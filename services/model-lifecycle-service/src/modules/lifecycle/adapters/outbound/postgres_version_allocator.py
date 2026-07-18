from __future__ import annotations

from src.modules.lifecycle.domain import ModelType
from src.platform.persistence.postgres.transaction import PostgresTransaction


class PostgresDatasetVersionAllocator:
    """Atomically allocates monotonically increasing per-model dataset versions."""

    def __init__(self, transaction: PostgresTransaction) -> None:
        self._transaction = transaction

    async def allocate(self, model_type: ModelType) -> str:
        async with self._transaction.transaction() as connection:
            row = await connection.fetchrow(
                """
                INSERT INTO model_lifecycle_dataset_versions (model_type, next_version)
                VALUES ($1, 2)
                ON CONFLICT (model_type)
                DO UPDATE SET next_version = model_lifecycle_dataset_versions.next_version + 1,
                              updated_at = CURRENT_TIMESTAMP
                RETURNING next_version - 1 AS allocated_version
                """,
                model_type.value,
            )
        if row is None:
            raise RuntimeError("dataset version allocation did not return a version")
        return f"v{int(row['allocated_version'])}"
