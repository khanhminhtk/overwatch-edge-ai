from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock

from src.platform.config import ConfigLoader
from src.platform.logger import Logger
from src.platform.persistence.postgres.config import PostgresConfig
from src.platform.persistence.postgres.pool import PostgresPool
from src.platform.persistence.postgres.transaction import (
    PostgresTransaction,
)


def _get_dsn() -> str:
    pwd = os.path.normpath(
        os.path.join(os.path.dirname(__file__), *([".."] * 8))
    )
    config = ConfigLoader.load(
        PostgresConfig,
        yaml_files=[f"{pwd}/services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml"],
        env_files=[f"{pwd}/services/model-lifecycle-service/config/.env"],
        section="PostgresSql",
    )
    return config.to_dsn()


class PostgresTransactionIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.logger = MagicMock(spec=Logger)
        self.dsn = _get_dsn()
        self.pool = PostgresPool(
            dsn=self.dsn,
            logger=self.logger,
            min_size=1,
            max_size=5,
            command_timeout=5.0,
        )
        await self.pool.start()
        self.tx = PostgresTransaction(pool=self.pool)

        async with self.pool.acquire() as conn:
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS _tx_test (id INT PRIMARY KEY, val TEXT)"
            )
            await conn.execute("DELETE FROM _tx_test")

    async def asyncTearDown(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS _tx_test")
        await self.pool.stop()

    async def test_transaction_commits_insert(self) -> None:
        async with self.tx.transaction() as conn:
            await conn.execute(
                "INSERT INTO _tx_test (id, val) VALUES (1, 'hello')"
            )

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT val FROM _tx_test WHERE id = 1"
            )
        self.assertEqual(row["val"], "hello")

    async def test_transaction_rolls_back_on_error(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO _tx_test (id, val) VALUES (10, 'before')"
            )

        with self.assertRaises(RuntimeError):
            async with self.tx.transaction() as conn:
                await conn.execute(
                    "INSERT INTO _tx_test (id, val) VALUES (20, 'inside')"
                )
                raise RuntimeError("rollback")

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id FROM _tx_test ORDER BY id"
            )
        ids = [r["id"] for r in rows]
        self.assertIn(10, ids)
        self.assertNotIn(20, ids)

    async def test_transaction_nests_multiple_operations(self) -> None:
        async with self.tx.transaction() as conn:
            await conn.execute(
                "INSERT INTO _tx_test (id, val) VALUES (100, 'a')"
            )
            await conn.execute(
                "INSERT INTO _tx_test (id, val) VALUES (200, 'b')"
            )

        async with self.pool.acquire() as conn:
            count = await conn.fetchval("SELECT count(*) FROM _tx_test")
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
