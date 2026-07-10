from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock

from src.platform.config import ConfigLoader
from src.platform.logger import Logger
from src.platform.persistence.postgres.config import PostgresConfig
from src.platform.persistence.postgres.errors import PostgresNotStartedError
from src.platform.persistence.postgres.pool import PostgresPool


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


class PostgresPoolIntegrationTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.logger = MagicMock(spec=Logger)
        self.dsn = _get_dsn()
        self.pool = PostgresPool(
            dsn=self.dsn,
            logger=self.logger,
            min_size=1,
            max_size=5,
            command_timeout=5.0,
        )

    async def asyncTearDown(self) -> None:
        if self.pool._pool is not None:
            await self.pool.stop()

    async def test_start_creates_usable_pool(self) -> None:
        await self.pool.start()
        pool = self.pool.pool
        async with pool.acquire() as conn:
            val = await conn.fetchval("SELECT 1")
        self.assertEqual(val, 1)

    async def test_acquire_returns_real_connection(self) -> None:
        await self.pool.start()
        async with self.pool.acquire() as conn:
            result = await conn.fetch("SELECT 42 AS number")
        self.assertEqual(result[0]["number"], 42)

    async def test_pool_handles_concurrent_acquires(self) -> None:
        await self.pool.start()
        pool = self.pool.pool

        async def query(n: int) -> int:
            async with pool.acquire() as conn:
                return await conn.fetchval(f"SELECT {n}")

        results = await self.asyncExec(query, [1, 2, 3])
        self.assertEqual(results, [1, 2, 3])

    async def asyncExec(self, fn, args_list):
        import asyncio
        tasks = [fn(a) for a in args_list]
        return await asyncio.gather(*tasks)

    async def test_stop_closes_pool(self) -> None:
        await self.pool.start()
        await self.pool.stop()
        self.assertIsNone(self.pool._pool)

    async def test_acquire_after_stop_raises(self) -> None:
        await self.pool.start()
        await self.pool.stop()
        with self.assertRaises(PostgresNotStartedError):
            async with self.pool.acquire():
                pass  # pragma: no cover

    async def test_pool_property_before_start_raises(self) -> None:
        with self.assertRaises(PostgresNotStartedError):
            _ = self.pool.pool

    async def test_double_start_is_safe(self) -> None:
        await self.pool.start()
        await self.pool.start()
        pool = self.pool.pool
        async with pool.acquire() as conn:
            val = await conn.fetchval("SELECT 1")
        self.assertEqual(val, 1)

    async def test_execute_query_via_acquire(self) -> None:
        await self.pool.start()
        async with self.pool.acquire() as conn:
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS _pool_test (id INT)"
            )
            await conn.execute(
                "INSERT INTO _pool_test VALUES (1), (2)"
            )
            rows = await conn.fetch("SELECT * FROM _pool_test ORDER BY id")
            await conn.execute("DROP TABLE _pool_test")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], 1)
        self.assertEqual(rows[1]["id"], 2)


if __name__ == "__main__":
    unittest.main()
