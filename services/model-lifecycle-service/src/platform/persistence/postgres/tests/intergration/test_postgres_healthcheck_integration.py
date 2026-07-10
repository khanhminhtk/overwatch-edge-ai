from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock

from src.platform.config import ConfigLoader
from src.platform.logger import Logger
from src.platform.persistence.postgres.config import PostgresConfig
from src.platform.persistence.postgres.pool import PostgresPool
from src.platform.persistence.postgres.healthcheck import (
    PostgresHealthCheck,
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


class PostgresHealthCheckIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.logger = MagicMock(spec=Logger)
        self.dsn = _get_dsn()
        self.pool = PostgresPool(
            dsn=self.dsn,
            logger=self.logger,
            min_size=1,
            max_size=2,
            command_timeout=5.0,
        )
        await self.pool.start()
        self.health = PostgresHealthCheck(pool=self.pool)

    async def asyncTearDown(self) -> None:
        await self.pool.stop()

    async def test_check_returns_true_when_db_is_up(self) -> None:
        result = await self.health.check()
        self.assertTrue(result)

    async def test_check_pool_is_not_disrupted_by_check(self) -> None:
        result = await self.health.check()
        self.assertTrue(result)

        async with self.pool.acquire() as conn:
            val = await conn.fetchval("SELECT 99")
        self.assertEqual(val, 99)

    async def test_check_fails_when_pool_stopped(self) -> None:
        await self.pool.stop()
        result = await self.health.check()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
