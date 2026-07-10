from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from src.platform.persistence.postgres.pool import PostgresPool
from src.platform.persistence.postgres.healthcheck import (
    PostgresHealthCheck,
)


class PostgresHealthCheckUnitTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.mock_pool = MagicMock(spec=PostgresPool)

        self.fake_connection = AsyncMock()
        self.fake_connection.execute = AsyncMock()

        self.mock_pool.acquire = MagicMock()
        self.mock_pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=self.fake_connection
        )
        self.mock_pool.acquire.return_value.__aexit__ = AsyncMock(
            return_value=None
        )

        self.health = PostgresHealthCheck(pool=self.mock_pool)

    async def test_check_returns_true_when_select_1_succeeds(self) -> None:
        self.fake_connection.execute.return_value = "SELECT 1"
        result = await self.health.check()
        self.assertTrue(result)
        self.fake_connection.execute.assert_awaited_once_with("SELECT 1")

    async def test_check_returns_false_when_result_mismatch(self) -> None:
        self.fake_connection.execute.return_value = "SELECT 0"
        result = await self.health.check()
        self.assertFalse(result)

    async def test_check_returns_false_on_connection_error(self) -> None:
        self.fake_connection.execute.side_effect = RuntimeError("db down")
        result = await self.health.check()
        self.assertFalse(result)

    async def test_check_returns_false_on_acquire_error(self) -> None:
        self.mock_pool.acquire.return_value.__aenter__.side_effect = (
            RuntimeError("pool closed")
        )
        result = await self.health.check()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
