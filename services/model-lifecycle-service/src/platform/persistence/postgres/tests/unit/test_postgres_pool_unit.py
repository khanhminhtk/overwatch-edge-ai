from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from src.platform.logger import Logger
from src.platform.persistence.postgres.errors import PostgresNotStartedError
from src.platform.persistence.postgres.pool import PostgresPool


class PostgresPoolUnitTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.dsn = "postgresql://u:p@h:5432/db"
        self.logger = MagicMock(spec=Logger)

        self.fake_conn = MagicMock()

        acquire_ctx = MagicMock()
        acquire_ctx.__aenter__ = AsyncMock(return_value=self.fake_conn)
        acquire_ctx.__aexit__ = AsyncMock(return_value=None)

        self.asyncpg_pool = MagicMock()
        self.asyncpg_pool.acquire = MagicMock(return_value=acquire_ctx)
        self.asyncpg_pool.close = AsyncMock()

        patcher = patch("src.platform.persistence.postgres.pool.asyncpg")
        self.mock_asyncpg = patcher.start()
        self.mock_asyncpg.create_pool = AsyncMock(
            return_value=self.asyncpg_pool
        )
        self.addCleanup(patcher.stop)

        self.pool = PostgresPool(
            dsn=self.dsn,
            logger=self.logger,
            min_size=2,
            max_size=20,
            command_timeout=15.0,
        )

    # --- start() ---

    async def test_start_creates_pool(self) -> None:
        await self.pool.start()
        self.mock_asyncpg.create_pool.assert_awaited_once_with(
            dsn=self.dsn,
            min_size=2,
            max_size=20,
            command_timeout=15.0,
        )
        self.assertIsNotNone(self.pool._pool)
        self.logger.info.assert_called_once()

    async def test_start_is_idempotent_when_already_started(self) -> None:
        await self.pool.start()
        self.mock_asyncpg.create_pool.reset_mock()
        self.logger.reset_mock()

        await self.pool.start()
        self.mock_asyncpg.create_pool.assert_not_awaited()
        self.logger.warning.assert_called_once()

    async def test_start_logs_info(self) -> None:
        await self.pool.start()
        log_arg = self.logger.info.call_args[0][0]
        self.assertIn("POOL_STARTED", log_arg)

    # --- pool property ---

    async def test_pool_property_returns_pool_when_started(self) -> None:
        await self.pool.start()
        self.assertIs(self.pool.pool, self.asyncpg_pool)

    async def test_pool_property_raises_when_not_started(self) -> None:
        with self.assertRaises(PostgresNotStartedError):
            _ = self.pool.pool

    async def test_pool_property_logs_error_when_not_started(self) -> None:
        with self.assertRaises(PostgresNotStartedError):
            _ = self.pool.pool
        self.logger.error.assert_called_once()

    # --- stop() ---

    async def test_stop_closes_pool(self) -> None:
        await self.pool.start()
        await self.pool.stop()
        self.asyncpg_pool.close.assert_awaited_once()
        self.assertIsNone(self.pool._pool)

    async def test_stop_logs_info(self) -> None:
        await self.pool.start()
        await self.pool.stop()
        log_arg = self.logger.info.call_args[0][0]
        self.assertIn("POOL_STOPPED", log_arg)

    async def test_stop_is_safe_when_not_started(self) -> None:
        await self.pool.stop()
        self.asyncpg_pool.close.assert_not_awaited()
        self.logger.warning.assert_called_once()

    async def test_stop_does_not_reclose_after_stop(self) -> None:
        await self.pool.start()
        await self.pool.stop()
        self.asyncpg_pool.close.reset_mock()

        await self.pool.stop()
        self.asyncpg_pool.close.assert_not_awaited()

    # --- acquire() ---

    async def test_acquire_returns_connection_from_pool(self) -> None:
        await self.pool.start()
        async with self.pool.acquire() as conn:
            self.assertIs(conn, self.fake_conn)

    async def test_acquire_raises_when_pool_not_started(self) -> None:
        with self.assertRaises(PostgresNotStartedError):
            async with self.pool.acquire():
                pass  # pragma: no cover

    async def test_acquire_logs_error_when_not_started(self) -> None:
        with self.assertRaises(PostgresNotStartedError):
            async with self.pool.acquire():
                pass  # pragma: no cover
        self.logger.error.assert_called_once()

    # --- defaults ---

    def test_default_constructor_values(self) -> None:
        p = PostgresPool(dsn=self.dsn, logger=self.logger)
        self.assertEqual(p._min_size, 1)
        self.assertEqual(p._max_size, 10)
        self.assertEqual(p._command_timeout, 30.0)


if __name__ == "__main__":
    unittest.main()
