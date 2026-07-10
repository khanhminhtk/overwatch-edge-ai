from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from src.platform.persistence.postgres.pool import PostgresPool
from src.platform.persistence.postgres.transaction import (
    PostgresTransaction,
)


class PostgresTransactionUnitTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.mock_pool = MagicMock(spec=PostgresPool)

        self.fake_connection = MagicMock()
        self.fake_connection.transaction = MagicMock()
        self.fake_connection.transaction.return_value.__aenter__ = AsyncMock(
            return_value=self.fake_connection
        )
        self.fake_connection.transaction.return_value.__aexit__ = AsyncMock(
            return_value=None
        )

        self.mock_pool.acquire = MagicMock()
        self.mock_pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=self.fake_connection
        )
        self.mock_pool.acquire.return_value.__aexit__ = AsyncMock(
            return_value=None
        )

        self.tx = PostgresTransaction(pool=self.mock_pool)

    async def test_transaction_yields_connection(self) -> None:
        async with self.tx.transaction() as conn:
            self.assertIs(conn, self.fake_connection)

    async def test_transaction_acquires_from_pool(self) -> None:
        async with self.tx.transaction():
            pass
        self.mock_pool.acquire.assert_called_once()

    async def test_transaction_opens_nested_db_transaction(self) -> None:
        async with self.tx.transaction():
            pass
        self.fake_connection.transaction.assert_called_once()
        self.fake_connection.transaction.return_value.__aenter__.assert_awaited_once()
        self.fake_connection.transaction.return_value.__aexit__.assert_awaited_once()

    async def test_transaction_commits_on_success(self) -> None:
        async with self.tx.transaction():
            pass
        # No exception → transaction.__aexit__ called without exception
        args = (
            self.fake_connection.transaction.return_value.__aexit__.call_args
        )
        self.assertIsNone(args[0][0])  # exc_type

    async def test_transaction_rolls_back_on_error(self) -> None:
        with self.assertRaises(RuntimeError):
            async with self.tx.transaction():
                raise RuntimeError("fail")

        self.fake_connection.transaction.return_value.__aexit__.assert_awaited_once()

    async def test_transaction_releases_connection_on_success(self) -> None:
        async with self.tx.transaction():
            pass
        self.mock_pool.acquire.return_value.__aexit__.assert_awaited_once()

    async def test_transaction_releases_connection_on_error(self) -> None:
        with self.assertRaises(ValueError):
            async with self.tx.transaction():
                raise ValueError("err")
        self.mock_pool.acquire.return_value.__aexit__.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
