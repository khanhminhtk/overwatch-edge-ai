from __future__ import annotations

import unittest

from src.platform.persistence.postgres.errors import (
    PostgresConnectionError,
    PostgresError,
    PostgresNotStartedError,
    PostgresTransactionError,
)


class PostgresErrorHierarchyTest(unittest.TestCase):
    def test_postgres_error_is_runtime_error(self) -> None:
        self.assertTrue(issubclass(PostgresError, RuntimeError))

    def test_postgres_not_started_error_is_postgres_error(self) -> None:
        self.assertTrue(issubclass(PostgresNotStartedError, PostgresError))

    def test_postgres_connection_error_is_postgres_error(self) -> None:
        self.assertTrue(issubclass(PostgresConnectionError, PostgresError))

    def test_postgres_transaction_error_is_postgres_error(self) -> None:
        self.assertTrue(issubclass(PostgresTransactionError, PostgresError))

    def test_can_raise_and_catch_base(self) -> None:
        with self.assertRaises(PostgresError):
            raise PostgresNotStartedError("not started")

    def test_can_raise_and_catch_connection(self) -> None:
        with self.assertRaises(PostgresError):
            raise PostgresConnectionError("connection failed")

    def test_can_raise_and_catch_transaction(self) -> None:
        with self.assertRaises(PostgresError):
            raise PostgresTransactionError("tx failed")

    def test_error_message_preserved(self) -> None:
        msg = "something went wrong"
        err = PostgresError(msg)
        self.assertEqual(str(err), msg)

    def test_not_started_message(self) -> None:
        err = PostgresNotStartedError("pool not started")
        self.assertEqual(str(err), "pool not started")

    def test_connection_error_message(self) -> None:
        err = PostgresConnectionError("timeout")
        self.assertEqual(str(err), "timeout")

    def test_transaction_error_message(self) -> None:
        err = PostgresTransactionError("rollback")
        self.assertEqual(str(err), "rollback")


if __name__ == "__main__":
    unittest.main()
