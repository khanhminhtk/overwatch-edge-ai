from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from src.platform.persistence.postgres.config import PostgresConfig


class PostgresConfigValidationTest(unittest.TestCase):
    def _make_valid(self, **overrides) -> PostgresConfig:
        kwargs = dict(
            host="localhost",
            port=5432,
            database="testdb",
            user="admin",
            password="secret",
        )
        kwargs.update(overrides)
        return PostgresConfig(**kwargs)

    def test_creates_with_valid_fields(self) -> None:
        cfg = self._make_valid()
        self.assertEqual(cfg.host, "localhost")
        self.assertEqual(cfg.port, 5432)
        self.assertEqual(cfg.database, "testdb")
        self.assertEqual(cfg.user, "admin")
        self.assertEqual(cfg.password, "secret")

    def test_empty_host_raises(self) -> None:
        for host in ("", "  ", "\t"):
            with self.subTest(host=repr(host)):
                with self.assertRaises(ValueError):
                    self._make_valid(host=host)

    def test_port_too_low_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_valid(port=0)

    def test_port_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_valid(port=-1)

    def test_port_too_high_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_valid(port=65536)

    def test_port_max_valid(self) -> None:
        cfg = self._make_valid(port=65535)
        self.assertEqual(cfg.port, 65535)

    def test_empty_database_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_valid(database="  ")

    def test_empty_user_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_valid(user="  ")

    def test_empty_password_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_valid(password="  ")

    def test_is_frozen(self) -> None:
        cfg = self._make_valid()
        with self.assertRaises(FrozenInstanceError):
            cfg.host = "other"

    def test_to_dict_returns_all_fields(self) -> None:
        cfg = self._make_valid(
            host="db.example.com",
            port=5433,
            database="mydb",
            user="user1",
            password="pass1",
        )
        d = cfg.to_dict()
        self.assertEqual(
            d,
            {
                "host": "db.example.com",
                "port": 5433,
                "database": "mydb",
                "user": "user1",
                "password": "pass1",
            },
        )

    def test_to_dsn_format(self) -> None:
        cfg = self._make_valid(
            host="myhost",
            port=5432,
            database="mydb",
            user="myuser",
            password="mypass",
        )
        expected = "postgresql://myuser:mypass@myhost:5432/mydb"
        self.assertEqual(cfg.to_dsn(), expected)

    def test_to_dsn_special_chars_in_password(self) -> None:
        cfg = self._make_valid(password="p@ss:w?rd")
        dsn = cfg.to_dsn()
        self.assertIn("localhost", dsn)
        self.assertIn("p@ss:w?rd", dsn)
        self.assertTrue(dsn.startswith("postgresql://"))


if __name__ == "__main__":
    unittest.main()
