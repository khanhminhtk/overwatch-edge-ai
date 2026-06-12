from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.infra.config.postgrest import PostgrestConfig
from src.infra.sql import PostgresSQLHandler


class _StubConfigLoader:
    def get_config(self):
        return {}

    def get_typed_config(self, dto_cls, config_section=None):
        if dto_cls is PostgrestConfig:
            return PostgrestConfig(
                user="postgres",
                password="postgres",
                database="overwatch",
                host="127.0.0.1",
                port=5432,
            )
        raise AssertionError(f"Unexpected dto_cls: {dto_cls}")

    def get_env_variable(self, var_name: str):
        values = {
            "POSTGRES_USER": "postgres",
            "POSTGRES_PASSWORD": "postgres",
            "POSTGRES_DB": "overwatch",
            "POSTGRES_HOST": "127.0.0.1",
            "POSTGRES_PORT": "5432",
            "SERVICE_HOST": "127.0.0.1",
        }
        return values[var_name]


class _RecordingSQLHandler(PostgresSQLHandler):
    def __init__(self):
        super().__init__(_StubConfigLoader(), logger=MagicMock())
        self.executed_queries = []

    def execute_query(self, query: str, params: tuple = None):
        self.executed_queries.append((query, params))
        return None


class PostgresSQLHandlerTest(unittest.TestCase):
    def test_ensure_schema_executes_schema_sql_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            schema_path = Path(tmp_dir) / "schema.sql"
            schema_path.write_text(
                "CREATE TABLE IF NOT EXISTS kafka_events (id BIGSERIAL PRIMARY KEY);",
                encoding="utf-8",
            )
            handler = _RecordingSQLHandler()

            handler.ensure_schema(schema_path=schema_path)

            self.assertEqual(len(handler.executed_queries), 1)
            self.assertIn("CREATE TABLE IF NOT EXISTS kafka_events", handler.executed_queries[0][0])

    def test_default_schema_path_points_to_repo_schema(self) -> None:
        schema_path = PostgresSQLHandler.default_schema_path()

        self.assertTrue(schema_path.exists())
        self.assertEqual(schema_path.name, "handler_kafka_schema.sql")

    @patch("src.infra.sql.psycopg.connect")
    def test_execute_query_returns_rows_for_with_update_returning_statements(self, connect_mock: MagicMock) -> None:
        cursor = MagicMock()
        cursor.description = [("id",)]
        cursor.fetchall.return_value = [(1,)]

        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.cursor.return_value.__enter__.return_value = cursor
        connect_mock.return_value = connection

        handler = PostgresSQLHandler(_StubConfigLoader(), logger=MagicMock())

        rows = handler.execute_query(
            "WITH candidate AS (SELECT 1) UPDATE kafka_events SET status = 'PROCESSING' RETURNING id;",
            params={"server_id": "edge-01"},
        )

        self.assertEqual(rows, [(1,)])
        cursor.fetchall.assert_called_once_with()
        connection.commit.assert_called_once_with()

    @patch("src.infra.sql.psycopg.connect")
    def test_execute_query_wraps_dict_params_for_json_columns(self, connect_mock: MagicMock) -> None:
        cursor = MagicMock()
        cursor.description = None

        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.cursor.return_value.__enter__.return_value = cursor
        connect_mock.return_value = connection

        handler = PostgresSQLHandler(_StubConfigLoader(), logger=MagicMock())
        handler.execute_query(
            "INSERT INTO kafka_events (payload) VALUES (%(payload)s)",
            params={"payload": {"model_name": "recognizer"}},
        )

        wrapped_payload = cursor.execute.call_args.args[1]["payload"]
        self.assertEqual(type(wrapped_payload).__name__, "Jsonb")


if __name__ == "__main__":
    unittest.main()
