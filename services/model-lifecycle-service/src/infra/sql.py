from pathlib import Path

import psycopg 
from psycopg.types.json import Jsonb

from src.utils.configloader import ConfigLoader
from src.utils.logger import Logger
from src.infra.config.postgrest import PostgrestConfig


class PostgresSQLHandler:
    def __init__(self, config_loader: ConfigLoader, logger: Logger):
        self.config_loader = config_loader
        self.sql_config: PostgrestConfig = self.config_loader.get_typed_config(PostgrestConfig, config_section="PostgresSql")
        self.dns = self._load_config()
        self.logger = logger

    @staticmethod
    def default_schema_path() -> Path:
        return Path(__file__).resolve().parents[4] / "postgres" / "sql" / "handler_kafka_schema.sql"

    def _load_config(self):
        dns = f"postgresql://{self.sql_config.user}:{self.sql_config.password}@{self.sql_config.host}:{self.sql_config.port}/{self.sql_config.database}"
        return dns

    def ensure_schema(self, schema_path: str | Path | None = None):
        schema_file = Path(schema_path) if schema_path is not None else self.default_schema_path()
        schema_sql = schema_file.read_text(encoding="utf-8")
        self.execute_query(schema_sql)

    def _normalize_value(self, params):
        if isinstance(params, (dict, list)):
            return Jsonb(params)
        return params

    def _normalize_params(self, params):
        if isinstance(params, dict):
            return {key: self._normalize_value(value) for key, value in params.items()}
        if isinstance(params, tuple):
            return tuple(self._normalize_value(value) for value in params)
        if isinstance(params, list):
            return [self._normalize_value(value) for value in params]
        return params
    
    def execute_query(self, query: str, params: tuple = None):
        try:
            with psycopg.connect(self.dns) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, self._normalize_params(params))
                    if cur.description is not None:
                        rows = cur.fetchall()
                        conn.commit()
                        return rows
                    conn.commit()
            self.logger.info(f"infra.sql Successfully executed query: {query} with params: {params}")
        except Exception as e:
            self.logger.error(f"infra.sql Error executing query: {e}")
            raise

if __name__ == "__main__":
    config_loader = ConfigLoader(
    env_file="/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus/services/model-lifecycle-service/config/.env", 
    config_yaml_file="/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus/services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml"
    )
    logger = Logger("PostgresSQLHandlerTest")
    sql_handler = PostgresSQLHandler(config_loader, logger)
#     sql_handler.ensure_schema()
#     result = sql_handler.execute_query("""
#     INSERT INTO kafka_events (
#     request_id,
#     topic,
#     partition_id,
#     message_offset,
#     consumer_group,
#     message_key,
#     event_type,
#     schema_name,
#     schema_version,
#     payload,
#     status,
#     produced_at
# )
# VALUES (
#     'train-20260601-001',
#     'model.lifecycle.events',
#     0,
#     12345,
#     'model-lifecycle-orchestrator',
#     'yolo_handwriting_ocr',
#     'MODEL_TRAINED',
#     'model_lifecycle_event',
#     'v1',
#     '{
#         \"request_id\": \"train-20260601-001\",
#         \"model_name\": \"yolo_handwriting_ocr\",
#         \"model_version\": \"v1.0.0\",
#         \"artifact_uri\": \"s3://artifacts/yolo_handwriting_ocr/v1.0.0/model.onnx\",
#         \"metrics\": {
#             \"val_accuracy\": 0.93,
#             \"val_loss\": 0.18
#         }
#     }'::jsonb,
#     'PROCESSED',
#     NOW()
# );  
#     """)
    # result = sql_handler.execute_query("SELECT * FROM kafka_events;")
    # print(result)

    # print("Cleaning up test data...")

    # delete_query = "DELETE FROM kafka_events WHERE request_id = 'train-20260601-001';"
    # sql_handler.execute_query(delete_query)

    # print("Test data cleaned up.")

    result = sql_handler.execute_query("SELECT * FROM kafka_events;")
    print(result)
