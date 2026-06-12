from __future__ import annotations

import argparse
import os
import socket
from dataclasses import dataclass
from typing import Any, Sequence

from src.utils.configloader import ConfigLoader
from src.utils.logger import Logger, LoggerConfig


DEFAULT_ENV_FILE = "config/.env"
DEFAULT_CONFIG_FILE = "config/model_lifecycle_orchestrator_config.yaml"


@dataclass(frozen=True)
class ConsumerRuntime:
    consumer: Any
    handler: Any
    shutdown: Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Kafka consumer for export job ingestion")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--config-file", default=DEFAULT_CONFIG_FILE)
    parser.add_argument(
        "--job-key",
        default=os.getenv("EXPORT_CONSUMER_JOB_KEY", "export_recognizer_tensorrt"),
    )
    parser.add_argument(
        "--server-id",
        default=os.getenv("EXPORT_SERVER_ID", socket.gethostname()),
    )
    parser.add_argument(
        "--dlq-topic",
        default=os.getenv("EXPORT_DLQ_TOPIC", "export.dlq"),
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
    )
    parser.add_argument(
        "--from-beginning",
        action="store_true",
        help="Replay from the earliest offset for assigned partitions, ignoring committed offsets.",
    )
    return parser


def create_consumer_runtime(
    *,
    env_file: str,
    config_file: str,
    job_key: str,
    server_id: str,
    dlq_topic: str,
    logger: Logger,
    from_beginning: bool = False,
) -> ConsumerRuntime:
    from src.applications.use_cases.export_job_repository import ExportJobRepository
    from src.infra.config.kafka import KafkaConfig
    from src.infra.export_kafka_command_handler import ExportKafkaCommandHandler
    from src.infra.kafka_consumer import GracefulShutdown, KafkaConsumerClient
    from src.infra.kafka_producer import KafkaProducerClient
    from src.infra.sql import PostgresSQLHandler

    config_loader = ConfigLoader(env_file=env_file, config_yaml_file=config_file)
    kafka_config: KafkaConfig = config_loader.get_typed_config(KafkaConfig)

    sql_handler = PostgresSQLHandler(config_loader=config_loader, logger=logger)
    sql_handler.ensure_schema()
    job_repository = ExportJobRepository(sql_handler=sql_handler, logger=logger)
    producer = KafkaProducerClient(logger=logger, config_loader=config_loader)
    handler = ExportKafkaCommandHandler(
        job_repository=job_repository,
        producer=producer,
        logger=logger,
        consumer_group=kafka_config.jobs[job_key].group_id,
        server_id=server_id,
        dlq_topic=dlq_topic,
    )
    consumer = KafkaConsumerClient(
        config_loader=config_loader,
        job_key=job_key,
        logger=logger,
        from_beginning=from_beginning,
    )
    shutdown = GracefulShutdown()
    shutdown.install()
    return ConsumerRuntime(consumer=consumer, handler=handler, shutdown=shutdown)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    Logger.configure(LoggerConfig(level=args.log_level))
    logger = Logger(__name__)
    runtime = create_consumer_runtime(
        env_file=args.env_file,
        config_file=args.config_file,
        job_key=args.job_key,
        server_id=args.server_id,
        dlq_topic=args.dlq_topic,
        logger=logger,
        from_beginning=args.from_beginning,
    )
    logger.info(
        "[EXPORT_KAFKA_CONSUMER_STARTING]",
        f"job_key={args.job_key}",
        f"server_id={args.server_id}",
        f"dlq_topic={args.dlq_topic}",
    )
    runtime.consumer.run(runtime.handler.handle_message, runtime.shutdown)


if __name__ == "__main__":
    main()
