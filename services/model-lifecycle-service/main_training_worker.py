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
class WorkerRuntime:
    worker: Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run training worker for queued jobs")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--config-file", default=DEFAULT_CONFIG_FILE)
    parser.add_argument(
        "--server-id",
        default=os.getenv("TRAINING_SERVER_ID", socket.gethostname()),
    )
    parser.add_argument(
        "--event-topic",
        default=os.getenv("TRAINING_EVENT_TOPIC", "training.events"),
    )
    parser.add_argument(
        "--training-mode",
        default=os.getenv("TRAINING_MODE", "local"),
    )
    parser.add_argument(
        "--idle-sleep-seconds",
        type=float,
        default=float(os.getenv("TRAINING_IDLE_SLEEP_SECONDS", "2.0")),
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
    )
    return parser


def create_worker_runtime(
    *,
    env_file: str,
    config_file: str,
    server_id: str,
    event_topic: str,
    training_mode: str,
    idle_sleep_seconds: float,
    logger: Logger,
) -> WorkerRuntime:
    from src.applications.use_cases.training_job_repository import TrainingJobRepository
    from src.applications.use_cases.training_worker import TrainingWorker
    from src.infra.kafka_producer import KafkaProducerClient
    from src.infra.sql import PostgresSQLHandler

    config_loader = ConfigLoader(env_file=env_file, config_yaml_file=config_file)
    sql_handler = PostgresSQLHandler(config_loader=config_loader, logger=logger)
    sql_handler.ensure_schema()
    job_repository = TrainingJobRepository(sql_handler=sql_handler, logger=logger)
    producer = KafkaProducerClient(logger=logger, config_loader=config_loader)
    worker = TrainingWorker(
        job_repository=job_repository,
        producer=producer,
        logger=logger,
        server_id=server_id,
        event_topic=event_topic,
        training_mode=training_mode,
        idle_sleep_seconds=idle_sleep_seconds,
    )
    return WorkerRuntime(worker=worker)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    Logger.configure(LoggerConfig(level=args.log_level))
    logger = Logger(__name__)
    runtime = create_worker_runtime(
        env_file=args.env_file,
        config_file=args.config_file,
        server_id=args.server_id,
        event_topic=args.event_topic,
        training_mode=args.training_mode,
        idle_sleep_seconds=args.idle_sleep_seconds,
        logger=logger,
    )
    logger.info(
        "[TRAINING_WORKER_STARTING]",
        f"server_id={args.server_id}",
        f"event_topic={args.event_topic}",
        f"training_mode={args.training_mode}",
    )
    runtime.worker.run_forever()


if __name__ == "__main__":
    main()
