from typing import TYPE_CHECKING

from src.applications.dtos.events.events_kafka_training import PayloadTrainingJob, TrainRequestedEvent
from src.applications.use_cases.training_job_repository import TrainingJobRepository
from src.infra.base_kafka_command_handler import BaseKafkaCommandHandler
from src.utils.logger import Logger

if TYPE_CHECKING:
    from src.infra.kafka_producer import KafkaProducerClient


class KafkaCommandHandler(BaseKafkaCommandHandler[TrainRequestedEvent, PayloadTrainingJob]):
    def __init__(
        self,
        job_repository: TrainingJobRepository,
        producer: "KafkaProducerClient",
        logger: Logger,
        consumer_group: str,
        server_id: str,
        dlq_topic: str,
    ) -> None:
        super().__init__(
            event_type="train_requested",
            event_model=TrainRequestedEvent,
            payload_model=PayloadTrainingJob,
            job_repository=job_repository,
            producer=producer,
            logger=logger,
            consumer_group=consumer_group,
            server_id=server_id,
            dlq_topic=dlq_topic,
        )

    def should_skip_event(
        self,
        *,
        event: TrainRequestedEvent,
        payload_model: PayloadTrainingJob,
    ) -> bool:
        target_server_id = payload_model.target_server_id
        if target_server_id is not None and target_server_id != self._server_id:
            self._logger.info(
                "[SKIP_TARGET_SERVER]",
                f"request_id={event.request_id}",
                f"target={target_server_id}",
                f"current={self._server_id}",
            )
            return True
        return False

    def log_created(
        self,
        *,
        event: TrainRequestedEvent,
        payload_model: PayloadTrainingJob,
    ) -> None:
        self._logger.info(
            "[JOB_CREATED]",
            f"request_id={event.request_id}",
            f"model_name={payload_model.model_name}",
            f"dataset_version={payload_model.dataset_version}",
        )

    def log_duplicate(self, *, event: TrainRequestedEvent) -> None:
        self._logger.info(
            "[JOB_DUPLICATE_SKIP]",
            f"request_id={event.request_id}",
        )
