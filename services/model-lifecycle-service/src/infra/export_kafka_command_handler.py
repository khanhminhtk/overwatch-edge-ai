from typing import TYPE_CHECKING

from src.applications.dtos.events.events_kafka_export import ExportRequestedEvent, PayloadExportJob
from src.applications.use_cases.export_job_repository import ExportJobRepository
from src.infra.base_kafka_command_handler import BaseKafkaCommandHandler
from src.utils.logger import Logger

if TYPE_CHECKING:
    from src.infra.kafka_producer import KafkaProducerClient


class ExportKafkaCommandHandler(BaseKafkaCommandHandler[ExportRequestedEvent, PayloadExportJob]):
    def __init__(
        self,
        job_repository: ExportJobRepository,
        producer: "KafkaProducerClient",
        logger: Logger,
        consumer_group: str,
        server_id: str,
        dlq_topic: str,
    ) -> None:
        super().__init__(
            event_type="export_requested",
            event_model=ExportRequestedEvent,
            payload_model=PayloadExportJob,
            job_repository=job_repository,
            producer=producer,
            logger=logger,
            consumer_group=consumer_group,
            server_id=server_id,
            dlq_topic=dlq_topic,
        )

    def log_created(
        self,
        *,
        event: ExportRequestedEvent,
        payload_model: PayloadExportJob,
    ) -> None:
        self._logger.info(
            "[EXPORT_JOB_CREATED]",
            f"request_id={event.request_id}",
            f"model_name={payload_model.model_name}",
            f"model_version={payload_model.model_version}",
        )

    def log_duplicate(self, *, event: ExportRequestedEvent) -> None:
        self._logger.info(
            "[EXPORT_JOB_DUPLICATE_SKIP]",
            f"request_id={event.request_id}",
        )
