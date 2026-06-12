from __future__ import annotations

import subprocess
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.applications.use_cases.base_job_worker import BaseJobWorker
from src.applications.use_cases.job_repository import BaseJobRepository
from src.utils.logger import Logger

if TYPE_CHECKING:
    from src.infra.kafka_producer import KafkaProducerClient


class BaseWorkerBashScript(BaseJobWorker):
    def __init__(
        self,
        *,
        job_repository: BaseJobRepository,
        producer: "KafkaProducerClient",
        logger: Logger,
        server_id: str,
        event_topic: str,
        training_mode: str = "local",
        idle_sleep_seconds: float = 2.0,
    ) -> None:
        super().__init__(
            job_repository=job_repository,
            producer=producer,
            logger=logger,
            server_id=server_id,
            event_topic=event_topic,
            idle_sleep_seconds=idle_sleep_seconds,
        )
        self._training_mode = training_mode
        self._subprocess_run = subprocess.run

    def _run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        script_name = self._resolve_script_name(job)
        script_path = self._scripts_root() / script_name
        self._subprocess_run(
            ["bash", str(script_path), "--mode", self._training_mode],
            check=True,
            text=True,
            capture_output=True,
        )
        return self._build_job_result(job)

    @staticmethod
    def _scripts_root() -> Path:
        return Path(__file__).resolve().parents[5] / "ml" / "training" / "scripts"

    @abstractmethod
    def _resolve_script_name(self, job: dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def _build_job_result(self, job: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
