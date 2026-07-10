from __future__ import annotations

from src.modules.job_control.adapters.outbound.scheduler.polling_job_runner import (
    PollingJobRunner,
)
from src.platform.logger import Logger


class MlflowTrackingProcessor:
    def __init__(
        self,
        *,
        runner: PollingJobRunner,
        logger: Logger | None = None,
    ) -> None:
        self._runner = runner
        self._logger = logger

    async def run_once(self):
        return await self._runner.run_once()

    async def run_forever(self) -> None:
        await self._runner.run_forever()

    def stop(self) -> None:
        self._runner.stop()


RecognizerTrackingJobProcessor = MlflowTrackingProcessor
