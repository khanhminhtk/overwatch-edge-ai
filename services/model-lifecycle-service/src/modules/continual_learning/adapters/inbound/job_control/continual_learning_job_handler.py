from __future__ import annotations

import asyncio

from src.modules.continual_learning.application.process_raw_images import ProcessRawImages
from src.modules.job_control.application.dto.claimed_job_dto import ClaimedJobDto
from src.modules.job_control.application.dto.job_result_dto import JobResultDto
from src.platform.logger import Logger
from src.platform.vision import VisionModelPort


class ContinualLearningJobHandler:
    def __init__(self, vision_model: VisionModelPort, default_raw_dir: str = "data/data_continue_learning/raw", default_output_dir: str = "data/data_continue_learning/processed", logger: Logger | None = None) -> None:
        self._vision_model = vision_model
        self._default_raw_dir = default_raw_dir
        self._default_output_dir = default_output_dir
        self._logger = logger or Logger("ContinualLearningJobHandler")
        self._processor = ProcessRawImages(vision_model=vision_model, logger=Logger("ProcessRawImages"))

    async def handle(self, job: ClaimedJobDto) -> JobResultDto:
        payload = job.payload or {}
        raw_dir = payload.get("raw_dir", self._default_raw_dir)
        output_dir = payload.get("output_dir", self._default_output_dir)
        class_id = int(payload.get("class_id", 0))

        try:
            self._logger.info("[CL_JOB_STARTED]", f"request_id={job.request_id}", f"raw_dir={raw_dir}", f"output_dir={output_dir}")
            await asyncio.to_thread(self._processor.execute, raw_dir=raw_dir, output_dir=output_dir, class_id=class_id)
        except Exception as exc:
            self._logger.exception("[CL_JOB_FAILED]", f"request_id={job.request_id}")
            return JobResultDto(request_id=job.request_id, success=False, error_message=str(exc))

        self._logger.info("[CL_JOB_SUCCEEDED]", f"request_id={job.request_id}")
        return JobResultDto(request_id=job.request_id, success=True)
