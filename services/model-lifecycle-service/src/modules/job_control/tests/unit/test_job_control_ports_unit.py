from __future__ import annotations

import sys
import unittest
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.job_control.application.dto.claimed_job_dto import ClaimedJobDto  # noqa: E402
from src.modules.job_control.application.dto.job_result_dto import JobResultDto  # noqa: E402
from src.modules.job_control.application.ports.job_claim_repository import (  # noqa: E402
    JobClaimRepository,
)
from src.modules.job_control.application.ports.job_handler import JobHandler  # noqa: E402
from src.modules.job_control.application.ports.job_record_repository import (  # noqa: E402
    JobRecordRepository,
)
from src.modules.job_control.domain.entity_objects.job_record import JobRecord  # noqa: E402
from src.modules.job_control.domain.value_objects.job_status import JobStatus  # noqa: E402
from src.modules.job_control.domain.value_objects.message_identity import MessageIdentity  # noqa: E402


class _RecordRepository:
    async def create_job_if_not_exists(
        self,
        *,
        job: JobRecord,
        schema_name: str | None,
        schema_version: str | None,
        message_key: str | None = None,
    ) -> bool:
        return True

    async def get_by_request_id(self, request_id: str) -> list[dict]:
        return [{"request_id": request_id}]

    async def mark_processed_by_request_id(self, request_id: str) -> bool:
        return True

    async def mark_failed_by_request_id(self, *, request_id: str, error_message: str) -> bool:
        return bool(request_id and error_message)


class _ClaimRepository:
    async def claim_next_pending_job(
        self,
        *,
        server_id: str,
    ) -> ClaimedJobDto | None:
        return ClaimedJobDto(
            request_id="req-1",
            event_type="yolo_detector",
            payload={"model_name": "yolo_detector"},
            status="PROCESSING",
        )


class _Handler:
    async def handle(self, job: ClaimedJobDto) -> JobResultDto:
        return JobResultDto(request_id=job.request_id, success=True)


class JobControlPortsUnitTest(unittest.TestCase):
    def test_job_record_repository_protocol_matches_postgres_repo_shape(self) -> None:
        self.assertIsInstance(_RecordRepository(), JobRecordRepository)

    def test_job_claim_repository_protocol_matches_claim_shape(self) -> None:
        self.assertIsInstance(_ClaimRepository(), JobClaimRepository)

    def test_job_handler_protocol_matches_execution_shape(self) -> None:
        self.assertIsInstance(_Handler(), JobHandler)

    def test_protocol_supporting_types_are_constructible(self) -> None:
        record = JobRecord(
            request_id="req-1",
            message_identity=MessageIdentity(
                topic="topic-a",
                partition_id=0,
                message_offset=1,
                consumer_group="group-a",
            ),
            event_type="yolo_detector",
            payload={"model_name": "yolo_detector"},
            status=JobStatus.RECEIVED,
        )
        self.assertEqual(record.request_id, "req-1")


if __name__ == "__main__":
    unittest.main()
