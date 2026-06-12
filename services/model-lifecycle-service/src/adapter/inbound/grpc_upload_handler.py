from __future__ import annotations

import grpc

from src.applications.dtos.download_file import (
    DownloadFileRequestDTO,
    DownloadFileResponseDTO,
)
from src.applications.dtos.health import HealthCheckRequestDTO, HealthCheckResponseDTO
from src.applications.dtos.upload_file import UploadFileRequestDTO, UploadFileResponseDTO
from src.adapter.outbound.get_link_upload_url import (
    DownloadFileOrchestrator,
    UploadFileOrchestrator,
)
from src.proto import model_lifecycle_orchestrator_pb2
from src.proto import model_lifecycle_orchestrator_pb2_grpc


class ModelLifecycleOrchestratorHandler(
    model_lifecycle_orchestrator_pb2_grpc.ModelLifecycleOrchestratorServiceServicer
):
    def __init__(
        self,
        upload_orchestrator: UploadFileOrchestrator,
        download_orchestrator: DownloadFileOrchestrator | None = None,
    ):
        self._upload_orchestrator = upload_orchestrator
        self._download_orchestrator = download_orchestrator

    def UploadFile(self, request, context):
        result: UploadFileResponseDTO = self._upload_orchestrator.execute(
            UploadFileRequestDTO.from_proto(request)
        )
        return result.to_proto()

    def DownloadFile(self, request, context):
        if self._download_orchestrator is None:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("Download orchestrator is not configured")
            raise NotImplementedError("Download orchestrator is not configured")
        result: DownloadFileResponseDTO = self._download_orchestrator.execute(
            DownloadFileRequestDTO.from_proto(request)
        )
        return result.to_proto()

    def HealthCheck(self, request, context):
        response = HealthCheckResponseDTO(
            request_id=HealthCheckRequestDTO.from_proto(request).request_id,
            code=200,
            message="Service is healthy",
            status=model_lifecycle_orchestrator_pb2.STATUS_OK,
        )
        return response.to_proto()
