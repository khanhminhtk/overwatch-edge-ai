from __future__ import annotations

from dataclasses import dataclass

from src.proto import model_lifecycle_orchestrator_pb2


@dataclass(frozen=True)
class UploadFileRequestDTO:
    request_id: str
    request_ip: str
    bucket_name: str
    object_name: str
    version: str
    data_type: int
    source_url: str

    @classmethod
    def from_proto(
        cls,
        request: model_lifecycle_orchestrator_pb2.UploadFileRequest,
    ) -> "UploadFileRequestDTO":
        return cls(
            request_id=request.request_id,
            request_ip=request.request_ip,
            bucket_name=request.bucket_name,
            object_name=request.object_name,
            version=request.version,
            data_type=request.data_type,
            source_url=request.source_url,
        )

    def to_proto(self) -> model_lifecycle_orchestrator_pb2.UploadFileRequest:
        return model_lifecycle_orchestrator_pb2.UploadFileRequest(
            request_id=self.request_id,
            request_ip=self.request_ip,
            bucket_name=self.bucket_name,
            object_name=self.object_name,
            version=self.version,
            data_type=self.data_type,
            source_url=self.source_url,
        )


@dataclass(frozen=True)
class UploadFileResponseDTO:
    request_id: str
    code: int
    message: str
    status: int

    def to_proto(self) -> model_lifecycle_orchestrator_pb2.UploadFileResponse:
        return model_lifecycle_orchestrator_pb2.UploadFileResponse(
            request_id=self.request_id,
            code=self.code,
            message=self.message,
            status=self.status,
        )
