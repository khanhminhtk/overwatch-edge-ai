from __future__ import annotations

from dataclasses import dataclass

from src.proto import model_lifecycle_orchestrator_pb2


@dataclass(frozen=True)
class HealthCheckRequestDTO:
    request_id: str

    @classmethod
    def from_proto(
        cls,
        request: model_lifecycle_orchestrator_pb2.HealthCheckRequest,
    ) -> "HealthCheckRequestDTO":
        return cls(request_id=request.request_id)

    def to_proto(self) -> model_lifecycle_orchestrator_pb2.HealthCheckRequest:
        return model_lifecycle_orchestrator_pb2.HealthCheckRequest(request_id=self.request_id)


@dataclass(frozen=True)
class HealthCheckResponseDTO:
    request_id: str
    code: int
    message: str
    status: int

    def to_proto(self) -> model_lifecycle_orchestrator_pb2.HealthCheckResponse:
        return model_lifecycle_orchestrator_pb2.HealthCheckResponse(
            request_id=self.request_id,
            code=self.code,
            message=self.message,
            status=self.status,
        )
