from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.applications.dtos.upload_file import (
    UploadFileRequestDTO,
    UploadFileResponseDTO,
)
from src.applications.ports.logger_port import LoggerPort
from src.applications.ports.upload_gateway_port import UploadGatewayPort
from src.applications.ports.upload_transport_port import UploadTransportPort
from src.domain.errors.upload_flow_error import UploadFlowError
from src.domain.value_objects.dataset_upload_spec import DatasetUploadSpec
from src.domain.value_objects.object_storage_target import (
    build_upload_object_name,
    finalize_storage_object_name,
    normalize_object_storage_target,
)
from src.proto import model_lifecycle_orchestrator_pb2


@dataclass(frozen=True)
class UploadContext:
    request_id: str
    request_ip: str
    bucket_name: str
    object_name: str
    version: str
    data_type: int
    source_url: str

    @classmethod
    def from_request(cls, request: UploadFileRequestDTO) -> "UploadContext":
        return cls(
            request_id=request.request_id,
            request_ip=request.request_ip,
            bucket_name=request.bucket_name,
            object_name=request.object_name,
            version=request.version,
            data_type=request.data_type,
            source_url=request.source_url,
        )


class UploadHandler:
    def __init__(
        self,
        gateway: UploadGatewayPort,
        uploader: UploadTransportPort,
        logger: LoggerPort,
        spec: DatasetUploadSpec,
    ):
        self._gateway = gateway
        self._uploader = uploader
        self._logger = logger
        self._spec = spec

    def execute(self, request: UploadFileRequestDTO) -> UploadFileResponseDTO:
        context = UploadContext.from_request(request)
        self._logger.info(
            "upload.start request_id=%s request_ip=%s data_type=%s bucket=%s object_name=%s version=%s",
            context.request_id,
            context.request_ip,
            context.data_type,
            context.bucket_name,
            context.object_name,
            context.version,
        )

        try:
            self._gateway.check(self._spec)
            self._logger.debug("upload.health.check.ok address=%s", self._spec.address)

            access_token = self._gateway.refresh_access_token(context.request_id, self._spec)
            self._logger.debug("upload.access_token.ok request_id=%s", context.request_id)

            bucket_name, object_name = normalize_object_storage_target(
                context.bucket_name,
                context.object_name,
            )
            source_path = self._spec.resolve_source_path(context.data_type)
            self._logger.debug("upload.source.resolved source_path=%s", source_path)

            upload_object_name = build_upload_object_name(
                object_name,
                context.version,
                context.source_url,
                source_path=source_path,
            )
            upload_object_name = finalize_storage_object_name(upload_object_name, source_path)
            self._logger.debug(
                "upload.target.normalized bucket_name=%s object_name=%s",
                bucket_name,
                upload_object_name,
            )

            upload_url = self._gateway.generate_upload_url(
                request_id=context.request_id,
                bucket_name=bucket_name,
                object_name=upload_object_name,
                access_token=access_token,
                version="",
                spec=self._spec,
            )
            self._logger.debug("upload.url.generated request_id=%s", context.request_id)

            if os.path.isdir(source_path):
                self._uploader.upload_folder(source_path, upload_url)
            else:
                self._uploader.upload(source_path, upload_url)

            self._logger.info("upload.completed request_id=%s", context.request_id)
            return UploadFileResponseDTO(
                request_id=context.request_id,
                code=200,
                message="Upload completed successfully",
                status=model_lifecycle_orchestrator_pb2.STATUS_OK,
            )
        except UploadFlowError as exc:
            self._logger.warning(
                "upload.failed request_id=%s code=%s message=%s",
                context.request_id,
                exc.code,
                exc.message,
            )
            return UploadFileResponseDTO(
                request_id=context.request_id,
                code=exc.code,
                message=exc.message,
                status=model_lifecycle_orchestrator_pb2.STATUS_ERROR,
            )
        except Exception as exc:
            self._logger.exception("upload.failed.unexpected request_id=%s", context.request_id)
            return UploadFileResponseDTO(
                request_id=context.request_id,
                code=500,
                message=str(exc),
                status=model_lifecycle_orchestrator_pb2.STATUS_ERROR,
            )
