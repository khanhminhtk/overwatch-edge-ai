from __future__ import annotations

from dataclasses import dataclass

from src.applications.dtos.download_file import (
    DownloadFileRequestDTO,
    DownloadFileResponseDTO,
)
from src.applications.ports.download_transport_port import DownloadTransportPort
from src.applications.ports.logger_port import LoggerPort
from src.applications.ports.upload_gateway_port import UploadGatewayPort
from src.domain.errors.upload_flow_error import UploadFlowError
from src.domain.value_objects.dataset_upload_spec import DatasetUploadSpec
from src.domain.value_objects.object_storage_target import (
    build_download_object_name,
    finalize_storage_object_name,
    normalize_object_storage_target,
)
from src.proto import model_lifecycle_orchestrator_pb2


@dataclass(frozen=True)
class DownloadContext:
    request_id: str
    request_ip: str
    bucket_name: str
    object_name: str
    version: str
    data_type: int
    destination_path: str

    @classmethod
    def from_request(cls, request: DownloadFileRequestDTO) -> "DownloadContext":
        return cls(
            request_id=request.request_id,
            request_ip=request.request_ip,
            bucket_name=request.bucket_name,
            object_name=request.object_name,
            version=request.version,
            data_type=request.data_type,
            destination_path=request.destination_path,
        )


class DownloadHandler:
    def __init__(
        self,
        gateway: UploadGatewayPort,
        downloader: DownloadTransportPort,
        logger: LoggerPort,
        spec: DatasetUploadSpec,
    ):
        self._gateway = gateway
        self._downloader = downloader
        self._logger = logger
        self._spec = spec

    def execute(self, request: DownloadFileRequestDTO) -> DownloadFileResponseDTO:
        context = DownloadContext.from_request(request)
        self._logger.info(
            "download.start request_id=%s request_ip=%s data_type=%s bucket=%s object_name=%s version=%s",
            context.request_id,
            context.request_ip,
            context.data_type,
            context.bucket_name,
            context.object_name,
            context.version,
        )

        try:
            self._gateway.check(self._spec)
            self._logger.debug("download.health.check.ok address=%s", self._spec.address)

            access_token = self._gateway.refresh_access_token(context.request_id, self._spec)
            self._logger.debug("download.access_token.ok request_id=%s", context.request_id)

            bucket_name, object_name = normalize_object_storage_target(
                context.bucket_name,
                context.object_name,
            )
            source_path = self._spec.resolve_source_path(context.data_type)
            download_object_name = build_download_object_name(
                object_name,
                context.version,
            )
            download_object_name = finalize_storage_object_name(download_object_name, source_path)
            self._logger.debug(
                "download.target.normalized bucket_name=%s object_name=%s",
                bucket_name,
                download_object_name,
            )

            download_url = self._gateway.generate_download_url(
                request_id=context.request_id,
                bucket_name=bucket_name,
                object_name=download_object_name,
                access_token=access_token,
                version="",
                spec=self._spec,
            )
            self._logger.debug("download.url.generated request_id=%s", context.request_id)

            self._downloader.download(
                download_url,
                context.destination_path,
                download_object_name,
            )

            self._logger.info("download.completed request_id=%s", context.request_id)
            return DownloadFileResponseDTO(
                request_id=context.request_id,
                code=200,
                message="Download completed successfully",
                status=model_lifecycle_orchestrator_pb2.STATUS_OK,
            )
        except UploadFlowError as exc:
            self._logger.warning(
                "download.failed request_id=%s code=%s message=%s",
                context.request_id,
                exc.code,
                exc.message,
            )
            return DownloadFileResponseDTO(
                request_id=context.request_id,
                code=exc.code,
                message=exc.message,
                status=model_lifecycle_orchestrator_pb2.STATUS_ERROR,
            )
        except Exception as exc:
            self._logger.exception("download.failed.unexpected request_id=%s", context.request_id)
            return DownloadFileResponseDTO(
                request_id=context.request_id,
                code=500,
                message=str(exc),
                status=model_lifecycle_orchestrator_pb2.STATUS_ERROR,
            )
