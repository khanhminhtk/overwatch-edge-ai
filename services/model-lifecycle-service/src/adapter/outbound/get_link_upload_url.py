from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import grpc
import requests

from src.applications.ports.download_transport_port import DownloadTransportPort
from src.applications.ports.logger_port import LoggerPort
from src.applications.ports.upload_gateway_port import UploadGatewayPort
from src.applications.ports.upload_transport_port import UploadTransportPort
from src.applications.services.download_handler import DownloadHandler
from src.applications.services.upload_handler import UploadHandler
from src.domain.errors.upload_flow_error import UploadFlowError
from src.domain.value_objects.dataset_upload_spec import DatasetUploadSpec
from src.domain.value_objects.object_storage_target import (
    build_upload_object_name,
    normalize_object_storage_target,
)
from src.infra.config.grpc import ObjectStorageRuntimeConfig
from src.infra.upload_file import upload_file
from src.proto import database_service_pb2
from src.proto import database_service_pb2_grpc
from src.proto import health_pb2
from src.proto import health_pb2_grpc
from src.utils.logger import Logger
from src.utils.zipfiledata import zip_directory


OBJECT_STORAGE_HEALTH_SERVICE = "objectstorage.ObjectStorageService"


MinioServiceConfig = DatasetUploadSpec


class GrpcChannelFactory:
    def create(self, config: DatasetUploadSpec) -> grpc.Channel:
        if config.ca_cert_path:
            with open(config.ca_cert_path, "rb") as cert_file:
                credentials = grpc.ssl_channel_credentials(cert_file.read())
            return grpc.secure_channel(config.address, credentials)
        return grpc.insecure_channel(config.address)


class MinioHealthClient:
    def __init__(self, channel_factory: GrpcChannelFactory | None = None):
        self._channel_factory = channel_factory or GrpcChannelFactory()

    def check(self, config: DatasetUploadSpec) -> None:
        try:
            with self._channel_factory.create(config) as channel:
                stub = health_pb2_grpc.HealthStub(channel)
                response = stub.Check(
                    health_pb2.HealthCheckRequest(service=config.health_service_name)
                )
        except grpc.RpcError as exc:
            raise UploadFlowError(503, f"Downstream health check failed: {exc.details()}") from exc

        if response.status != health_pb2.HealthCheckResponse.SERVING:
            raise UploadFlowError(503, "Downstream health is not serving")


class MinioAuthClient:
    def __init__(self, channel_factory: GrpcChannelFactory | None = None):
        self._channel_factory = channel_factory or GrpcChannelFactory()

    def refresh_access_token(self, request_id: str, config: DatasetUploadSpec) -> str:
        try:
            with self._channel_factory.create(config) as channel:
                stub = database_service_pb2_grpc.ObjectStorageServiceStub(channel)
                response = stub.RefreshAccessToken(
                    database_service_pb2.RefreshAccessTokenRequest(
                        request_id=request_id,
                        refresh_token=config.refresh_token,
                    )
                )
        except grpc.RpcError as exc:
            raise UploadFlowError(503, f"Refresh access token RPC failed: {exc.details()}") from exc

        if not response.success or not response.access_token:
            raise UploadFlowError(401, response.message or "Failed to refresh access token")
        return response.access_token


class MinioUploadUrlClient:
    def __init__(self, channel_factory: GrpcChannelFactory | None = None):
        self._channel_factory = channel_factory or GrpcChannelFactory()

    def generate_upload_url(
        self,
        request_id: str,
        bucket_name: str,
        object_name: str,
        access_token: str,
        version: str,
        config: DatasetUploadSpec,
    ) -> str:
        metadata = (("authorization", f"Bearer {access_token}"),)
        if version:
            object_name = f"{version}/{object_name}"

        try:
            with self._channel_factory.create(config) as channel:
                stub = database_service_pb2_grpc.ObjectStorageServiceStub(channel)
                response = stub.GenerateUploadUrl(
                    database_service_pb2.GenerateUploadUrlRequest(
                        request_id=request_id,
                        bucket_name=bucket_name,
                        object_name=object_name,
                    ),
                    metadata=metadata,
                )
        except grpc.RpcError as exc:
            raise UploadFlowError(503, f"Generate upload URL RPC failed: {exc.details()}") from exc

        if not response.status or not response.url:
            raise UploadFlowError(502, "Failed to generate upload URL")
        return response.url


class MinioDownloadUrlClient:
    def __init__(self, channel_factory: GrpcChannelFactory | None = None):
        self._channel_factory = channel_factory or GrpcChannelFactory()

    def generate_download_url(
        self,
        request_id: str,
        bucket_name: str,
        object_name: str,
        access_token: str,
        version: str,
        config: DatasetUploadSpec,
    ) -> str:
        metadata = (("authorization", f"Bearer {access_token}"),)
        if version:
            object_name = f"{version}/{object_name}"

        try:
            with self._channel_factory.create(config) as channel:
                stub = database_service_pb2_grpc.ObjectStorageServiceStub(channel)
                response = stub.GenerateDownloadUrl(
                    database_service_pb2.GenerateDownloadUrlRequest(
                        request_id=request_id,
                        bucket_name=bucket_name,
                        object_name=object_name,
                    ),
                    metadata=metadata,
                )
        except grpc.RpcError as exc:
            raise UploadFlowError(503, f"Generate download URL RPC failed: {exc.details()}") from exc

        if not response.status or not response.url:
            raise UploadFlowError(502, "Failed to generate download URL")
        return response.url


class MinioUploadGateway(UploadGatewayPort):
    def __init__(
        self,
        health_client: MinioHealthClient | None = None,
        auth_client: MinioAuthClient | None = None,
        upload_url_client: MinioUploadUrlClient | None = None,
    ):
        self._health_client = health_client or MinioHealthClient()
        self._auth_client = auth_client or MinioAuthClient()
        self._upload_url_client = upload_url_client or MinioUploadUrlClient()
        self._download_url_client = MinioDownloadUrlClient()

    def check(self, spec: DatasetUploadSpec) -> None:
        self._health_client.check(spec)

    def refresh_access_token(self, request_id: str, spec: DatasetUploadSpec) -> str:
        return self._auth_client.refresh_access_token(request_id, spec)

    def generate_upload_url(
        self,
        request_id: str,
        bucket_name: str,
        object_name: str,
        access_token: str,
        version: str,
        spec: DatasetUploadSpec,
    ) -> str:
        return self._upload_url_client.generate_upload_url(
            request_id=request_id,
            bucket_name=bucket_name,
            object_name=object_name,
            access_token=access_token,
            version=version,
            config=spec,
        )

    def generate_download_url(
        self,
        request_id: str,
        bucket_name: str,
        object_name: str,
        access_token: str,
        version: str,
        spec: DatasetUploadSpec,
    ) -> str:
        return self._download_url_client.generate_download_url(
            request_id=request_id,
            bucket_name=bucket_name,
            object_name=object_name,
            access_token=access_token,
            version=version,
            config=spec,
        )


class LocalUploadTransport(UploadTransportPort):
    def upload(self, file_path: str, upload_url: str) -> None:
        path = Path(file_path)
        if not path.exists():
            raise UploadFlowError(404, f"File not found: {file_path}")

        response = upload_file(
            file_path=file_path,
            url=upload_url,
            headers={"Content-Type": "application/octet-stream"},
        )
        if response.status_code < 200 or response.status_code >= 300:
            raise UploadFlowError(502, f"Upload failed with HTTP {response.status_code}")

    def upload_folder(self, folder_path: str, upload_url: str) -> None:
        zip_path = f"{folder_path}.zip"
        try:
            zip_directory(folder_path, zip_path)
            upload_file(
                file_path=zip_path,
                url=upload_url,
                headers={"Content-Type": "application/zip"},
            )
        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)


class LocalDownloadTransport(DownloadTransportPort):
    def download(self, download_url: str, destination_path: str, object_name: str) -> None:
        destination = Path(destination_path)
        if destination.exists() and destination.is_dir():
            target_path = self._resolve_target_file(destination, object_name)
        elif destination.suffix == "":
            target_path = self._resolve_target_file(destination, object_name)
        else:
            target_path = destination

        target_path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(download_url, stream=True, timeout=60)
        if response.status_code < 200 or response.status_code >= 300:
            raise UploadFlowError(502, f"Download failed with HTTP {response.status_code}")

        with open(target_path, "wb") as file_handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    file_handle.write(chunk)

    def _resolve_target_file(self, destination: Path, object_name: str) -> Path:
        filename = self._filename_from_object_name(object_name)
        if not filename:
            raise UploadFlowError(400, "Unable to determine downloaded file name")
        return destination / filename

    def _filename_from_object_name(self, object_name: str) -> str:
        return Path(object_name).name.strip()


class FilePathResolver:
    def __init__(self, config_or_spec: Mapping[str, str] | DatasetUploadSpec):
        if isinstance(config_or_spec, DatasetUploadSpec):
            self._spec = config_or_spec
        else:
            self._spec = DatasetUploadSpec.from_mapping(config_or_spec)

    def resolve(self, data_type: int) -> str:
        return self._spec.resolve_source_path(data_type)


class UploadFileOrchestrator(UploadHandler):
    def __init__(
        self,
        gateway: UploadGatewayPort,
        uploader: UploadTransportPort,
        spec: DatasetUploadSpec,
        logger: LoggerPort | None = None,
    ):
        super().__init__(
            gateway=gateway,
            uploader=uploader,
            logger=logger or Logger(__name__),
            spec=spec,
        )


class DownloadFileOrchestrator(DownloadHandler):
    def __init__(
        self,
        gateway: UploadGatewayPort,
        downloader: DownloadTransportPort,
        spec: DatasetUploadSpec,
        logger: LoggerPort | None = None,
    ):
        super().__init__(
            gateway=gateway,
            downloader=downloader,
            logger=logger or Logger(__name__),
            spec=spec,
        )


def build_minio_service_config(
    config: Mapping[str, str] | DatasetUploadSpec | ObjectStorageRuntimeConfig,
) -> DatasetUploadSpec:
    if isinstance(config, DatasetUploadSpec):
        return config
    if isinstance(config, ObjectStorageRuntimeConfig):
        return config.to_dataset_upload_spec()
    return DatasetUploadSpec.from_mapping(config)


def build_upload_orchestrator(
    config: Mapping[str, str] | DatasetUploadSpec | ObjectStorageRuntimeConfig,
    logger: LoggerPort | None = None,
) -> UploadHandler:
    upload_spec = build_minio_service_config(config)
    return UploadFileOrchestrator(
        gateway=MinioUploadGateway(
            health_client=MinioHealthClient(),
            auth_client=MinioAuthClient(),
            upload_url_client=MinioUploadUrlClient(),
        ),
        uploader=LocalUploadTransport(),
        spec=upload_spec,
        logger=logger or Logger(__name__),
    )


def build_download_orchestrator(
    config: Mapping[str, str] | DatasetUploadSpec | ObjectStorageRuntimeConfig,
    logger: LoggerPort | None = None,
) -> DownloadHandler:
    download_spec = build_minio_service_config(config)
    return DownloadFileOrchestrator(
        gateway=MinioUploadGateway(
            health_client=MinioHealthClient(),
            auth_client=MinioAuthClient(),
            upload_url_client=MinioUploadUrlClient(),
        ),
        downloader=LocalDownloadTransport(),
        spec=download_spec,
        logger=logger or Logger(__name__),
    )
