from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.domain.value_objects.dataset_upload_spec import DatasetUploadSpec


@dataclass(frozen=True)
class GrpcServerConfig:
    grpc_model_lifecycle_orchestrator_host: str = "0.0.0.0"
    grpc_model_lifecycle_orchestrator_port: int = 50051

    @property
    def host(self) -> str:
        return self.grpc_model_lifecycle_orchestrator_host

    @property
    def port(self) -> int:
        return self.grpc_model_lifecycle_orchestrator_port

    @property
    def listen_address(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass(frozen=True)
class ObjectStorageRuntimeConfig:
    object_storage_grpc_host: str = "127.0.0.1"
    object_storage_grpc_port: int = 50001
    ip_host_minio_service: str = "127.0.0.1"
    port_minio_service: int = 50001
    refesh_token_minio_service: str = ""
    grpc_ca_cert_path: str = ""
    data_raw_recognizer: str = ""
    data_raw_detection: str = ""
    data_train_recognizer_version: str = ""
    data_text_recognizer_version: str = ""
    data_train_detection_version: str = ""
    data_text_detection_version: str = ""
    artifact_recognizer_file_path: str = ""
    artifact_detection_file_path: str = ""
    image_inferences_file_path: str = ""
    image_raw_inferences_file_path: str = ""
    image_label_inferences_file_path: str = ""
    health_service_name: str = "objectstorage.ObjectStorageService"

    @property
    def grpc_host(self) -> str:
        return self.object_storage_grpc_host or self.ip_host_minio_service

    @property
    def grpc_port(self) -> int:
        return self.object_storage_grpc_port or self.port_minio_service

    @property
    def refresh_token(self) -> str:
        return self.refesh_token_minio_service

    def to_dataset_upload_spec(self) -> "DatasetUploadSpec":
        from src.domain.value_objects.dataset_upload_spec import DatasetUploadSpec
        from src.proto import model_lifecycle_orchestrator_pb2

        return DatasetUploadSpec(
            host=self.grpc_host,
            port=str(self.grpc_port),
            refresh_token=self.refresh_token,
            ca_cert_path=self.grpc_ca_cert_path,
            health_service_name=self.health_service_name,
            source_paths={
                model_lifecycle_orchestrator_pb2.DATA_TYPE_RAW_RECOGNIZER: self.data_raw_recognizer,
                model_lifecycle_orchestrator_pb2.DATA_TYPE_RAW_DETECTION: self.data_raw_detection,
                model_lifecycle_orchestrator_pb2.DATA_TYPE_TRAIN_RECOGNIZER_VERSION: self.data_train_recognizer_version,
                model_lifecycle_orchestrator_pb2.DATA_TYPE_TEXT_RECOGNIZER_VERSION: self.data_text_recognizer_version,
                model_lifecycle_orchestrator_pb2.DATA_TYPE_TRAIN_DETECTION_VERSION: self.data_train_detection_version,
                model_lifecycle_orchestrator_pb2.DATA_TYPE_TEXT_DETECTION_VERSION: self.data_text_detection_version,
                model_lifecycle_orchestrator_pb2.DATA_TYPE_ARTIFACT_RECOGNIZER: self.artifact_recognizer_file_path,
                model_lifecycle_orchestrator_pb2.DATA_TYPE_ARTIFACT_DETECTION: self.artifact_detection_file_path,
                model_lifecycle_orchestrator_pb2.DATA_TYPE_IMAGE_INFERENCES: self.image_inferences_file_path,
                model_lifecycle_orchestrator_pb2.DATA_TYPE_IMAGE_RAW_INFERENCES: self.image_raw_inferences_file_path,
                model_lifecycle_orchestrator_pb2.DATA_TYPE_LABEL_INFERENCES: self.image_label_inferences_file_path,
            },
        )
