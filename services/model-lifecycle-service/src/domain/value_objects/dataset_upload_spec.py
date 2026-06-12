from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.domain.errors.upload_flow_error import UploadFlowError
from src.proto import model_lifecycle_orchestrator_pb2


@dataclass(frozen=True)
class DatasetUploadSpec:
    host: str
    port: str
    refresh_token: str
    ca_cert_path: str = ""
    health_service_name: str = "objectstorage.ObjectStorageService"
    source_paths: Mapping[int, str] | None = None

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @classmethod
    def from_mapping(cls, config: Mapping[str, str]) -> "DatasetUploadSpec":
        def first_value(*keys: str) -> str:
            for key in keys:
                value = str(config.get(key, "")).strip()
                if value:
                    return value
            return ""

        return cls(
            host=first_value("object_storage_grpc_host", "ip_host_minio_service", "grpc_model_lifecycle_orchestrator_host") or "127.0.0.1",
            port=first_value("object_storage_grpc_port", "port_minio_service", "grpc_model_lifecycle_orchestrator_port") or "50001",
            refresh_token=str(config.get("refesh_token_minio_service", "")).strip(),
            ca_cert_path=str(config.get("grpc_ca_cert_path", "")).strip(),
            source_paths={
                model_lifecycle_orchestrator_pb2.DATA_TYPE_RAW_RECOGNIZER: first_value("data_raw_recognizer", "raw_file_path"),
                model_lifecycle_orchestrator_pb2.DATA_TYPE_RAW_DETECTION: first_value("data_raw_detection", "train_file_path"),
                model_lifecycle_orchestrator_pb2.DATA_TYPE_TRAIN_RECOGNIZER_VERSION: first_value("data_train_recognizer_version", "validation_file_path"),
                model_lifecycle_orchestrator_pb2.DATA_TYPE_TEXT_RECOGNIZER_VERSION: first_value("data_text_recognizer_version", "model_file_path"),
                model_lifecycle_orchestrator_pb2.DATA_TYPE_TRAIN_DETECTION_VERSION: first_value("data_train_detection_version", "train_file_path"),
                model_lifecycle_orchestrator_pb2.DATA_TYPE_TEXT_DETECTION_VERSION: first_value("data_text_detection_version", "validation_file_path"),
                model_lifecycle_orchestrator_pb2.DATA_TYPE_ARTIFACT_RECOGNIZER: first_value("artifact_recognizer_file_path", "artifact_file_path"),
                model_lifecycle_orchestrator_pb2.DATA_TYPE_ARTIFACT_DETECTION: first_value("artifact_detection_file_path", "artifact_file_path"),
                model_lifecycle_orchestrator_pb2.DATA_TYPE_IMAGE_INFERENCES: first_value("image_inferences_file_path"),
                model_lifecycle_orchestrator_pb2.DATA_TYPE_IMAGE_RAW_INFERENCES: first_value("image_raw_inferences_file_path"),
                model_lifecycle_orchestrator_pb2.DATA_TYPE_LABEL_INFERENCES: first_value("image_label_inferences_file_path"),
            },
        )

    def resolve_source_path(self, data_type: int) -> str:
        if data_type == model_lifecycle_orchestrator_pb2.DATA_TYPE_UNSPECIFIED:
            raise UploadFlowError(400, "Unsupported data_type")

        if self.source_paths is None:
            raise UploadFlowError(500, "Upload source paths are not configured")

        file_path = str(self.source_paths.get(data_type, "")).strip()
        if not file_path:
            raise UploadFlowError(404, f"Missing configured file path for data_type={data_type}")
        return file_path
