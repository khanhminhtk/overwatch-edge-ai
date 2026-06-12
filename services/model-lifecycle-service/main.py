from __future__ import annotations

from concurrent import futures

import grpc

from src.adapter.inbound.grpc_upload_handler import ModelLifecycleOrchestratorHandler
from src.adapter.inbound.grpc_reflection_handler import register_server_reflection
from src.adapter.outbound.get_link_upload_url import (
    build_download_orchestrator,
    build_upload_orchestrator,
)
from src.infra.config.grpc import GrpcServerConfig, ObjectStorageRuntimeConfig
from src.proto import database_service_pb2
from src.proto import health_pb2
from src.proto import model_lifecycle_orchestrator_pb2
from src.proto import model_lifecycle_orchestrator_pb2_grpc
from src.utils.configloader import ConfigLoader
from src.utils.logger import Logger, LoggerConfig


def create_server(object_storage_config: ObjectStorageRuntimeConfig) -> grpc.Server:
    upload_orchestrator = build_upload_orchestrator(object_storage_config)
    download_orchestrator = build_download_orchestrator(object_storage_config)
    handler = ModelLifecycleOrchestratorHandler(
        upload_orchestrator=upload_orchestrator,
        download_orchestrator=download_orchestrator,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_lifecycle_orchestrator_pb2_grpc.add_ModelLifecycleOrchestratorServiceServicer_to_server(
        handler,
        server,
    )
    register_server_reflection(
        server,
        service_names=[
            model_lifecycle_orchestrator_pb2.DESCRIPTOR.services_by_name[
                "ModelLifecycleOrchestratorService"
            ].full_name,
        ],
        descriptor_sources=[
            health_pb2.DESCRIPTOR,
            database_service_pb2.DESCRIPTOR,
            model_lifecycle_orchestrator_pb2.DESCRIPTOR,
        ],
    )
    return server


def main() -> None:
    loader = ConfigLoader(
        env_file="config/.env",
        config_yaml_file="config/model_lifecycle_orchestrator_config.yaml",
    )
    grpc_server_config = loader.get_typed_config(GrpcServerConfig)
    object_storage_config = loader.get_typed_config(ObjectStorageRuntimeConfig)

    Logger.configure(LoggerConfig(level="INFO"))
    logger = Logger(__name__)

    server = create_server(object_storage_config)
    listen_address = grpc_server_config.listen_address
    server.add_insecure_port(listen_address)
    server.start()
    logger.info(f"Model lifecycle orchestrator started on {listen_address}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
