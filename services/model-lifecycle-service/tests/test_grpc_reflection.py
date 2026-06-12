from __future__ import annotations

import unittest

from main import create_server
from src.adapter.inbound.grpc_reflection_handler import ServerReflectionService
from src.infra.config.grpc import ObjectStorageRuntimeConfig
from src.proto import database_service_pb2
from src.proto import health_pb2
from src.proto import model_lifecycle_orchestrator_pb2
from src.proto.grpc.reflection.v1alpha import reflection_pb2


class GrpcReflectionTest(unittest.TestCase):
    def test_create_server_registers_reflection_service(self) -> None:
        server = create_server(
            ObjectStorageRuntimeConfig(
                object_storage_grpc_host="127.0.0.1",
                object_storage_grpc_port=50001,
                ip_host_minio_service="127.0.0.1",
                port_minio_service=50001,
                refesh_token_minio_service="refresh-token",
                grpc_ca_cert_path="",
                data_raw_recognizer="/tmp/raw-recognizer",
                data_raw_detection="/tmp/raw-detection",
                data_train_recognizer_version="/tmp/train-recognizer",
                data_text_recognizer_version="/tmp/text-recognizer",
                data_train_detection_version="/tmp/train-detection",
                data_text_detection_version="/tmp/text-detection",
                artifact_recognizer_file_path="/tmp/artifact-recognizer",
                artifact_detection_file_path="/tmp/artifact-detection",
                image_inferences_file_path="/tmp/image-inferences",
                image_raw_inferences_file_path="/tmp/image-raw-inferences",
                image_label_inferences_file_path="/tmp/image-label-inferences",
            )
        )
        service_names = [
            getattr(handler, "service_name", lambda: None)()
            for handler in server._state.generic_handlers
        ]
        self.assertIn("grpc.reflection.v1alpha.ServerReflection", service_names)

    def test_reflection_lists_model_lifecycle_service(self) -> None:
        service_name = model_lifecycle_orchestrator_pb2.DESCRIPTOR.services_by_name[
            "ModelLifecycleOrchestratorService"
        ].full_name
        service = ServerReflectionService(
            service_names=[service_name],
            descriptor_sources=[
                health_pb2.DESCRIPTOR,
                database_service_pb2.DESCRIPTOR,
                model_lifecycle_orchestrator_pb2.DESCRIPTOR,
            ],
        )

        class FakeContext:
            def __init__(self) -> None:
                self.code = None
                self.details = None

            def set_code(self, code) -> None:
                self.code = code

            def set_details(self, details) -> None:
                self.details = details

        request = reflection_pb2.ServerReflectionRequest(list_services="")
        response = next(service.ServerReflectionInfo(iter([request]), FakeContext()))
        service_names_response = [
            item.name for item in response.list_services_response.service
        ]
        self.assertIn(service_name, service_names_response)


if __name__ == "__main__":
    unittest.main()
