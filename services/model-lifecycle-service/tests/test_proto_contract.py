import unittest

from src.proto import model_lifecycle_orchestrator_pb2
from src.proto import model_lifecycle_orchestrator_pb2_grpc


class ProtoContractTest(unittest.TestCase):
    def test_generated_proto_modules_expose_service_contract(self) -> None:
        self.assertTrue(hasattr(model_lifecycle_orchestrator_pb2, "UploadFileRequest"))
        self.assertTrue(hasattr(model_lifecycle_orchestrator_pb2, "DownloadFileRequest"))
        self.assertTrue(hasattr(model_lifecycle_orchestrator_pb2, "HealthCheckResponse"))
        self.assertTrue(
            hasattr(
                model_lifecycle_orchestrator_pb2_grpc,
                "ModelLifecycleOrchestratorServiceStub",
            )
        )

    def test_upload_file_request_exposes_source_url_field(self) -> None:
        request = model_lifecycle_orchestrator_pb2.UploadFileRequest()

        self.assertTrue(hasattr(request, "source_url"))

    def test_download_file_request_exposes_destination_path_field(self) -> None:
        request = model_lifecycle_orchestrator_pb2.DownloadFileRequest()

        self.assertTrue(hasattr(request, "destination_path"))


if __name__ == "__main__":
    unittest.main()
