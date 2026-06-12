import unittest

from src.applications.dtos.upload_file import (
    UploadFileRequestDTO,
    UploadFileResponseDTO,
)
from src.applications.dtos.download_file import DownloadFileResponseDTO
from src.adapter.outbound.get_link_upload_url import (
    build_upload_object_name,
    FilePathResolver,
    MinioServiceConfig,
    normalize_object_storage_target,
    UploadFileOrchestrator,
    UploadFlowError,
)
from src.adapter.inbound.grpc_upload_handler import ModelLifecycleOrchestratorHandler
from src.proto import model_lifecycle_orchestrator_pb2


class FakeFileUploader:
    def __init__(self, success: bool):
        self.success = success

    def upload(self, file_path: str, upload_url: str) -> None:
        if not self.success:
            raise RuntimeError("upload failed")

    def upload_folder(self, folder_path: str, upload_url: str) -> None:
        if not self.success:
            raise RuntimeError("upload folder failed")


class FakeUploadGateway:
    def __init__(self, serving: bool, token: str, url: str):
        self.serving = serving
        self.token = token
        self.url = url
        self.calls = []

    def check(self, config: MinioServiceConfig) -> None:
        if not self.serving:
            raise UploadFlowError(503, "downstream health is not serving")

    def refresh_access_token(self, request_id: str, config: MinioServiceConfig) -> str:
        return self.token

    def generate_upload_url(
        self,
        request_id: str,
        bucket_name: str,
        object_name: str,
        access_token: str,
        version: str,
        spec: MinioServiceConfig,
    ) -> str:
        self.calls.append(
            {
                "request_id": request_id,
                "bucket_name": bucket_name,
                "object_name": object_name,
                "access_token": access_token,
                "version": version,
            }
        )
        return self.url


class FakeOrchestrator:
    def __init__(self, result: UploadFileResponseDTO):
        self.result = result

    def execute(self, request: UploadFileRequestDTO) -> UploadFileResponseDTO:
        return self.result


class FakeDownloadOrchestrator:
    def __init__(self, result: DownloadFileResponseDTO):
        self.result = result

    def execute(self, request):
        return self.result


class FilePathResolverTest(unittest.TestCase):
    def test_resolve_raw_path_returns_expected_env_value(self) -> None:
        resolver = FilePathResolver(
            {
                "raw_file_path": "/tmp/raw.bin",
                "train_file_path": "/tmp/train.bin",
                "artifact_file_path": "/tmp/artifact.bin",
                "validation_file_path": "/tmp/validation.bin",
                "model_file_path": "/tmp/model.bin",
            }
        )

        resolved = resolver.resolve(model_lifecycle_orchestrator_pb2.DATA_TYPE_RAW_RECOGNIZER)

        self.assertEqual(resolved, "/tmp/raw.bin")

    def test_normalize_object_storage_target_moves_bucket_path_into_object_prefix(self) -> None:
        bucket_name, object_name = normalize_object_storage_target(
            "success/lable",
            "lable_1.txt",
        )

        self.assertEqual(bucket_name, "success")
        self.assertEqual(object_name, "lable/lable_1.txt")

    def test_build_upload_object_name_appends_source_file_name_for_directory_target(self) -> None:
        object_name = build_upload_object_name(
            "success/lable",
            "v1",
            "/tmp/lable_1.txt",
        )

        self.assertEqual(object_name, "v1/success/lable/lable_1.txt")

    def test_build_upload_object_name_keeps_explicit_file_name(self) -> None:
        object_name = build_upload_object_name(
            "success/lable/lable.txt",
            "v1",
            "/tmp/lable_1.txt",
        )

        self.assertEqual(object_name, "v1/success/lable/lable.txt")

    def test_build_upload_object_name_appends_source_suffix_for_file_stem(self) -> None:
        object_name = build_upload_object_name(
            "detection",
            "v1",
            "",
            "/tmp/detection.pt",
        )

        self.assertEqual(object_name, "v1/detection.pt")


class UploadFileOrchestratorTest(unittest.TestCase):
    def test_request_dto_round_trips_source_url(self) -> None:
        request = model_lifecycle_orchestrator_pb2.UploadFileRequest(
            request_id="req-source",
            request_ip="127.0.0.1",
            bucket_name="bucket-a",
            object_name="raw.bin",
            version="v1",
            data_type=model_lifecycle_orchestrator_pb2.DATA_TYPE_RAW_RECOGNIZER,
            source_url="https://device.example/raw.bin",
        )

        dto = UploadFileRequestDTO.from_proto(request)

        self.assertEqual(dto.source_url, "https://device.example/raw.bin")
        self.assertEqual(dto.to_proto().source_url, "https://device.example/raw.bin")

    def test_execute_returns_success_when_all_dependencies_succeed(self) -> None:
        gateway = FakeUploadGateway(serving=True, token="access-token", url="https://upload.example")
        orchestrator = UploadFileOrchestrator(
            gateway=gateway,
            uploader=FakeFileUploader(success=True),
            spec=MinioServiceConfig(
                host="127.0.0.1",
                port="50001",
                refresh_token="refresh-token",
                source_paths={model_lifecycle_orchestrator_pb2.DATA_TYPE_RAW_RECOGNIZER: "/tmp/raw.bin"},
            ),
        )

        result = orchestrator.execute(
            UploadFileRequestDTO(
                request_id="req-1",
                request_ip="127.0.0.1",
                bucket_name="bucket-a",
                object_name="raw.bin",
                version="v0",
                data_type=model_lifecycle_orchestrator_pb2.DATA_TYPE_RAW_RECOGNIZER,
                source_url="",
            )
        )

        self.assertEqual(result.code, 200)
        self.assertEqual(result.message, "Upload completed successfully")
        self.assertEqual(result.status, model_lifecycle_orchestrator_pb2.STATUS_OK)

    def test_execute_appends_source_file_name_when_object_name_is_directory(self) -> None:
        gateway = FakeUploadGateway(serving=True, token="access-token", url="https://upload.example")
        orchestrator = UploadFileOrchestrator(
            gateway=gateway,
            uploader=FakeFileUploader(success=True),
            spec=MinioServiceConfig(
                host="127.0.0.1",
                port="50001",
                refresh_token="refresh-token",
                source_paths={model_lifecycle_orchestrator_pb2.DATA_TYPE_RAW_RECOGNIZER: "/tmp/raw.bin"},
            ),
        )

        result = orchestrator.execute(
            UploadFileRequestDTO(
                request_id="req-1",
                request_ip="127.0.0.1",
                bucket_name="inferences",
                object_name="success/lable",
                version="v1",
                data_type=model_lifecycle_orchestrator_pb2.DATA_TYPE_RAW_RECOGNIZER,
                source_url="/tmp/lable_1.txt",
            )
        )

        self.assertEqual(result.code, 200)
        self.assertEqual(gateway.calls[0]["bucket_name"], "inferences")
        self.assertEqual(gateway.calls[0]["object_name"], "v1/success/lable/lable_1.txt")
        self.assertEqual(gateway.calls[0]["version"], "")

    def test_execute_appends_source_file_suffix_when_object_name_is_stem(self) -> None:
        gateway = FakeUploadGateway(serving=True, token="access-token", url="https://upload.example")
        orchestrator = UploadFileOrchestrator(
            gateway=gateway,
            uploader=FakeFileUploader(success=True),
            spec=MinioServiceConfig(
                host="127.0.0.1",
                port="50001",
                refresh_token="refresh-token",
                source_paths={
                    model_lifecycle_orchestrator_pb2.DATA_TYPE_ARTIFACT_DETECTION: "/tmp/detection.pt"
                },
            ),
        )

        result = orchestrator.execute(
            UploadFileRequestDTO(
                request_id="req-1",
                request_ip="127.0.0.1",
                bucket_name="artifacts",
                object_name="detection",
                version="v1",
                data_type=model_lifecycle_orchestrator_pb2.DATA_TYPE_ARTIFACT_DETECTION,
                source_url="",
            )
        )

        self.assertEqual(result.code, 200)
        self.assertEqual(gateway.calls[0]["object_name"], "v1/detection.pt")

    def test_execute_returns_health_failure_when_service_is_not_serving(self) -> None:
        orchestrator = UploadFileOrchestrator(
            gateway=FakeUploadGateway(serving=False, token="access-token", url="https://upload.example"),
            uploader=FakeFileUploader(success=True),
            spec=MinioServiceConfig(
                host="127.0.0.1",
                port="50001",
                refresh_token="refresh-token",
                source_paths={model_lifecycle_orchestrator_pb2.DATA_TYPE_RAW_RECOGNIZER: "/tmp/raw.bin"},
            ),
        )

        result = orchestrator.execute(
            UploadFileRequestDTO(
                request_id="req-2",
                request_ip="127.0.0.1",
                bucket_name="bucket-a",
                object_name="raw.bin",
                version="v0",
                data_type=model_lifecycle_orchestrator_pb2.DATA_TYPE_RAW_RECOGNIZER,
                source_url="",
            )
        )

        self.assertEqual(result.code, 503)
        self.assertEqual(result.status, model_lifecycle_orchestrator_pb2.STATUS_ERROR)

    def test_execute_returns_invalid_data_type_failure_for_unspecified_type(self) -> None:
        orchestrator = UploadFileOrchestrator(
            gateway=FakeUploadGateway(serving=True, token="access-token", url="https://upload.example"),
            uploader=FakeFileUploader(success=True),
            spec=MinioServiceConfig(
                host="127.0.0.1",
                port="50001",
                refresh_token="refresh-token",
                source_paths={
                    model_lifecycle_orchestrator_pb2.DATA_TYPE_RAW_RECOGNIZER: "/tmp/raw.bin",
                    model_lifecycle_orchestrator_pb2.DATA_TYPE_RAW_DETECTION: "/tmp/train.bin",
                    model_lifecycle_orchestrator_pb2.DATA_TYPE_ARTIFACT_RECOGNIZER: "/tmp/artifact.bin",
                    model_lifecycle_orchestrator_pb2.DATA_TYPE_TEXT_RECOGNIZER_VERSION: "/tmp/validation.bin",
                    model_lifecycle_orchestrator_pb2.DATA_TYPE_TEXT_DETECTION_VERSION: "/tmp/model.bin",
                },
            ),
        )

        result = orchestrator.execute(
            UploadFileRequestDTO(
                request_id="req-4",
                request_ip="127.0.0.1",
                bucket_name="bucket-a",
                object_name="raw.bin",
                version="v0",
                data_type=model_lifecycle_orchestrator_pb2.DATA_TYPE_UNSPECIFIED,
                source_url="",
            )
        )

        self.assertEqual(result.code, 400)
        self.assertEqual(result.status, model_lifecycle_orchestrator_pb2.STATUS_ERROR)


class GrpcHandlerTest(unittest.TestCase):
    def test_upload_file_maps_orchestrator_result_to_proto_response(self) -> None:
        handler = ModelLifecycleOrchestratorHandler(
            upload_orchestrator=FakeOrchestrator(
                UploadFileResponseDTO(
                    request_id="req-3",
                    code=200,
                    message="Upload completed successfully",
                    status=model_lifecycle_orchestrator_pb2.STATUS_OK,
                )
            )
        )

        response = handler.UploadFile(
            model_lifecycle_orchestrator_pb2.UploadFileRequest(
                request_id="req-3",
                bucket_name="bucket-a",
                object_name="raw.bin",
                version="v0",
                data_type=model_lifecycle_orchestrator_pb2.DATA_TYPE_RAW_RECOGNIZER,
            ),
            None,
        )

        self.assertEqual(response.request_id, "req-3")
        self.assertEqual(response.code, 200)
        self.assertEqual(response.status, model_lifecycle_orchestrator_pb2.STATUS_OK)

    def test_download_file_maps_orchestrator_result_to_proto_response(self) -> None:
        handler = ModelLifecycleOrchestratorHandler(
            upload_orchestrator=FakeOrchestrator(
                UploadFileResponseDTO(
                    request_id="unused",
                    code=200,
                    message="Upload completed successfully",
                    status=model_lifecycle_orchestrator_pb2.STATUS_OK,
                )
            ),
            download_orchestrator=FakeDownloadOrchestrator(
                DownloadFileResponseDTO(
                    request_id="req-dl-1",
                    code=200,
                    message="Download completed successfully",
                    status=model_lifecycle_orchestrator_pb2.STATUS_OK,
                )
            ),
        )

        response = handler.DownloadFile(
            model_lifecycle_orchestrator_pb2.DownloadFileRequest(
                request_id="req-dl-1",
                request_ip="127.0.0.1",
                bucket_name="artifacts",
                object_name="detection.pt",
                version="v1",
                data_type=model_lifecycle_orchestrator_pb2.DATA_TYPE_ARTIFACT_DETECTION,
                destination_path="/tmp/detection.pt",
            ),
            None,
        )

        self.assertEqual(response.request_id, "req-dl-1")
        self.assertEqual(response.code, 200)
        self.assertEqual(response.status, model_lifecycle_orchestrator_pb2.STATUS_OK)

    def test_health_check_returns_ok_response(self) -> None:
        handler = ModelLifecycleOrchestratorHandler(
            upload_orchestrator=FakeOrchestrator(
                UploadFileResponseDTO(
                    request_id="unused",
                    code=200,
                    message="Upload completed successfully",
                    status=model_lifecycle_orchestrator_pb2.STATUS_OK,
                )
            )
        )

        response = handler.HealthCheck(
            model_lifecycle_orchestrator_pb2.HealthCheckRequest(request_id="req-health"),
            None,
        )

        self.assertEqual(response.request_id, "req-health")
        self.assertEqual(response.code, 200)
        self.assertEqual(response.message, "Service is healthy")
        self.assertEqual(response.status, model_lifecycle_orchestrator_pb2.STATUS_OK)


if __name__ == "__main__":
    unittest.main()
