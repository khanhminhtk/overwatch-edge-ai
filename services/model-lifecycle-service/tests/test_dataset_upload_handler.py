from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.applications.dtos.upload_file import UploadFileRequestDTO
from src.applications.services.upload_handler import UploadHandler
from src.domain.value_objects.dataset_upload_spec import DatasetUploadSpec
from src.proto import model_lifecycle_orchestrator_pb2


class FakeLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.messages.append(message % args if args else message)

    def info(self, message: str, *args: object, **kwargs: object) -> None:
        self.messages.append(message % args if args else message)

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self.messages.append(message % args if args else message)

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self.messages.append(message % args if args else message)

    def exception(self, message: str, *args: object, **kwargs: object) -> None:
        self.messages.append(message % args if args else message)


class FakeGateway:
    def __init__(self, url: str = "https://upload.example") -> None:
        self.url = url
        self.calls: list[tuple[str, object]] = []

    def check(self, spec: DatasetUploadSpec) -> None:
        self.calls.append(("check", spec.address))

    def refresh_access_token(self, request_id: str, spec: DatasetUploadSpec) -> str:
        self.calls.append(("refresh", request_id))
        return "access-token"

    def generate_upload_url(
        self,
        request_id: str,
        bucket_name: str,
        object_name: str,
        access_token: str,
        version: str,
        spec: DatasetUploadSpec,
    ) -> str:
        self.calls.append(
            (
                "generate",
                {
                    "request_id": request_id,
                    "bucket_name": bucket_name,
                    "object_name": object_name,
                    "access_token": access_token,
                    "version": version,
                    "address": spec.address,
                },
            )
        )
        return self.url


class FakeUploader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def upload(self, file_path: str, upload_url: str) -> None:
        self.calls.append(("file", file_path, upload_url))

    def upload_folder(self, folder_path: str, upload_url: str) -> None:
        self.calls.append(("folder", folder_path, upload_url))


class DatasetUploadSpecTest(unittest.TestCase):
    def test_resolve_source_path_for_data_type(self) -> None:
        spec = DatasetUploadSpec.from_mapping(
            {
                "object_storage_grpc_host": "127.0.0.1",
                "object_storage_grpc_port": "50001",
                "refesh_token_minio_service": "refresh-token",
                "grpc_ca_cert_path": "",
                "data_raw_recognizer": "/tmp/raw-recognizer",
                "data_raw_detection": "/tmp/raw-detection",
                "data_train_recognizer_version": "/tmp/train-recognizer",
                "data_text_recognizer_version": "/tmp/text-recognizer",
                "data_train_detection_version": "/tmp/train-detection",
                "data_text_detection_version": "/tmp/text-detection",
                "artifact_recognizer_file_path": "/tmp/artifact-recognizer",
                "artifact_detection_file_path": "/tmp/artifact-detection",
                "image_inferences_file_path": "/tmp/image-inferences",
                "image_raw_inferences_file_path": "/tmp/image-raw-inferences",
                "image_label_inferences_file_path": "/tmp/image-label-inferences",
            }
        )

        self.assertEqual(
            spec.resolve_source_path(model_lifecycle_orchestrator_pb2.DATA_TYPE_RAW_RECOGNIZER),
            "/tmp/raw-recognizer",
        )
        self.assertEqual(spec.address, "127.0.0.1:50001")


class UploadHandlerTest(unittest.TestCase):
    def test_execute_appends_zip_extension_for_directory_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_dir = Path(tmp_dir)

            spec = DatasetUploadSpec.from_mapping(
                {
                    "object_storage_grpc_host": "127.0.0.1",
                    "object_storage_grpc_port": "50001",
                    "refesh_token_minio_service": "refresh-token",
                    "grpc_ca_cert_path": "",
                    "data_raw_recognizer": str(source_dir),
                    "data_raw_detection": "/tmp/raw-detection",
                    "data_train_recognizer_version": "/tmp/train-recognizer",
                    "data_text_recognizer_version": "/tmp/text-recognizer",
                    "data_train_detection_version": "/tmp/train-detection",
                    "data_text_detection_version": "/tmp/text-detection",
                    "artifact_recognizer_file_path": "/tmp/artifact-recognizer",
                    "artifact_detection_file_path": "/tmp/artifact-detection",
                    "image_inferences_file_path": "/tmp/image-inferences",
                    "image_raw_inferences_file_path": "/tmp/image-raw-inferences",
                    "image_label_inferences_file_path": "/tmp/image-label-inferences",
                }
            )

            logger = FakeLogger()
            gateway = FakeGateway()
            uploader = FakeUploader()
            handler = UploadHandler(
                gateway=gateway,
                uploader=uploader,
                logger=logger,
                spec=spec,
            )

            result = handler.execute(
                UploadFileRequestDTO(
                    request_id="req-zip",
                    request_ip="127.0.0.1",
                    bucket_name="artifacts",
                    object_name="detection",
                    version="v3",
                    data_type=model_lifecycle_orchestrator_pb2.DATA_TYPE_RAW_RECOGNIZER,
                    source_url="",
                )
            )

            self.assertEqual(result.code, 200)
            self.assertEqual(gateway.calls[2][1]["object_name"], "v3/detection.zip")

    def test_execute_uploads_file_and_logs_flow(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"payload")
            source_path = tmp_file.name

        self.addCleanup(lambda: Path(source_path).unlink(missing_ok=True))

        spec = DatasetUploadSpec.from_mapping(
            {
                "object_storage_grpc_host": "127.0.0.1",
                "object_storage_grpc_port": "50001",
                "refesh_token_minio_service": "refresh-token",
                "grpc_ca_cert_path": "",
                "data_raw_recognizer": source_path,
                "data_raw_detection": "/tmp/raw-detection",
                "data_train_recognizer_version": "/tmp/train-recognizer",
                "data_text_recognizer_version": "/tmp/text-recognizer",
                "data_train_detection_version": "/tmp/train-detection",
                "data_text_detection_version": "/tmp/text-detection",
                "artifact_recognizer_file_path": "/tmp/artifact-recognizer",
                "artifact_detection_file_path": "/tmp/artifact-detection",
                "image_inferences_file_path": "/tmp/image-inferences",
                "image_raw_inferences_file_path": "/tmp/image-raw-inferences",
                "image_label_inferences_file_path": "/tmp/image-label-inferences",
            }
        )

        logger = FakeLogger()
        gateway = FakeGateway()
        uploader = FakeUploader()
        handler = UploadHandler(
            gateway=gateway,
            uploader=uploader,
            logger=logger,
            spec=spec,
        )

        result = handler.execute(
            UploadFileRequestDTO(
                request_id="req-1",
                request_ip="127.0.0.1",
                bucket_name="success/lable",
                object_name="lable_1.txt",
                version="v1",
                data_type=model_lifecycle_orchestrator_pb2.DATA_TYPE_RAW_RECOGNIZER,
                source_url="/tmp/lable_1.txt",
            )
        )

        self.assertEqual(result.code, 200)
        self.assertEqual(uploader.calls[0][0], "file")
        self.assertEqual(uploader.calls[0][1], source_path)
        self.assertEqual(gateway.calls[0], ("check", "127.0.0.1:50001"))
        self.assertEqual(gateway.calls[2][0], "generate")
        self.assertTrue(any("upload.start" in message for message in logger.messages))
        self.assertTrue(any("upload.completed" in message for message in logger.messages))


if __name__ == "__main__":
    unittest.main()
