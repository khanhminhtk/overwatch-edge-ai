from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.adapter.outbound.get_link_upload_url import (
    LocalDownloadTransport,
    MinioServiceConfig,
    UploadFlowError,
)
from src.applications.dtos.download_file import DownloadFileRequestDTO
from src.applications.services.download_handler import DownloadHandler
from src.domain.value_objects.dataset_upload_spec import DatasetUploadSpec
from src.proto import model_lifecycle_orchestrator_pb2


class FakeDownloadGateway:
    def __init__(self, serving: bool, token: str, url: str):
        self.serving = serving
        self.token = token
        self.url = url
        self.calls = []

    def check(self, config: DatasetUploadSpec) -> None:
        if not self.serving:
            raise UploadFlowError(503, "downstream health is not serving")

    def refresh_access_token(self, request_id: str, config: DatasetUploadSpec) -> str:
        return self.token

    def generate_download_url(
        self,
        request_id: str,
        bucket_name: str,
        object_name: str,
        access_token: str,
        version: str,
        spec: DatasetUploadSpec,
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


class FakeDownloadTransport:
    def __init__(self):
        self.calls = []

    def download(self, download_url: str, destination_path: str, object_name: str) -> None:
        self.calls.append(
            {
                "download_url": download_url,
                "destination_path": destination_path,
                "object_name": object_name,
            }
        )


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


class DownloadFlowTest(unittest.TestCase):
    def test_request_dto_round_trips_destination_path(self) -> None:
        request = model_lifecycle_orchestrator_pb2.DownloadFileRequest(
            request_id="req-download-1",
            request_ip="127.0.0.1",
            bucket_name="artifacts",
            object_name="detection.pt",
            version="v1",
            data_type=model_lifecycle_orchestrator_pb2.DATA_TYPE_ARTIFACT_DETECTION,
            destination_path="/tmp/detection.pt",
        )

        dto = DownloadFileRequestDTO.from_proto(request)

        self.assertEqual(dto.destination_path, "/tmp/detection.pt")
        self.assertEqual(dto.to_proto().destination_path, "/tmp/detection.pt")

    def test_execute_downloads_to_destination_path(self) -> None:
        gateway = FakeDownloadGateway(serving=True, token="access-token", url="https://download.example")
        transport = FakeDownloadTransport()
        spec = DatasetUploadSpec(
            host="127.0.0.1",
            port="50001",
            refresh_token="refresh-token",
            ca_cert_path="",
            source_paths={
                model_lifecycle_orchestrator_pb2.DATA_TYPE_ARTIFACT_DETECTION: "/tmp/not-used.pt"
            },
        )
        handler = DownloadHandler(gateway=gateway, downloader=transport, logger=FakeLogger(), spec=spec)

        result = handler.execute(
            DownloadFileRequestDTO(
                request_id="req-download-1",
                request_ip="127.0.0.1",
                bucket_name="artifacts",
                object_name="detection.pt",
                version="v1",
                data_type=model_lifecycle_orchestrator_pb2.DATA_TYPE_ARTIFACT_DETECTION,
                destination_path="/tmp/detection.pt",
            )
        )

        self.assertEqual(result.code, 200)
        self.assertEqual(gateway.calls[0]["object_name"], "v1/detection.pt")
        self.assertEqual(transport.calls[0]["destination_path"], "/tmp/detection.pt")
        self.assertEqual(transport.calls[0]["object_name"], "v1/detection.pt")

    def test_execute_resolves_file_stem_to_uploaded_object_name(self) -> None:
        gateway = FakeDownloadGateway(serving=True, token="access-token", url="https://download.example")
        transport = FakeDownloadTransport()
        spec = DatasetUploadSpec(
            host="127.0.0.1",
            port="50001",
            refresh_token="refresh-token",
            ca_cert_path="",
            source_paths={
                model_lifecycle_orchestrator_pb2.DATA_TYPE_ARTIFACT_DETECTION: "/tmp/detection.pt"
            },
        )
        handler = DownloadHandler(gateway=gateway, downloader=transport, logger=FakeLogger(), spec=spec)

        result = handler.execute(
            DownloadFileRequestDTO(
                request_id="req-download-2",
                request_ip="127.0.0.1",
                bucket_name="artifacts",
                object_name="detection",
                version="v4",
                data_type=model_lifecycle_orchestrator_pb2.DATA_TYPE_ARTIFACT_DETECTION,
                destination_path="/tmp/downloads",
            )
        )

        self.assertEqual(result.code, 200)
        self.assertEqual(gateway.calls[0]["object_name"], "v4/detection.pt")
        self.assertEqual(transport.calls[0]["object_name"], "v4/detection.pt")

    def test_local_download_transport_saves_resolved_object_name_under_destination_directory(self) -> None:
        transport = LocalDownloadTransport()

        class FakeResponse:
            status_code = 200

            def iter_content(self, chunk_size: int):
                yield b"payload"

        with tempfile.TemporaryDirectory() as tmp_dir:
            destination_dir = Path(tmp_dir) / "test"
            destination_dir.mkdir(parents=True, exist_ok=True)

            with patch("src.adapter.outbound.get_link_upload_url.requests.get", return_value=FakeResponse()):
                transport.download(
                    "https://storage.example/download?signature=abc",
                    str(destination_dir),
                    "v1/detection.zip",
                )

            self.assertTrue((destination_dir / "detection.zip").exists())
            self.assertEqual((destination_dir / "detection.zip").read_bytes(), b"payload")


if __name__ == "__main__":
    unittest.main()
