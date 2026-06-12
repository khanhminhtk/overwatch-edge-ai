import unittest

from src.applications.dtos.upload_file import UploadFileRequestDTO
from src.proto import model_lifecycle_orchestrator_pb2


class UploadFileRequestDTOTest(unittest.TestCase):
    def test_round_trips_source_url(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
