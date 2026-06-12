from __future__ import annotations

from typing import Protocol

from src.domain.value_objects.dataset_upload_spec import DatasetUploadSpec


class UploadGatewayPort(Protocol):
    def check(self, spec: DatasetUploadSpec) -> None: ...

    def refresh_access_token(self, request_id: str, spec: DatasetUploadSpec) -> str: ...

    def generate_upload_url(
        self,
        request_id: str,
        bucket_name: str,
        object_name: str,
        access_token: str,
        version: str,
        spec: DatasetUploadSpec,
    ) -> str: ...

    def generate_download_url(
        self,
        request_id: str,
        bucket_name: str,
        object_name: str,
        access_token: str,
        version: str,
        spec: DatasetUploadSpec,
    ) -> str: ...
