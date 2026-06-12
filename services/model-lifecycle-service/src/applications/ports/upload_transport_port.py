from __future__ import annotations

from typing import Protocol


class UploadTransportPort(Protocol):
    def upload(self, file_path: str, upload_url: str) -> None: ...

    def upload_folder(self, folder_path: str, upload_url: str) -> None: ...
