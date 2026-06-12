from __future__ import annotations

from typing import Protocol


class DownloadTransportPort(Protocol):
    def download(self, download_url: str, destination_path: str, object_name: str) -> None: ...
