from __future__ import annotations

from typing import Any, Protocol


class MlflowModelRegistryPort(Protocol):
    def log_pyfunc_model(
        self,
        run_id: str,
        model_name: str,
        python_model: Any,
        input_example: Any,
        signature: Any,
        registered_model_name: str | None = None,
        tags: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> str: ...

    def register_model(self, model_uri: str, registered_model_name: str) -> str: ...

    def set_model_version_tags(
        self,
        registered_model_name: str,
        model_version: str,
        tags: dict[str, str],
    ) -> None: ...

    def set_model_alias(self, registered_model_name: str, alias: str, model_version: str) -> None: ...

    def transition_model_stage(
        self,
        registered_model_name: str,
        model_version: str,
        stage: str,
    ) -> None: ...

    def export_model_download_url(self, model_name: str, version: str) -> str: ...
