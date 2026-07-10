from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence


class ExperimentTracker(Protocol):
    def ensure_experiment(self, experiment_name: str) -> str: ...

    def start_run(self, experiment_name: str, run_name: str) -> str: ...

    def resume_run(self, run_id: str) -> str: ...

    def end_run(self, run_id: str, status: str = "FINISHED") -> None: ...

    def set_tags(self, run_id: str, tags: Mapping[str, str]) -> None: ...

    def log_params(self, run_id: str, params: Mapping[str, Any]) -> None: ...

    def log_metrics(self, run_id: str, metrics: Mapping[str, float]) -> None: ...

    def log_metric_series(
        self,
        run_id: str,
        metric_name: str,
        rows: Sequence[Mapping[str, int | float]],
    ) -> None: ...


class ArtifactStore(Protocol):
    def log_artifact(self, run_id: str, local_path: str, artifact_path: str) -> None: ...

    def log_artifacts(self, run_id: str, local_dir: str, artifact_path: str) -> None: ...

    def log_text(self, run_id: str, content: str, artifact_file: str) -> None: ...

    def log_json(self, run_id: str, payload: dict[str, Any], artifact_file: str) -> None: ...

    def log_external_artifact_uri(self, run_id: str, key: str, uri: str) -> None: ...


class ModelRegistry(Protocol):
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

    def register_model(self, model_uri: str, registered_model_name: str, run_id: str) -> str: ...

    def set_model_version_tags(
        self,
        registered_model_name: str,
        model_version: str,
        tags: dict[str, str],
    ) -> None: ...

    def set_model_alias(self, registered_model_name: str, alias: str, model_version: str) -> None: ...

    def get_model_version_by_alias(
        self,
        registered_model_name: str,
        alias: str,
    ) -> Any | None: ...

    def get_model_version(
        self,
        registered_model_name: str,
        version: str,
    ) -> Any: ...

    def transition_model_stage(
        self,
        registered_model_name: str,
        model_version: str,
        stage: str,
    ) -> None: ...

    def export_model_download_url(
        self,
        model_name: str,
        version: str,
        artifact_path: str | None = None,
    ) -> str: ...

    def download_model_artifact(
        self,
        model_name: str,
        version: str,
        artifact_path: str,
        output_path: str,
    ) -> str: ...


class TracePort(Protocol):
    def log_trace(
        self,
        name: str,
        request: Any | None = None,
        response: Any | None = None,
        intermediate_outputs: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        start_time_ms: int | None = None,
        execution_time_ms: int | None = None,
    ) -> str: ...

    def log_trace_metric(self, run_id: str, key: str, value: float) -> None: ...
