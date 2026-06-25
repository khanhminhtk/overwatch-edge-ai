"""Application ports."""

__all__ = [
    "ConfigProviderPort",
    "DownloadTransportPort",
    "IMessageBlokerClient",
    "IMessageBlokerConsummer",
    "IMessageBlokerProducer",
    "IModelAI",
    "LoggerPort",
    "MlflowArtifactPort",
    "MlflowModelRegistryPort",
    "MlflowTracePort",
    "MlflowTrackingPort",
    "UploadGatewayPort",
    "UploadTransportPort",
]


def __getattr__(name: str):
    if name == "ConfigProviderPort":
        from src.applications.ports.config_provider_port import ConfigProviderPort

        return ConfigProviderPort

    if name == "DownloadTransportPort":
        from src.applications.ports.download_transport_port import DownloadTransportPort

        return DownloadTransportPort

    if name == "IModelAI":
        from src.applications.ports.modelai import IModelAI

        return IModelAI

    if name == "LoggerPort":
        from src.applications.ports.logger_port import LoggerPort

        return LoggerPort

    if name in {"IMessageBlokerClient", "IMessageBlokerConsummer", "IMessageBlokerProducer"}:
        from src.applications.ports.message_bloker import (
            IMessageBlokerClient,
            IMessageBlokerConsummer,
            IMessageBlokerProducer,
        )

        return {
            "IMessageBlokerClient": IMessageBlokerClient,
            "IMessageBlokerConsummer": IMessageBlokerConsummer,
            "IMessageBlokerProducer": IMessageBlokerProducer,
        }[name]

    if name == "MlflowArtifactPort":
        from src.applications.ports.mlflow_artifact_port import MlflowArtifactPort

        return MlflowArtifactPort

    if name == "MlflowModelRegistryPort":
        from src.applications.ports.mlflow_model_registry_port import MlflowModelRegistryPort

        return MlflowModelRegistryPort

    if name == "MlflowTracePort":
        from src.applications.ports.mlflow_trace_port import MlflowTracePort

        return MlflowTracePort

    if name == "MlflowTrackingPort":
        from src.applications.ports.mlflow_tracking_port import MlflowTrackingPort

        return MlflowTrackingPort

    if name == "UploadGatewayPort":
        from src.applications.ports.upload_gateway_port import UploadGatewayPort

        return UploadGatewayPort

    if name == "UploadTransportPort":
        from src.applications.ports.upload_transport_port import UploadTransportPort

        return UploadTransportPort

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
