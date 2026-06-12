__all__ = [
    "HealthCheckRequestDTO",
    "HealthCheckResponseDTO",
    "MlflowArtifactPathsDTO",
    "MlflowEnvironmentDTO",
    "MlflowEvaluationDTO",
    "MlflowLineageDTO",
    "MlflowLoggedModelDTO",
    "MlflowModelCardDTO",
    "MlflowModelSignatureDTO",
    "MlflowPredictionSampleDTO",
    "MlflowPredictionSetDTO",
    "MlflowProductionRunMetadataDTO",
    "MlflowProductionSummaryDTO",
    "MlflowRegistryDTO",
    "MlflowRunParamsDTO",
    "MlflowRunTagsDTO",
    "MlflowSelectedMetricsDTO",
    "MlflowSummaryMetricsDTO",
    "MlflowTensorFieldSignatureDTO",
    "UploadFileRequestDTO",
    "UploadFileResponseDTO",
]


def __getattr__(name: str):
    if name in {"HealthCheckRequestDTO", "HealthCheckResponseDTO"}:
        from src.applications.dtos.health import HealthCheckRequestDTO, HealthCheckResponseDTO

        return {
            "HealthCheckRequestDTO": HealthCheckRequestDTO,
            "HealthCheckResponseDTO": HealthCheckResponseDTO,
        }[name]

    if name in {"UploadFileRequestDTO", "UploadFileResponseDTO"}:
        from src.applications.dtos.upload_file import UploadFileRequestDTO, UploadFileResponseDTO

        return {
            "UploadFileRequestDTO": UploadFileRequestDTO,
            "UploadFileResponseDTO": UploadFileResponseDTO,
        }[name]

    if name in {
        "MlflowArtifactPathsDTO",
        "MlflowEnvironmentDTO",
        "MlflowEvaluationDTO",
        "MlflowLineageDTO",
        "MlflowLoggedModelDTO",
        "MlflowModelCardDTO",
        "MlflowModelSignatureDTO",
        "MlflowPredictionSampleDTO",
        "MlflowPredictionSetDTO",
        "MlflowProductionRunMetadataDTO",
        "MlflowProductionSummaryDTO",
        "MlflowRegistryDTO",
        "MlflowRunParamsDTO",
        "MlflowRunTagsDTO",
        "MlflowSelectedMetricsDTO",
        "MlflowSummaryMetricsDTO",
        "MlflowTensorFieldSignatureDTO",
    }:
        from src.applications.dtos.mlflow_production_run_metadata import (
            MlflowArtifactPathsDTO,
            MlflowEnvironmentDTO,
            MlflowEvaluationDTO,
            MlflowLineageDTO,
            MlflowLoggedModelDTO,
            MlflowModelCardDTO,
            MlflowModelSignatureDTO,
            MlflowPredictionSampleDTO,
            MlflowPredictionSetDTO,
            MlflowProductionRunMetadataDTO,
            MlflowProductionSummaryDTO,
            MlflowRegistryDTO,
            MlflowRunParamsDTO,
            MlflowRunTagsDTO,
            MlflowSelectedMetricsDTO,
            MlflowSummaryMetricsDTO,
            MlflowTensorFieldSignatureDTO,
        )

        return {
            "MlflowArtifactPathsDTO": MlflowArtifactPathsDTO,
            "MlflowEnvironmentDTO": MlflowEnvironmentDTO,
            "MlflowEvaluationDTO": MlflowEvaluationDTO,
            "MlflowLineageDTO": MlflowLineageDTO,
            "MlflowLoggedModelDTO": MlflowLoggedModelDTO,
            "MlflowModelCardDTO": MlflowModelCardDTO,
            "MlflowModelSignatureDTO": MlflowModelSignatureDTO,
            "MlflowPredictionSampleDTO": MlflowPredictionSampleDTO,
            "MlflowPredictionSetDTO": MlflowPredictionSetDTO,
            "MlflowProductionRunMetadataDTO": MlflowProductionRunMetadataDTO,
            "MlflowProductionSummaryDTO": MlflowProductionSummaryDTO,
            "MlflowRegistryDTO": MlflowRegistryDTO,
            "MlflowRunParamsDTO": MlflowRunParamsDTO,
            "MlflowRunTagsDTO": MlflowRunTagsDTO,
            "MlflowSelectedMetricsDTO": MlflowSelectedMetricsDTO,
            "MlflowSummaryMetricsDTO": MlflowSummaryMetricsDTO,
            "MlflowTensorFieldSignatureDTO": MlflowTensorFieldSignatureDTO,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
