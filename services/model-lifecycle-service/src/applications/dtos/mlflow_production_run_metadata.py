from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.domain.value_objects.mlflow_production_run_metadata import (
    MlflowArtifactPaths,
    MlflowEnvironment,
    MlflowEvaluation,
    MlflowLineage,
    MlflowLoggedModel,
    MlflowModelCard,
    MlflowModelSignature,
    MlflowPredictionSample,
    MlflowPredictionSet,
    MlflowProductionRunMetadata,
    MlflowProductionSummary,
    MlflowRegistry,
    MlflowRunParams,
    MlflowRunTags,
    MlflowSelectedMetrics,
    MlflowSummaryMetrics,
    MlflowTensorFieldSignature,
)


@dataclass(frozen=True)
class MlflowEnvironmentDTO:
    is_mocked: bool
    host_name: str
    gpu_name: str
    gpu_memory_gb: float
    gpu_memory_used_gb: float
    gpu_utilization_avg: float
    cuda_version: str
    driver_version: str
    docker_image: str
    python_version: str
    pytorch_version: str
    platform: str

    def to_domain(self) -> MlflowEnvironment:
        return MlflowEnvironment(**asdict(self))


@dataclass(frozen=True)
class MlflowLineageDTO:
    is_mocked: bool
    git_commit: str
    git_branch: str
    pipeline_name: str
    pipeline_run_id: str
    orchestrator_job_id: str
    parent_run_id: str

    def to_domain(self) -> MlflowLineage:
        return MlflowLineage(**asdict(self))


@dataclass(frozen=True)
class MlflowRegistryDTO:
    is_mocked: bool
    registered_model_name: str
    model_version: str
    deployment_stage: str
    approval_status: str
    owner_team: str

    def to_domain(self) -> MlflowRegistry:
        return MlflowRegistry(**asdict(self))


@dataclass(frozen=True)
class MlflowTensorFieldSignatureDTO:
    name: str
    dtype: str
    shape: list[int]
    description: str

    def to_domain(self) -> MlflowTensorFieldSignature:
        return MlflowTensorFieldSignature(**asdict(self))


@dataclass(frozen=True)
class MlflowModelSignatureDTO:
    is_mocked: bool
    input_signature: MlflowTensorFieldSignatureDTO
    output_signature: MlflowTensorFieldSignatureDTO

    def to_domain(self) -> MlflowModelSignature:
        return MlflowModelSignature(
            is_mocked=self.is_mocked,
            input_signature=self.input_signature.to_domain(),
            output_signature=self.output_signature.to_domain(),
        )


@dataclass(frozen=True)
class MlflowEvaluationDTO:
    is_mocked: bool
    inference_latency_ms_p50: float
    inference_latency_ms_p95: float
    throughput_samples_per_sec: float
    train_duration_minutes: float
    cpu_utilization_avg: float
    ram_used_gb: float
    business_gate: str

    def to_domain(self) -> MlflowEvaluation:
        return MlflowEvaluation(**asdict(self))


@dataclass(frozen=True)
class MlflowPredictionSampleDTO:
    image_id: str
    ground_truth: str
    prediction: str
    confidence: float

    def to_domain(self) -> MlflowPredictionSample:
        return MlflowPredictionSample(**asdict(self))


@dataclass(frozen=True)
class MlflowPredictionSetDTO:
    is_mocked: bool
    samples: list[MlflowPredictionSampleDTO] = field(default_factory=list)

    def to_domain(self) -> MlflowPredictionSet:
        return MlflowPredictionSet(
            is_mocked=self.is_mocked,
            samples=[sample.to_domain() for sample in self.samples],
        )


@dataclass(frozen=True)
class MlflowModelCardDTO:
    model_name: str
    purpose: str
    architecture: str
    training_status: str
    data_source: str
    known_limitations: list[str] = field(default_factory=list)

    def to_domain(self) -> MlflowModelCard:
        return MlflowModelCard(**asdict(self))


@dataclass(frozen=True)
class MlflowProductionSummaryDTO:
    run_goal: str
    uses_real_tensorboard_metrics: bool
    uses_real_checkpoints: bool
    uses_real_dataset_zip: bool
    uses_real_manifest_even_if_wrong: bool
    contains_mock_metadata_sections: bool

    def to_domain(self) -> MlflowProductionSummary:
        return MlflowProductionSummary(**asdict(self))


@dataclass(frozen=True)
class MlflowSelectedMetricsDTO:
    train_final_metrics: dict[str, float | None] = field(default_factory=dict)
    train_best_metrics: dict[str, float | None] = field(default_factory=dict)
    val_final_metrics: dict[str, float | None] = field(default_factory=dict)
    val_best_metrics: dict[str, float | None] = field(default_factory=dict)

    def to_domain(self) -> MlflowSelectedMetrics:
        return MlflowSelectedMetrics(**asdict(self))


@dataclass(frozen=True)
class MlflowSummaryMetricsDTO:
    checkpoint_metrics: dict[str, float] = field(default_factory=dict)
    system_metrics: dict[str, float] = field(default_factory=dict)
    benchmark_metrics: dict[str, float] = field(default_factory=dict)
    trace_metrics: dict[str, float] = field(default_factory=dict)
    logged_model_metrics: dict[str, float] = field(default_factory=dict)

    def to_domain(self) -> MlflowSummaryMetrics:
        return MlflowSummaryMetrics(**asdict(self))


@dataclass(frozen=True)
class MlflowRunTagsDTO:
    source: str
    goal: str
    tracking_level: str
    model_type: str
    framework: str
    task: str
    registered_model_name: str
    deployment_stage: str
    git_commit: str
    docker_image: str
    host_name: str
    gpu_name: str
    metadata_mocked: str
    tensorboard_dir: str
    checkpoint_dir: str
    dataset_zip_path: str
    manifest_path: str

    def to_domain(self) -> MlflowRunTags:
        return MlflowRunTags(**asdict(self))


@dataclass(frozen=True)
class MlflowRunParamsDTO:
    run_created_from: str
    last_epoch: int
    last_epoch_idx: int
    objective_metric_name: str
    objective_metric_mode: str
    best_objective_epoch: int
    best_objective_value_from_last_ckpt: float
    validation_metric_name: str
    best_validation_metric_epoch: int
    best_validation_metric_value_from_last_ckpt: float
    encoder_num_classes: int
    encoder_vocab_size: int
    model_total_params: int
    checkpoint_saved_by: str
    checkpoint_version: str
    manifest_dataset_name: str
    manifest_dataset_version: str
    manifest_task: str
    dataset_zip_size_bytes: int
    dataset_zip_sha256: str
    manifest_sha256: str
    python_version: str
    pytorch_version: str
    platform: str
    cuda_version: str
    driver_version: str
    gpu_memory_gb: float
    registered_model_version: str
    pipeline_name: str
    pipeline_run_id: str

    def to_domain(self) -> MlflowRunParams:
        return MlflowRunParams(**asdict(self))


@dataclass(frozen=True)
class MlflowArtifactPathsDTO:
    tensorboard_dir: str
    checkpoint_dir: str
    dataset_zip_path: str
    manifest_path: str
    last_checkpoint_path: str
    best_loss_checkpoint_path: str
    best_objective_checkpoint_path: str

    def to_domain(self) -> MlflowArtifactPaths:
        return MlflowArtifactPaths(**asdict(self))


@dataclass(frozen=True)
class MlflowLoggedModelDTO:
    model_artifact_name: str
    model_family: str
    model_type: str
    source: str
    is_demo_model: bool

    def to_domain(self) -> MlflowLoggedModel:
        return MlflowLoggedModel(**asdict(self))


@dataclass(frozen=True)
class MlflowProductionRunMetadataDTO:
    tracking_uri: str
    experiment_name: str
    run_name: str
    environment: MlflowEnvironmentDTO
    lineage: MlflowLineageDTO
    registry: MlflowRegistryDTO
    signature: MlflowModelSignatureDTO
    evaluation: MlflowEvaluationDTO
    predictions: MlflowPredictionSetDTO
    model_card: MlflowModelCardDTO
    production_summary: MlflowProductionSummaryDTO
    run_tags: MlflowRunTagsDTO
    run_params: MlflowRunParamsDTO
    summary_metrics: MlflowSummaryMetricsDTO
    selected_metrics: MlflowSelectedMetricsDTO
    artifact_paths: MlflowArtifactPathsDTO
    logged_model: MlflowLoggedModelDTO

    def to_domain(self) -> MlflowProductionRunMetadata:
        return MlflowProductionRunMetadata(
            tracking_uri=self.tracking_uri,
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            environment=self.environment.to_domain(),
            lineage=self.lineage.to_domain(),
            registry=self.registry.to_domain(),
            signature=self.signature.to_domain(),
            evaluation=self.evaluation.to_domain(),
            predictions=self.predictions.to_domain(),
            model_card=self.model_card.to_domain(),
            production_summary=self.production_summary.to_domain(),
            run_tags=self.run_tags.to_domain(),
            run_params=self.run_params.to_domain(),
            summary_metrics=self.summary_metrics.to_domain(),
            selected_metrics=self.selected_metrics.to_domain(),
            artifact_paths=self.artifact_paths.to_domain(),
            logged_model=self.logged_model.to_domain(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
