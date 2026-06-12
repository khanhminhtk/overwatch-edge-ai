from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class MlflowEnvironment:
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


@dataclass(frozen=True)
class MlflowLineage:
    is_mocked: bool
    git_commit: str
    git_branch: str
    pipeline_name: str
    pipeline_run_id: str
    orchestrator_job_id: str
    parent_run_id: str


@dataclass(frozen=True)
class MlflowRegistry:
    is_mocked: bool
    registered_model_name: str
    model_version: str
    deployment_stage: str
    approval_status: str
    owner_team: str


@dataclass(frozen=True)
class MlflowTensorFieldSignature:
    name: str
    dtype: str
    shape: list[int]
    description: str


@dataclass(frozen=True)
class MlflowModelSignature:
    is_mocked: bool
    input_signature: MlflowTensorFieldSignature
    output_signature: MlflowTensorFieldSignature


@dataclass(frozen=True)
class MlflowEvaluation:
    is_mocked: bool
    inference_latency_ms_p50: float
    inference_latency_ms_p95: float
    throughput_samples_per_sec: float
    train_duration_minutes: float
    cpu_utilization_avg: float
    ram_used_gb: float
    business_gate: str


@dataclass(frozen=True)
class MlflowPredictionSample:
    image_id: str
    ground_truth: str
    prediction: str
    confidence: float


@dataclass(frozen=True)
class MlflowPredictionSet:
    is_mocked: bool
    samples: list[MlflowPredictionSample] = field(default_factory=list)


@dataclass(frozen=True)
class MlflowModelCard:
    model_name: str
    purpose: str
    architecture: str
    training_status: str
    data_source: str
    known_limitations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MlflowProductionSummary:
    run_goal: str
    uses_real_tensorboard_metrics: bool
    uses_real_checkpoints: bool
    uses_real_dataset_zip: bool
    uses_real_manifest_even_if_wrong: bool
    contains_mock_metadata_sections: bool


@dataclass(frozen=True)
class MlflowSelectedMetrics:
    train_final_metrics: dict[str, float | None] = field(default_factory=dict)
    train_best_metrics: dict[str, float | None] = field(default_factory=dict)
    val_final_metrics: dict[str, float | None] = field(default_factory=dict)
    val_best_metrics: dict[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class MlflowSummaryMetrics:
    checkpoint_metrics: dict[str, float] = field(default_factory=dict)
    system_metrics: dict[str, float] = field(default_factory=dict)
    benchmark_metrics: dict[str, float] = field(default_factory=dict)
    trace_metrics: dict[str, float] = field(default_factory=dict)
    logged_model_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MlflowRunTags:
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


@dataclass(frozen=True)
class MlflowRunParams:
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


@dataclass(frozen=True)
class MlflowArtifactPaths:
    tensorboard_dir: str
    checkpoint_dir: str
    dataset_zip_path: str
    manifest_path: str
    last_checkpoint_path: str
    best_loss_checkpoint_path: str
    best_objective_checkpoint_path: str


@dataclass(frozen=True)
class MlflowLoggedModel:
    model_artifact_name: str
    model_family: str
    model_type: str
    source: str
    is_demo_model: bool


@dataclass(frozen=True)
class MlflowProductionRunMetadata:
    tracking_uri: str
    experiment_name: str
    run_name: str
    environment: MlflowEnvironment
    lineage: MlflowLineage
    registry: MlflowRegistry
    signature: MlflowModelSignature
    evaluation: MlflowEvaluation
    predictions: MlflowPredictionSet
    model_card: MlflowModelCard
    production_summary: MlflowProductionSummary
    run_tags: MlflowRunTags
    run_params: MlflowRunParams
    summary_metrics: MlflowSummaryMetrics
    selected_metrics: MlflowSelectedMetrics
    artifact_paths: MlflowArtifactPaths
    logged_model: MlflowLoggedModel

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
