from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class TrackingArtifactDownloadSpec:
    artifact_path: str
    root_dir: str

    @classmethod
    def for_detection(cls, artifact_path: str) -> "TrackingArtifactDownloadSpec":
        return cls._build(
            artifact_path=artifact_path,
            allowed_roots={"checkpoints", "dataset", "training"},
        )

    @classmethod
    def for_recognizer(cls, artifact_path: str) -> "TrackingArtifactDownloadSpec":
        return cls._build(
            artifact_path=artifact_path,
            allowed_roots={"checkpoints", "dataset", "tensorboard"},
        )

    @classmethod
    def _build(
        cls,
        *,
        artifact_path: str,
        allowed_roots: set[str],
    ) -> "TrackingArtifactDownloadSpec":
        normalized = artifact_path.strip().strip("/")
        if not normalized:
            raise ValueError("artifact_path must not be empty")

        root_dir = normalized.split("/", 1)[0]
        if root_dir not in allowed_roots:
            raise ValueError(
                f"Unsupported artifact root: {root_dir}. Allowed roots: {sorted(allowed_roots)}"
            )

        return cls(
            artifact_path=normalized,
            root_dir=root_dir,
        )
