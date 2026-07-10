from pydantic.dataclasses import dataclass

@dataclass(frozen=True, kw_only=True, slots=True)
class MlflowConfig:
    tracking_uri: str

    def __post_init__(self) -> None:
        if not self.tracking_uri.strip():
            raise ValueError("tracking_uri must not be empty")