from pydantic.dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class GoogleVisionConfig:
    certificate_path: str
    image_test_path: str | None = None

    def __post_init__(self) -> None:
        if not self.certificate_path.strip():
            raise ValueError("certificate_path must not be empty")
        
    def convert_path_to_absolute(self, base_path: str) -> None:
        object.__setattr__(
            self,
            "certificate_path",
            f"{base_path}/{self.certificate_path}",
        )
        if self.image_test_path is not None:
            object.__setattr__(
                self,
                "image_test_path",
                f"{base_path}/{self.image_test_path}",
            )
