from pathlib import Path

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
            str(self._resolve_path(self.certificate_path, base_path)),
        )
        if self.image_test_path is not None:
            object.__setattr__(
                self,
                "image_test_path",
                str(self._resolve_path(self.image_test_path, base_path)),
            )

    @staticmethod
    def _resolve_path(path_value: str, base_path: str) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path

        service_root = Path(base_path).resolve()
        repo_root = service_root.parent.parent
        service_candidate = service_root / path
        repo_candidate = repo_root / path

        if service_candidate.exists():
            return service_candidate
        if repo_candidate.exists():
            return repo_candidate
        if path.parts and path.parts[0] == "services":
            return repo_candidate
        return service_candidate
