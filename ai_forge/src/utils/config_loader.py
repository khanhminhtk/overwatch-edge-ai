from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from src.domain.value_objet.config import DomainConfig, RecognizerConfig


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_INT_PATTERN = re.compile(r"^[+-]?\d+$")
_FLOAT_PATTERN = re.compile(r"^[+-]?(\d+\.\d*|\.\d+)$")


class ConfigLoader:
    def __init__(
        self,
        yaml_relative_paths: list[str],
        project_root: str | Path | None = None,
        env_relative_path: str = "config/.env",
        recognizer_arch_path: tuple[str, ...] = ("recognizer", "architecture"),
    ) -> None:
        if not yaml_relative_paths:
            raise ValueError("ConfigLoader.__init__: yaml_relative_paths must not be empty")
        self.yaml_relative_paths = yaml_relative_paths
        self.project_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
        self.env_relative_path = env_relative_path
        self.recognizer_arch_path = recognizer_arch_path

    def _load_dotenv_file(self, env_path: Path) -> dict[str, str]:
        env: dict[str, str] = {}
        if not env_path.is_file():
            return env

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
        return env

    def _deep_merge(self, base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        for key, value in incoming.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _interpolate_env(self, raw_text: str, env_vars: dict[str, str]) -> str:
        def to_yaml_scalar(value: str) -> str:
            lowered = value.lower()
            if lowered in {"true", "false", "null"}:
                return lowered
            if _INT_PATTERN.fullmatch(value) or _FLOAT_PATTERN.fullmatch(value):
                return value
            if re.fullmatch(r"[A-Za-z0-9_./-]+", value):
                return value
            return "'" + value.replace("'", "''") + "'"

        def replace(match: re.Match[str]) -> str:
            env_key = match.group(1)
            if env_key not in env_vars:
                raise ValueError(f"ConfigLoader._interpolate_env: Missing environment variable for config interpolation: {env_key}")
            return to_yaml_scalar(env_vars[env_key])

        return _ENV_PATTERN.sub(replace, raw_text)

    def load_domain_config(self) -> DomainConfig:
        try:
            import yaml
        except Exception as exc:
            raise ImportError("ConfigLoader.load_domain_config: Config loading requires PyYAML. Install with `pip install pyyaml`.") from exc

        env_path = self.project_root / self.env_relative_path
        env_vars = self._load_dotenv_file(env_path)
        env_vars = {**env_vars, **dict(os.environ)}

        merged: dict[str, Any] = {}
        for rel_path in self.yaml_relative_paths:
            yaml_path = self.project_root / rel_path
            if not yaml_path.is_file():
                raise ValueError(f"ConfigLoader.load_domain_config: Config file not found: {yaml_path}")

            raw_text = yaml_path.read_text(encoding="utf-8")
            resolved_text = self._interpolate_env(raw_text, env_vars)
            parsed = yaml.safe_load(resolved_text)
            if parsed is None:
                parsed = {}
            if not isinstance(parsed, dict):
                raise ValueError(f"ConfigLoader.load_domain_config: Config root must be mapping for file: {yaml_path}")
            self._deep_merge(merged, parsed)

        return DomainConfig(data=merged)

    def load_recognizer(self) -> RecognizerConfig:
        cfg = self.load_domain_config()
        recognizer_arch = cfg.get(*self.recognizer_arch_path)
        if not isinstance(recognizer_arch, dict):
            path_text = ".".join(self.recognizer_arch_path)
            raise ValueError(
                f"ConfigLoader.load_recognizer: Cannot find recognizer architecture config at '{path_text}'."
            )
        recognizer_arch = self._normalize_recognizer_arch(recognizer_arch)
        return RecognizerConfig.from_mapping(recognizer_arch)

    def _normalize_recognizer_arch(self, recognizer_arch: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(recognizer_arch)
        config_training = normalized.get("config_training")
        if not isinstance(config_training, dict):
            return normalized

        config_training = dict(config_training)
        dataset_cfg = config_training.get("dataset")
        if isinstance(dataset_cfg, dict):
            dataset_cfg = dict(dataset_cfg)
            transforms_cfg = dataset_cfg.get("transforms")
            if isinstance(transforms_cfg, dict):
                transforms_cfg = dict(transforms_cfg)
            else:
                transforms_cfg = {}
            transforms_cfg["enabled"] = bool(transforms_cfg.get("enabled", True))
            transforms_cfg["train"] = dict(transforms_cfg.get("train", {}))
            transforms_cfg["val"] = dict(transforms_cfg.get("val", {}))
            dataset_cfg["transforms"] = transforms_cfg
            config_training["dataset"] = dataset_cfg

        normalized["config_training"] = config_training
        return normalized
