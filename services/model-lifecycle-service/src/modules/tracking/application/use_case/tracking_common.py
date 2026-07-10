from __future__ import annotations

import subprocess
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT, MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE

F = TypeVar("F", bound=Callable[..., Any])


def log_exceptions(tag: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                owner = args[0] if args else None
                logger = getattr(owner, "_logger", None)
                if logger is not None:
                    logger.exception(tag, f"function={func.__name__}", f"error={exc}")
                raise

        return cast(F, wrapper)

    return decorator


def get_commit_hash(logger: Any) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
    except Exception:
        logger.warning("Failed to get git commit hash.")
        return "unknown_commit_hash"


def resolve_path(pwd: str, path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(pwd) / resolved
    return resolved


def build_run_name(run: Any, git_commit: str) -> str:
    return f"{run.experiment_name}_{run.config.model_name}_{git_commit}_{run.config.hardware.gpu_name}"


def build_common_run_tags(config: Any, git_commit: str, source: str) -> dict[str, str]:
    hardware = config.hardware
    return {
        MLFLOW_SOURCE_NAME: source,
        MLFLOW_SOURCE_TYPE: "LOCAL",
        "model_type": config.model_type,
        "task": config.task_name,
        "registered_model_name": config.model_name,
        "deployment_stage": config.deployment_stage,
        MLFLOW_GIT_COMMIT: git_commit,
        "host_name": hardware.gpu_host_name,
        "gpu_name": hardware.gpu_name,
        "framework": config.framework,
    }
