from __future__ import annotations

import runpy
import sys
from pathlib import Path


MODULE_NAME = (
    "services.model-lifecycle-service.src.applications.use_cases.mlflow.recognizer.mlflow_tracking"
)
SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def test_mlflow_tracking_module_imports_from_repo_root() -> None:
    original_sys_path = sys.path[:]

    try:
        sys.path = [
            path
            for path in original_sys_path
            if Path(path or ".").resolve() != SERVICE_ROOT
        ]
        repo_root = str(REPO_ROOT)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        runpy.run_module(MODULE_NAME, run_name="__test__")
    finally:
        sys.path = original_sys_path
