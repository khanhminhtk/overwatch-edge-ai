from __future__ import annotations

import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.dataset.application.use_case.create_god_dataset_detection import (
    CreateGodDatasetDetection,
)
from src.platform.logger import Logger


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create god dataset for detection")
    parser.add_argument("--source-data-path", required=True)
    parser.add_argument("--subset-percent", type=float, default=10.0)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dataset-version", default="v1")
    args = parser.parse_args()

    logger = Logger("GodDatasetDetection")
    creator = CreateGodDatasetDetection(logger=logger)
    result = creator.execute(
        source_data_path=args.source_data_path,
        subset_percent=args.subset_percent,
        output_root=args.output_root,
        dataset_version=args.dataset_version,
    )
    print(result)


if __name__ == "__main__":
    main()
