from __future__ import annotations

import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[5]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.dataset.application.manifest_builder import ManifestBuilder
from src.platform.logger import Logger


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create dataset manifest for existing data")
    parser.add_argument("--data-path", required=True, help="Path to dataset root")
    parser.add_argument(
        "--data-type", required=True, choices=["detection", "recognition"],
        help="Type of dataset",
    )
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--archive-path", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_path).expanduser().resolve()
    if not data_root.exists():
        print(f"Error: data path does not exist: {data_root}", file=sys.stderr)
        sys.exit(1)

    archive_path = Path(args.archive_path).expanduser().resolve() if args.archive_path else None

    logger = Logger(f"Manifest-{args.data_type}")
    builder = ManifestBuilder(data_types=args.data_type, logger=logger)
    manifest, archive, manifest_path = builder.build(
        data_root,
        dataset_version=args.dataset_version,
        dataset_name=args.dataset_name,
        archive_path=archive_path,
    )
    print(f"Manifest: {manifest_path}")
    print(f"Archive:  {archive}")
    print(f"Dataset:  {manifest['dataset_name']} v{manifest['dataset_version']}")


if __name__ == "__main__":
    main()
