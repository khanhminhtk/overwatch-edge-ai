from __future__ import annotations

import os
from pathlib import Path


def normalize_object_storage_target(bucket_name: str, object_name: str) -> tuple[str, str]:
    bucket_parts = [part for part in bucket_name.split("/") if part]
    normalized_object_name = object_name.lstrip("/")

    if len(bucket_parts) <= 1:
        return bucket_name, normalized_object_name

    normalized_bucket_name = bucket_parts[0]
    object_prefix = "/".join(bucket_parts[1:])
    if normalized_object_name:
        normalized_object_name = f"{object_prefix}/{normalized_object_name}"
    else:
        normalized_object_name = object_prefix
    return normalized_bucket_name, normalized_object_name


def build_upload_object_name(
    object_name: str,
    version: str,
    source_url: str,
    source_path: str = "",
) -> str:
    normalized_object_name = object_name.strip().strip("/")
    source_file_name = Path(source_url).name.strip()

    if source_file_name:
        object_path = Path(normalized_object_name) if normalized_object_name else None
        treat_as_directory = (
            not normalized_object_name
            or object_name.endswith("/")
            or (object_path is not None and object_path.suffix == "")
        )
        if treat_as_directory:
            normalized_object_name = (
                f"{normalized_object_name}/{source_file_name}"
                if normalized_object_name
                else source_file_name
            )
    elif normalized_object_name and Path(normalized_object_name).suffix == "":
        source_suffix = Path(source_path).suffix.strip()
        if source_suffix:
            normalized_object_name = f"{normalized_object_name}{source_suffix}"

    if version:
        return f"{version}/{normalized_object_name}" if normalized_object_name else version
    return normalized_object_name


def build_download_object_name(object_name: str, version: str) -> str:
    normalized_object_name = object_name.strip().strip("/")
    if version:
        return f"{version}/{normalized_object_name}" if normalized_object_name else version
    return normalized_object_name


def finalize_storage_object_name(object_name: str, source_path: str) -> str:
    normalized_object_name = object_name.strip().strip("/")
    if not normalized_object_name:
        return normalized_object_name

    object_path = Path(normalized_object_name)
    if object_path.suffix:
        return normalized_object_name

    source_path_obj = Path(source_path)
    if source_path_obj.suffix:
        return f"{normalized_object_name}{source_path_obj.suffix}"

    if source_path and os.path.isdir(source_path):
        return f"{normalized_object_name}.zip"

    return normalized_object_name
