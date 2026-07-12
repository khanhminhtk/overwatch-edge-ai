from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image_rgb(image_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float64") / 255.0


def iter_image_files(directory: Path) -> Iterator[Path]:
    if not directory.exists():
        return
    yield from (
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def is_corrupted_image(image_path: Path) -> bool:
    return load_image_rgb(image_path) is None


def compute_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def zip_directory(src_dir: Path, output_zip: Path) -> Path:
    if not src_dir.is_dir():
        raise NotADirectoryError(str(src_dir))
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in src_dir.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(src_dir))
    return output_zip
