import zipfile
from pathlib import Path

def zip_directory(src_dir: str | Path, output_zip: str | Path) -> Path:
    src_dir = Path(src_dir)
    output_zip = Path(output_zip)

    if not src_dir.is_dir():
        raise NotADirectoryError(f"{src_dir} is not a directory")

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file_path in src_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(src_dir)
                zipf.write(file_path, arcname=arcname)

    return output_zip