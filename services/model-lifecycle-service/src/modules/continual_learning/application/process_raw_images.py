from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

from src.modules.continual_learning.domain.learned_sample import ProcessedResult
from src.platform.logger import Logger
from src.platform.vision import VisionModelPort


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_RECOGNIZER_PATCHES = 16


class ProcessRawImages:
    def __init__(
        self,
        vision_model: VisionModelPort,
        logger: Logger | None = None,
        min_recognizer_width: int = MIN_RECOGNIZER_PATCHES,
    ) -> None:
        self._vision_model = vision_model
        self._logger = logger or Logger("ProcessRawImages")
        self._min_recognizer_width = min_recognizer_width

    def execute(self, *, raw_dir: str, output_dir: str, class_id: int = 0) -> list[ProcessedResult]:
        raw_path = Path(raw_dir).expanduser().resolve()
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw directory not found: {raw_path}")

        out = Path(output_dir).expanduser().resolve()
        det_img_dir = out / "detection" / "train" / "images"
        det_lbl_dir = out / "detection" / "train" / "labels"
        rec_base = out / "recognizer" / "train_data"
        det_img_dir.mkdir(parents=True, exist_ok=True)
        det_lbl_dir.mkdir(parents=True, exist_ok=True)
        rec_base.mkdir(parents=True, exist_ok=True)

        images = sorted(p for p in raw_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        if not images:
            raise FileNotFoundError(f"No supported images in {raw_path}")

        self._logger.info("[PROCESS_RAW_STARTED]", f"images={len(images)}")
        results = []
        for img_path in tqdm(images, desc="Processing"):
            results.append(self._process_one(img_path, det_img_dir, det_lbl_dir, rec_base, class_id))
        self._logger.info("[PROCESS_RAW_COMPLETED]", f"total={len(images)}")
        return results

    def _process_one(self, img_path, det_img_dir, det_lbl_dir, rec_base, class_id):
        stem = img_path.stem
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to read: {img_path}")
        h, w = image.shape[:2]

        preds = self._vision_model.execute(str(img_path))
        words = preds if isinstance(preds, list) else preds.get("words", [])

        yolo_labels = []
        rec_labels = {}
        rec_count = 0
        group_dir: Path | None = None

        for idx, wp in enumerate(words):
            text = wp.text.strip()
            if not text:
                continue
            bbox = wp.bounding_box
            yolo = bbox.to_yolo(image_width=w, image_height=h)
            yolo_labels.append((class_id, yolo[0], yolo[1], yolo[2], yolo[3]))
            x1, y1 = max(0, int(bbox.x_min)), max(0, int(bbox.y_min))
            x2, y2 = min(w, int(bbox.x_max)), min(h, int(bbox.y_max))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            if crop.shape[1] < self._min_recognizer_width:
                self._logger.info(
                    "[PROCESS_RAW_SKIP_RECOGNIZER_CROP]",
                    f"image={img_path.name}",
                    f"word_index={idx}",
                    f"crop_width={crop.shape[1]}",
                    f"min_width={self._min_recognizer_width}",
                )
                continue
            if group_dir is None:
                group_dir = self._build_group_dir(rec_base=rec_base)
                group_dir.mkdir(parents=True, exist_ok=True)
            crop_name = f"{rec_count + 1}.jpg"
            group_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(group_dir / crop_name), crop)
            rec_labels[crop_name] = text
            rec_count += 1

        det_count = 0
        if yolo_labels:
            shutil.copy2(img_path, det_img_dir / img_path.name)
            det_count = 1
            (det_lbl_dir / f"{stem}.txt").write_text(
                "\n".join(f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}" for c, x, y, bw, bh in yolo_labels)
            )

        if rec_labels:
            (group_dir / "label.json").write_text(
                json.dumps(rec_labels, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        return ProcessedResult(
            source_image=img_path.name,
            detection_dir=str(det_img_dir / img_path.name),
            recognizer_dir=str(group_dir or rec_base),
            detection_samples=det_count,
            recognizer_samples=rec_count,
        )

    @staticmethod
    def _build_group_dir(*, rec_base: Path) -> Path:
        max_existing = 0
        for child in rec_base.iterdir():
            if child.is_dir() and child.name.isdigit():
                max_existing = max(max_existing, int(child.name))
        return rec_base / str(max_existing + 1)
