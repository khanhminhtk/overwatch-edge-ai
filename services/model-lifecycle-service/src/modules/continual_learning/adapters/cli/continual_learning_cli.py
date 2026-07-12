from __future__ import annotations

import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[6]
service_root = str(SERVICE_ROOT)
if service_root not in sys.path:
    sys.path.insert(0, service_root)

from src.modules.continual_learning.application.process_raw_images import ProcessRawImages
from src.platform.config import ConfigLoader
from src.platform.logger import Logger
from src.platform.vision import GoogleVisionConfig, GoogleVisionOCR


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Process raw images for continual learning")
    parser.add_argument("--raw-dir", default="data/data_continue_learning/raw")
    parser.add_argument("--output-dir", default="data/data_continue_learning/processed")
    parser.add_argument("--class-id", type=int, default=0)
    args = parser.parse_args()

    logger = Logger("ContinualLearning")
    env_base = str(Path(__file__).resolve().parents[4])

    cfg = ConfigLoader.load(
        GoogleVisionConfig,
        yaml_files=[f"{env_base}/config/model_lifecycle_orchestrator_config.yaml"],
        env_files=[f"{env_base}/config/.env"],
        section="GoogleVision",
    )
    cfg.convert_path_to_absolute(env_base)
    vision = GoogleVisionOCR(config=cfg)

    processor = ProcessRawImages(vision_model=vision, logger=logger)
    results = processor.execute(raw_dir=args.raw_dir, output_dir=args.output_dir, class_id=args.class_id)

    for r in results:
        print(f"  {r.source_image}: detection={r.detection_samples} recognizer={r.recognizer_samples}")


if __name__ == "__main__":
    main()
