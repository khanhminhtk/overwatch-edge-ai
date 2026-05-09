from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn.parameter import UninitializedParameter

from src.infra.onnx.clear_pt import load_recognizer_pt
from src.utils.config_loader import ConfigLoader


def export_recognizer_to_onnx(
    checkpoint_path: str,
    output_path: str,
    config_relative_path: str = "config/training/recognizer_ctc.yaml",
    env_relative_path: str = "config/.env",
) -> str:
    project_root = Path(__file__).resolve().parents[3]
    loader = ConfigLoader(
        yaml_relative_paths=[config_relative_path],
        project_root=project_root,
        env_relative_path=env_relative_path,
    )
    model = load_recognizer_pt(config_loader=loader, path=checkpoint_path)
    model.eval()

    recog_cfg = loader.load_recognizer()
    cfg_training = dict(recog_cfg.config_training)
    dataset_cfg = dict(cfg_training.get("dataset", {}))
    num_patches = int(dataset_cfg.get("num_patches", 16))
    patch_size_cfg = dataset_cfg.get("patch_size", [64, 64])
    patch_h = int(patch_size_cfg[0])
    patch_w = int(patch_size_cfg[1])

    dummy_input = torch.randn(1, num_patches, 3, patch_h, patch_w, device="cpu")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    has_uninitialized_params = any(
        isinstance(param, UninitializedParameter) for param in model.parameters()
    )
    if has_uninitialized_params:
        with torch.no_grad():
            model(dummy_input)

    export_kwargs = dict(
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["logits", "aux_loss"],
        dynamic_axes={
            "images": {0: "batch_size", 1: "num_patches"},
            "logits": {0: "batch_size", 1: "num_patches"},
        },
    )

    with torch.no_grad():
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(out_path),
                dynamo=True,
                **export_kwargs,
            )
        except Exception:
            torch.onnx.export(
                model,
                dummy_input,
                str(out_path),
                **export_kwargs,
            )
    return str(out_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export RecognizerCTCModel checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/recognizer/best_loss.pt",
        help="Path to recognizer checkpoint (.pt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/onnx/recognizer.onnx",
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training/recognizer_ctc.yaml",
        help="Recognizer config YAML path relative to project root.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="config/.env",
        help="Dotenv path relative to project root.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out = export_recognizer_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        config_relative_path=args.config,
        env_relative_path=args.env,
    )
    print(f"Recognizer ONNX exported: {out}")


if __name__ == "__main__":
    main()
