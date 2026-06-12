from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.application.use_cases.orchestration.train_recognizer_orchestration import resolve_vocab
from src.application.use_cases.recognizer.build_dataloader import load_dataloader_vit_ctc
from src.application.use_cases.recognizer.load_model_vit_ctc import load_model_vit_ctc
from src.application.use_cases.recognizer.recognizerCTCModel import RecognizerCTCModel
from src.infra.data.dataloaders.ctc_collate import CTCLabelEncoder
from src.utils.config_loader import ConfigLoader


def _build_loader(project_root: Path, config_relative_path: str, env_relative_path: str) -> ConfigLoader:
    return ConfigLoader(
        yaml_relative_paths=[config_relative_path],
        project_root=project_root,
        env_relative_path=env_relative_path,
    )


def _build_dummy_input(loader: ConfigLoader, seed: int = 42) -> torch.Tensor:
    recog_cfg = loader.load_recognizer()
    cfg_training = dict(recog_cfg.config_training)
    dataset_cfg = dict(cfg_training.get("dataset", {}))
    num_patches = int(dataset_cfg.get("num_patches", 16))
    patch_size_cfg = dataset_cfg.get("patch_size", [64, 64])
    patch_h = int(patch_size_cfg[0])
    patch_w = int(patch_size_cfg[1])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(1, num_patches, 3, patch_h, patch_w, generator=generator, device="cpu")


def _torch_latency_ms(model: torch.nn.Module, input_tensor: torch.Tensor, warmup: int, runs: int) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(input_tensor)
        elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000.0


def _export_fp32_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
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
    return output_path


def _convert_fp16(fp32_onnx_path: Path, fp16_onnx_path: Path) -> Path:
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(fp32_onnx_path))
    fp16_model = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(fp16_model, str(fp16_onnx_path))
    return fp16_onnx_path


def _convert_int8(fp32_onnx_path: Path, int8_onnx_path: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        weight_type=QuantType.QInt8,
    )
    return int8_onnx_path


def _run_ort(onnx_path: Path, input_tensor: torch.Tensor, warmup: int, runs: int) -> tuple[float, np.ndarray]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    data = input_tensor.detach().cpu().numpy().astype(np.float32)

    for _ in range(warmup):
        _ = sess.run(None, {input_name: data})

    start = time.perf_counter()
    out: list[np.ndarray] | tuple[np.ndarray, ...] = []
    for _ in range(runs):
        out = sess.run(None, {input_name: data})
    elapsed = time.perf_counter() - start

    first_output = np.asarray(out[0])
    return (elapsed / runs) * 1000.0, first_output


def _torch_output(model: torch.nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        out = model(input_tensor)
    logits = out[0] if isinstance(out, tuple) else out
    return logits.detach().cpu().numpy()


def _mae(ref: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(ref - pred)))


def _speedup_percent(base_ms: float, target_ms: float) -> float:
    if base_ms <= 0:
        return 0.0
    return ((base_ms - target_ms) / base_ms) * 100.0


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _ctc_greedy_decode(logits: np.ndarray, blank_idx: int = 0) -> list[list[int]]:
    pred = np.argmax(logits, axis=-1)  # [B, T]
    decoded: list[list[int]] = []
    for seq in pred:
        tokens: list[int] = []
        prev = None
        for idx in seq.tolist():
            if idx == blank_idx:
                prev = idx
                continue
            if prev == idx:
                continue
            tokens.append(int(idx))
            prev = idx
        decoded.append(tokens)
    return decoded


def _build_val_loader(loader: ConfigLoader) -> tuple[Any, CTCLabelEncoder]:
    recog_cfg = loader.load_recognizer()
    cfg_training = dict(recog_cfg.config_training)
    dataset_cfg = cfg_training["dataset"]
    data_root = Path(dataset_cfg["data_root"])
    train_split = str(dataset_cfg["train_split"])
    vocab = resolve_vocab(config_training=cfg_training, data_root=data_root, train_split=train_split)
    encoder = CTCLabelEncoder(vocab=vocab)
    _, _, _, val_loader = load_dataloader_vit_ctc(config=loader, encoder=encoder)
    return val_loader, encoder


def _extract_model_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("benchmark_recognizer_onnx: checkpoint must be a dict-like object.")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("benchmark_recognizer_onnx: checkpoint does not contain a valid state_dict.")
    return state_dict


def _load_recognizer_checkpoint(loader: ConfigLoader, checkpoint_path: str) -> RecognizerCTCModel:
    recog_cfg = loader.load_recognizer()
    cfg_training = dict(recog_cfg.config_training)
    dataset_cfg = cfg_training["dataset"]
    data_root = Path(dataset_cfg["data_root"])
    train_split = str(dataset_cfg["train_split"])
    vocab = resolve_vocab(config_training=cfg_training, data_root=data_root, train_split=train_split)
    encoder = CTCLabelEncoder(vocab=vocab)

    backbone_model = load_model_vit_ctc(config=loader, device="cpu")
    model = RecognizerCTCModel(model=backbone_model, vocab_size=encoder.num_classes).to("cpu")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_model_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _cer_onnx(onnx_path: Path, val_loader: Any, encoder: CTCLabelEncoder) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    total_edit_distance = 0
    total_chars = 0
    for batch in val_loader:
        images = batch.images.detach().cpu().numpy().astype(np.float32)
        outputs = sess.run(None, {input_name: images})
        logits = np.asarray(outputs[0])
        pred_tokens = _ctc_greedy_decode(logits, blank_idx=0)
        pred_texts = [encoder.decode(seq) for seq in pred_tokens]
        for pred_text, gt_text in zip(pred_texts, batch.texts):
            total_edit_distance += _levenshtein_distance(pred_text, gt_text)
            total_chars += max(len(gt_text), 1)

    return float(total_edit_distance / max(total_chars, 1))


def benchmark_recognizer_onnx(
    checkpoint_path: str,
    output_dir: str,
    config_relative_path: str = "config/training/recognizer_ctc.yaml",
    env_relative_path: str = "config/.env",
    warmup: int = 10,
    runs: int = 50,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[4]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = _build_loader(project_root, config_relative_path, env_relative_path)
    model = _load_recognizer_checkpoint(loader, checkpoint_path)
    dummy_input = _build_dummy_input(loader)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    best_cer_checkpoint = float(checkpoint.get("best_val_cer", float("nan"))) if isinstance(checkpoint, dict) else float("nan")
    val_loader, encoder = _build_val_loader(loader)

    fp32_onnx_path = out_dir / "recognizer_fp32.onnx"
    fp16_onnx_path = out_dir / "recognizer_fp16.onnx"
    int8_onnx_path = out_dir / "recognizer_int8.onnx"

    _export_fp32_onnx(model, dummy_input, fp32_onnx_path)
    _convert_fp16(fp32_onnx_path, fp16_onnx_path)
    _convert_int8(fp32_onnx_path, int8_onnx_path)

    torch_ms = _torch_latency_ms(model, dummy_input, warmup=warmup, runs=runs)
    torch_logits = _torch_output(model, dummy_input)

    fp32_ms, fp32_logits = _run_ort(fp32_onnx_path, dummy_input, warmup=warmup, runs=runs)
    fp16_ms, fp16_logits = _run_ort(fp16_onnx_path, dummy_input, warmup=warmup, runs=runs)
    int8_ms, int8_logits = _run_ort(int8_onnx_path, dummy_input, warmup=warmup, runs=runs)
    fp16_cer = _cer_onnx(fp16_onnx_path, val_loader, encoder)
    int8_cer = _cer_onnx(int8_onnx_path, val_loader, encoder)

    summary = {
        "checkpoint": checkpoint_path,
        "latency_ms": {
            "pytorch_checkpoint": torch_ms,
            "onnx_fp32": fp32_ms,
            "onnx_fp16": fp16_ms,
            "onnx_int8": int8_ms,
        },
        "speedup_percent_vs_checkpoint": {
            "onnx_fp32": _speedup_percent(torch_ms, fp32_ms),
            "onnx_fp16": _speedup_percent(torch_ms, fp16_ms),
            "onnx_int8": _speedup_percent(torch_ms, int8_ms),
        },
        "output_mae_vs_checkpoint": {
            "onnx_fp32": _mae(torch_logits, fp32_logits),
            "onnx_fp16": _mae(torch_logits, fp16_logits),
            "onnx_int8": _mae(torch_logits, int8_logits),
        },
        "cer": {
            "checkpoint_best_val_cer": best_cer_checkpoint,
            "onnx_fp16_val_cer": fp16_cer,
            "onnx_int8_val_cer": int8_cer,
            "delta_vs_checkpoint": {
                "onnx_fp16": fp16_cer - best_cer_checkpoint if not np.isnan(best_cer_checkpoint) else None,
                "onnx_int8": int8_cer - best_cer_checkpoint if not np.isnan(best_cer_checkpoint) else None,
            },
        },
        "artifacts": {
            "onnx_fp32": str(fp32_onnx_path),
            "onnx_fp16": str(fp16_onnx_path),
            "onnx_int8": str(int8_onnx_path),
        },
        "benchmark_config": {"warmup": warmup, "runs": runs},
    }

    summary_path = out_dir / "summary_recognizer_onnx.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_file"] = str(summary_path)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark recognizer checkpoint vs ONNX FP16/INT8.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/recognizer/best_loss.pt",
        help="Recognizer checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory to save ONNX models and summary.",
    )
    parser.add_argument("--config", type=str, default="config/training/recognizer_ctc.yaml")
    parser.add_argument("--env", type=str, default="config/.env")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = benchmark_recognizer_onnx(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        config_relative_path=args.config,
        env_relative_path=args.env,
        warmup=args.warmup,
        runs=args.runs,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
