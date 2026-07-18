#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tritonclient.grpc as grpcclient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=Path("82777d8a836702395b76.jpg"),
        help="path to the input image (default: 82777d8a836702395b76.jpg)",
    )
    parser.add_argument("--url", default="localhost:8001", help="Triton gRPC endpoint")
    parser.add_argument("--model", default="detector", help="Triton model name")
    parser.add_argument("--version", default="", help="optional Triton model version")
    parser.add_argument("--width", type=int, default=640, help="detector input width")
    parser.add_argument("--height", type=int, default=640, help="detector input height")
    return parser.parse_args()


def preprocess(image_path: Path, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("--width and --height must be positive")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    rgb = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2RGB)
    return (
        np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...], dtype=np.float32)
        / 255.0
    )


def main() -> int:
    args = parse_args()
    tensor = preprocess(args.image, args.width, args.height)
    client = grpcclient.InferenceServerClient(url=args.url, verbose=False)

    if not client.is_server_live() or not client.is_server_ready():
        raise RuntimeError(f"Triton server is not ready at {args.url}")
    if not client.is_model_ready(args.model, args.version):
        version = f" version {args.version}" if args.version else ""
        raise RuntimeError(f"model {args.model!r}{version} is not ready")

    metadata = client.get_model_metadata(args.model, args.version)
    input_names = {item.name for item in metadata.inputs}
    output_names = [item.name for item in metadata.outputs]
    if "images" not in input_names:
        raise RuntimeError(
            f"model input 'images' not found; server exposes {sorted(input_names)}"
        )
    if "output0" not in output_names:
        raise RuntimeError(
            f"model output 'output0' not found; server exposes {output_names}"
        )

    request_input = grpcclient.InferInput("images", tensor.shape, "FP32")
    request_input.set_data_from_numpy(tensor)
    result = client.infer(
        model_name=args.model,
        model_version=args.version,
        inputs=[request_input],
        outputs=[grpcclient.InferRequestedOutput("output0")],
    )
    output = result.as_numpy("output0")
    if output is None:
        raise RuntimeError("Triton response did not include output0")

    print(
        f"server={args.url} model={args.model} input=images shape={tuple(tensor.shape)}"
    )
    print(
        f"output=output0 shape={tuple(output.shape)} dtype={output.dtype} "
        f"min={output.min():.6f} max={output.max():.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
