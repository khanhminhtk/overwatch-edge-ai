from typing import Any, Dict, List, Optional

import torch


def shape_of(value: Any) -> Any:
    if torch.is_tensor(value):
        return {
            "type": "tensor",
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }

    if isinstance(value, tuple):
        return [shape_of(v) for v in value]

    if isinstance(value, list):
        return [shape_of(v) for v in value]

    if isinstance(value, dict):
        return {k: shape_of(v) for k, v in value.items()}

    return type(value).__name__


def tensor_norm(value: Any) -> Optional[float]:
    norms: List[float] = []

    def collect(x: Any) -> None:
        if torch.is_tensor(x):
            if x.numel() > 0:
                norms.append(x.detach().float().norm().item())
        elif isinstance(x, (tuple, list)):
            for item in x:
                collect(item)
        elif isinstance(x, dict):
            for item in x.values():
                collect(item)

    collect(value)

    if not norms:
        return None

    return sum(norms) / len(norms)


def assert_finite(name: str, value: Any) -> None:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return

        if not torch.isfinite(value).all():
            finite_ratio = torch.isfinite(value).float().mean().item()
            raise FloatingPointError(
                f"NaN/Inf detected at {name}. "
                f"shape={tuple(value.shape)}, "
                f"dtype={value.dtype}, "
                f"device={value.device}, "
                f"finite_ratio={finite_ratio:.6f}"
            )

    elif isinstance(value, tuple):
        for i, item in enumerate(value):
            assert_finite(f"{name}[{i}]", item)

    elif isinstance(value, list):
        for i, item in enumerate(value):
            assert_finite(f"{name}[{i}]", item)

    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite(f"{name}.{key}", item)
