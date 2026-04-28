from __future__ import annotations

import torch
import torch.nn as nn

from src.domain.ports.common.common_component_ports import CacheableActivationPort


class _SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(a)
        y = (a * s) * b
        ctx.save_for_backward(a, b, s)
        return y

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        a, b, s = ctx.saved_tensors

        grad_a = None
        grad_b = None

        if ctx.needs_input_grad[0]:
            grad_a = grad_output * b * (s + a * s * (1 - s))
        if ctx.needs_input_grad[1]:
            grad_b = grad_output * (a * s)

        return grad_a, grad_b


class SwiGLU(nn.Module, CacheableActivationPort):
    def __init__(
        self,
        device: str | torch.device,
        out_feature: int,
        enable_cache: bool = False,
    ) -> None:
        super().__init__()
        if out_feature <= 0:
            raise ValueError(
                f"SwiGLU.__init__: out_feature must be > 0, got {out_feature}"
            )

        self.out_feature = out_feature
        self.enable_cache = enable_cache

        self.w_1 = nn.LazyLinear(
            out_features=self.out_feature,
            bias=False,
            device=device,
        )
        self.w_2 = nn.LazyLinear(
            out_features=self.out_feature,
            bias=False,
            device=device,
        )

        self.cache: dict[str, torch.Tensor] = {}

    def clear_cache(self) -> None:
        self.cache.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_w_1 = self.w_1(x)
        out_w_2 = self.w_2(x)

        if self.enable_cache:
            s = torch.sigmoid(out_w_1)
            self.cache = {
                "out_w_1": out_w_1.detach(),
                "out_w_2": out_w_2.detach(),
                "sigmoid": s.detach(),
            }
        else:
            self.cache.clear()

        return _SwiGLUFunction.apply(out_w_1, out_w_2)

    def extra_repr(self) -> str:
        return (
            f"out_feature={self.out_feature}, "
            f"enable_cache={self.enable_cache}"
        )
