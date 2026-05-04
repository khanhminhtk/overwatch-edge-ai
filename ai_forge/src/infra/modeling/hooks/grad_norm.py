from typing import Any, Dict, Iterable, List

import torch.nn as nn

from .common import tensor_norm


class GradNormHook:
    def __init__(self, model: nn.Module, module_names: Iterable[str]):
        self.model = model
        if isinstance(module_names, str):
            self.module_names = [module_names]
        else:
            self.module_names = list(module_names)
        self.grad_input_norms: Dict[str, float] = {}
        self.grad_output_norms: Dict[str, float] = {}
        self._handles: List[Any] = []
        self._modules = dict(model.named_modules())

        missing = [name for name in self.module_names if name not in self._modules]
        if missing:
            available = list(self._modules.keys())
            preview = available[:30]
            raise ValueError(
                f"GradNormHook: modules not found: {missing}. "
                f"Available examples: {preview}"
            )

    def _make_hook(self, name: str):
        def hook(
            module: nn.Module,
            grad_input: tuple[Any, ...],
            grad_output: tuple[Any, ...],
        ) -> None:
            in_norm = tensor_norm(grad_input)
            out_norm = tensor_norm(grad_output)

            if in_norm is not None:
                self.grad_input_norms[name] = in_norm

            if out_norm is not None:
                self.grad_output_norms[name] = out_norm

        return hook

    def __enter__(self) -> "GradNormHook":
        for name in self.module_names:
            module = self._modules[name]
            handle = module.register_full_backward_hook(self._make_hook(name))
            self._handles.append(handle)

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()

        self._handles.clear()
