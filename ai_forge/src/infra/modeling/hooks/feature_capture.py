from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn


class FeatureCaptureHook:
    def __init__(
        self,
        model: nn.Module,
        module_names: Iterable[str],
        detach: bool = True,
        clone: bool = False,
        to_cpu: bool = False,
    ) -> None:
        self.model = model
        if isinstance(module_names, str):
            self.module_names = [module_names]
        else:
            self.module_names = list(module_names)
        self.detach = detach
        self.clone = clone
        self.to_cpu = to_cpu

        self.outputs: Dict[str, Any] = {}
        self._handles: List[Any] = []
        self._modules = dict(model.named_modules())

        missing = [name for name in self.module_names if name not in self._modules]
        if missing:
            available = list(self._modules.keys())
            preview = available[:30]
            raise ValueError(
                f"FeatureCaptureHook: modules not found: {missing}. "
                f"Available examples: {preview}"
            )

    def _process_output(self, value: Any) -> Any:
        if torch.is_tensor(value):
            if self.detach:
                value = value.detach()
            if self.clone:
                value = value.clone()
            if self.to_cpu:
                value = value.cpu()

        elif isinstance(value, tuple):
            value = tuple(self._process_output(v) for v in value)

        elif isinstance(value, list):
            value = [self._process_output(v) for v in value]

        elif isinstance(value, dict):
            value = {k: self._process_output(v) for k, v in value.items()}

        return value

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any):
            self.outputs[name] = self._process_output(output)

        return hook

    def register(self) -> None:
        if self._handles:
            raise RuntimeError("FeatureCaptureHooks.register: hooks are already registered.")

        for name in self.module_names:
            module = self._modules[name]
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def clear(self) -> None:
        self.outputs.clear()

    def __enter__(self) -> "FeatureCaptureHook":
        self.register()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.remove()
