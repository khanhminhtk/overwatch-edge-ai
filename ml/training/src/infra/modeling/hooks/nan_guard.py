from typing import Any, Iterable, List, Optional

import torch.nn as nn

from .common import assert_finite


class NaNInfGuardHook:
    def __init__(
        self,
        model: nn.Module,
        check_inputs: bool = False,
        check_outputs: bool = True,
        module_names: Optional[Iterable[str]] = None,
    ):
        self.model = model
        self.check_inputs = check_inputs
        self.check_outputs = check_outputs
        self.module_names = set(module_names) if module_names is not None else None
        self._handles: List[Any] = []

    def _should_hook(self, name: str) -> bool:
        if name == "":
            return False

        if self.module_names is None:
            return True

        return name in self.module_names

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if self.check_inputs:
                assert_finite(f"{name}.inputs", inputs)

            if self.check_outputs:
                assert_finite(f"{name}.output", output)

        return hook

    def __enter__(self) -> "NaNInfGuardHook":
        if self._handles:
            raise RuntimeError("NaNInfGuard.__enter__: hooks are already registered.")

        for name, module in self.model.named_modules():
            if self._should_hook(name):
                handle = module.register_forward_hook(self._make_hook(name))
                self._handles.append(handle)

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()

        self._handles.clear()
