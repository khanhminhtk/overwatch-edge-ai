from typing import Any, Iterable, List, Optional

import torch.nn as nn

from .common import shape_of


class ShapeLoggerHook:
    def __init__(
        self,
        model: nn.Module,
        max_depth: int = 10,
        leaf_only: bool = False,
        include_names: Optional[Iterable[str]] = None,
    ) -> None:
        self.model = model
        self.max_depth = max_depth
        self.leaf_only = leaf_only
        self.include_names = set(include_names) if include_names is not None else None
        self._handles: List[Any] = []

    def _should_hook(self, name: str, module: nn.Module) -> bool:
        if name == "":
            return False

        if self.include_names is not None:
            return name in self.include_names

        if name.count(".") > self.max_depth:
            return False

        if self.leaf_only and any(True for _ in module.children()):
            return False

        return True

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            print(f"\n[ShapeLogger] {name}")
            print(f"  input : {shape_of(inputs)}")
            print(f"  output: {shape_of(output)}")

        return hook

    def __enter__(self) -> "ShapeLoggerHook":
        if self._handles:
            raise RuntimeError("ShapeLoggerHook.__enter__: hooks are already registered.")

        for name, module in self.model.named_modules():
            if self._should_hook(name, module):
                handle = module.register_forward_hook(self._make_hook(name))
                self._handles.append(handle)

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()

        self._handles.clear()
