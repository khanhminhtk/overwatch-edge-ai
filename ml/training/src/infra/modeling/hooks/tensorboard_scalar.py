from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from .common import tensor_norm


class TensorBoardScalarHook:
    def __init__(
        self,
        model: nn.Module,
        log_dir: str,
        module_names: Optional[Iterable[str]] = None,
        log_every_n_steps: int = 1,
        leaf_only: bool = True,
        log_histograms_every_n_steps: int = 0,
        flush_every_n_steps: int = 100,
        max_modules: Optional[int] = None,
    ) -> None:
        if log_every_n_steps <= 0:
            raise ValueError("TensorBoardScalarHook.__init__: log_every_n_steps must be > 0")
        if log_histograms_every_n_steps < 0:
            raise ValueError("TensorBoardScalarHook.__init__: log_histograms_every_n_steps must be >= 0")
        if flush_every_n_steps <= 0:
            raise ValueError("TensorBoardScalarHook.__init__: flush_every_n_steps must be > 0")
        if max_modules is not None and max_modules <= 0:
            raise ValueError("TensorBoardScalarHook.__init__: max_modules must be > 0 when provided")

        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:
            raise ImportError(
                "TensorBoardScalarHook.__init__: requires tensorboard. Install with `pip install tensorboard`."
            ) from exc

        self.model = model
        self.writer = SummaryWriter(log_dir=log_dir)
        self.module_names = set(module_names) if module_names is not None else None
        self.leaf_only = leaf_only
        self.log_every_n_steps = log_every_n_steps
        self.log_histograms_every_n_steps = log_histograms_every_n_steps
        self.flush_every_n_steps = flush_every_n_steps
        self.max_modules = max_modules
        self.global_step = 0
        self._handles: List[Any] = []
        self._modules = dict(model.named_modules())

        if self.module_names is not None:
            missing = [name for name in self.module_names if name not in self._modules]
            if missing:
                available = list(self._modules.keys())
                preview = available[:30]
                raise ValueError(
                    f"TensorBoardScalarHook.__init__: modules not found: {missing}. "
                    f"Available examples: {preview}"
                )

    def _should_hook(self, name: str, module: nn.Module) -> bool:
        if name == "":
            return False
        if self.module_names is None:
            if self.leaf_only and any(True for _ in module.children()):
                return False
            return True
        return name in self.module_names

    def _tensor_stats(self, value: Any) -> Dict[str, float]:
        if not torch.is_tensor(value) or value.numel() == 0:
            return {}
        v = value.detach().float()
        return {
            "norm": float(v.norm().item()),
            "mean": float(v.mean().item()),
            "std": float(v.std(unbiased=False).item()),
            "finite_ratio": float(torch.isfinite(v).float().mean().item()),
        }

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if self.global_step % self.log_every_n_steps != 0:
                return

            stats = self._tensor_stats(output)
            for key, val in stats.items():
                self.writer.add_scalar(f"{name}/output_{key}", val, self.global_step)

            in_norm = tensor_norm(inputs)
            out_norm = tensor_norm(output)
            if in_norm is not None:
                self.writer.add_scalar(f"{name}/input_norm", in_norm, self.global_step)
            if out_norm is not None:
                self.writer.add_scalar(f"{name}/output_norm", out_norm, self.global_step)
            if (
                self.log_histograms_every_n_steps > 0
                and torch.is_tensor(output)
                and output.numel() > 0
                and self.global_step % self.log_histograms_every_n_steps == 0
            ):
                self.writer.add_histogram(f"{name}/output_hist", output.detach().float(), self.global_step)

        return hook

    def step(self, n: int = 1) -> None:
        if n <= 0:
            raise ValueError(f"TensorBoardScalarHook.step: n must be > 0, got {n}")
        self.global_step += n
        if self.global_step % self.flush_every_n_steps == 0:
            self.writer.flush()

    def log_optimizer_lrs(self, optimizer: torch.optim.Optimizer, tag: str = "train/lr") -> None:
        for idx, group in enumerate(optimizer.param_groups):
            if "lr" in group:
                self.writer.add_scalar(f"{tag}/group_{idx}", float(group["lr"]), self.global_step)

    def __enter__(self) -> "TensorBoardScalarHook":
        if self._handles:
            raise RuntimeError("TensorBoardScalarHook.__enter__: hooks are already registered.")
        registered = 0
        for name, module in self._modules.items():
            if self._should_hook(name, module):
                handle = module.register_forward_hook(self._make_hook(name))
                self._handles.append(handle)
                registered += 1
                if self.max_modules is not None and registered >= self.max_modules:
                    break
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.writer.flush()
        self.writer.close()
