from contextlib import ExitStack, nullcontext
import logging
import math
from pathlib import Path
import random

import torch

from src.application.use_cases.recognizer.recognizerCTCModel import RecognizerCTCModel
from src.application.use_cases.recognizer.compute_ctc_loss import compute_ctc_loss
from src.infra.data.dataloaders.recognizer_dataloader import RecognizerDataLoader
from src.domain.value_objet.config import RecognizerConfig
from src.infra.modeling.hooks import (
    TensorBoardScalarHook,
    NaNInfGuardHook,
    GradNormHook,
    tqdm_wrap,
)

LOGGER = logging.getLogger(__name__)


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

def _ctc_greedy_decode(logits: torch.Tensor, blank_idx: int = 0) -> list[list[int]]:
    pred = logits.argmax(dim=-1)  # [B, T]
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


def _word_error_distance(pred_text: str, gt_text: str) -> int:
    pred_words = pred_text.split()
    gt_words = gt_text.split()
    return _levenshtein_distance(pred_words, gt_words)

class RecognizerTrainer:
    def __init__(
        self,
        config: RecognizerConfig,
        model: RecognizerCTCModel,
        train_loader: RecognizerDataLoader,
        val_loader: RecognizerDataLoader,
        criterion: torch.nn.CTCLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        aux_loss_weight: float = 0.0,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.aux_loss_weight = aux_loss_weight

    def _validate_training_config(self) -> None:
        cfg = self.config.config_training
        required_sections = ("io", "hooks", "loop", "loss")
        missing = [name for name in required_sections if name not in cfg]
        if missing:
            raise ValueError(f"RecognizerTrainer: missing config sections: {missing}")

        io_cfg = cfg["io"]
        loop_cfg = cfg["loop"]
        loss_cfg = cfg["loss"]
        hooks_cfg = cfg["hooks"]
        optimizer_cfg = cfg.get("optimizer", {})
        if "save_dir" not in io_cfg:
            raise ValueError("RecognizerTrainer: io.save_dir is required")
        if int(loop_cfg["epochs"]) <= 0:
            raise ValueError(f"RecognizerTrainer: loop.epochs must be > 0, got {loop_cfg['epochs']}")
        if float(loss_cfg["aux_loss_weight"]) < 0:
            raise ValueError(
                f"RecognizerTrainer: loss.aux_loss_weight must be >= 0, got {loss_cfg['aux_loss_weight']}"
            )
        if "tensorboard" not in hooks_cfg:
            raise ValueError("RecognizerTrainer: hooks.tensorboard is required")
        grad_accum_steps = int(optimizer_cfg.get("grad_accum_steps", 1))
        if grad_accum_steps <= 0:
            raise ValueError(
                f"RecognizerTrainer: optimizer.grad_accum_steps must be > 0, got {grad_accum_steps}"
            )

    @staticmethod
    def _resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
        amp_dtype_text = amp_dtype.strip().lower()
        if amp_dtype_text == "float16":
            return torch.float16
        if amp_dtype_text == "bfloat16":
            return torch.bfloat16
        raise ValueError(
            f"RecognizerTrainer: unsupported optimizer.amp_dtype='{amp_dtype}'. Expected 'float16' or 'bfloat16'."
        )

    def _build_scheduler(self, optimizer_cfg: dict, epochs: int) -> torch.optim.lr_scheduler.LRScheduler | None:
        scheduler_cfg = dict(optimizer_cfg.get("scheduler", {}))
        if not bool(scheduler_cfg.get("enabled", False)):
            return None
        grad_accum_steps = int(optimizer_cfg.get("grad_accum_steps", 1))
        if grad_accum_steps <= 0:
            return None
        num_batches = len(self.train_loader)
        steps_per_epoch = max(1, math.ceil(num_batches / grad_accum_steps))
        total_steps = max(1, epochs * steps_per_epoch)
        warmup_epochs = float(scheduler_cfg.get("warmup_epochs", 0.0))
        warmup_steps = max(0, int(round(warmup_epochs * steps_per_epoch)))
        min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.1))

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max(1e-8, float(step + 1) / float(warmup_steps))
            if total_steps <= warmup_steps:
                return 1.0
            progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _setup_reproducibility(self) -> int:
        cfg = self.config.config_training
        dataset_cfg = cfg.get("dataset", {})
        loop_cfg = cfg.get("loop", {})
        seed = int(loop_cfg.get("seed", dataset_cfg.get("split_seed", 42)))
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            import numpy as np  # type: ignore

            np.random.seed(seed)
        except Exception:
            pass
        return seed

    def _run_epoch(
            self,
            epoch_idx: int,
            training: bool = True,
            tb_hook: TensorBoardScalarHook | None = None,
            log_prefix: str | None = None,
            log_lr_every_n_steps: int = 0,
            aux_loss_weight: float | None = None,
            grad_accum_steps: int = 1,
            amp_enabled: bool = False,
            amp_dtype: torch.dtype = torch.float16,
            scaler: torch.cuda.amp.GradScaler | None = None,
            scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        ) -> dict[str, float]:
        self.model.train(training)
        loader = self.train_loader if training else self.val_loader
        total_loss = 0.0
        total_ctc_loss = 0.0
        total_aux_loss = 0.0
        total_edit_distance = 0
        total_chars = 0
        total_word_distance = 0
        total_words = 0
        total_seq_match = 0
        total_seqs = 0
        total_pred_tokens = 0
        total_argmax_tokens = 0
        total_non_blank_pred_tokens = 0
        total_gt_chars = 0
        num_steps = 0
        num_batches = len(loader)
        phase_step = 0
        num_epochs = int(self.config.config_training["loop"]["epochs"])
        epoch_num = epoch_idx + 1
        progress = tqdm_wrap(
            loader,
            desc=f"{'Train' if training else 'Val'} Epoch {epoch_num}/{num_epochs}",
            total=num_batches,
            leave=False,
        )

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(progress):
            with torch.set_grad_enabled(training):
                autocast_ctx = (
                    torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype)
                    if self.device.type == "cuda"
                    else nullcontext()
                )
                with autocast_ctx:
                    loss, metrics = compute_ctc_loss(
                        model=self.model,
                        batch=batch,
                        criterion=self.criterion,
                        device=self.device,
                        aux_loss_weight=aux_loss_weight if aux_loss_weight is not None else self.aux_loss_weight,
                    )

                if training:
                    scaled_loss = loss / max(grad_accum_steps, 1)
                    if scaler is not None and scaler.is_enabled():
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    should_step = ((batch_idx + 1) % max(grad_accum_steps, 1) == 0) or ((batch_idx + 1) == num_batches)
                    if should_step:
                        grad_clip = float(self.config.config_training["loop"].get("grad_clip_norm", 0.0))
                        if grad_clip > 0:
                            if scaler is not None and scaler.is_enabled():
                                scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

                        if scaler is not None and scaler.is_enabled():
                            scaler.step(self.optimizer)
                            scaler.update()
                        else:
                            self.optimizer.step()

                        if scheduler is not None:
                            scheduler.step()

                        self.optimizer.zero_grad(set_to_none=True)
                        if tb_hook is not None:
                            tb_hook.step()
                            if log_lr_every_n_steps > 0 and tb_hook.global_step % log_lr_every_n_steps == 0:
                                tb_hook.log_optimizer_lrs(self.optimizer, tag="train/lr")

            total_loss += metrics["loss"]
            total_ctc_loss += metrics["ctc_loss"]
            total_aux_loss += metrics["aux_loss"]
            pred_tokens = _ctc_greedy_decode(metrics["logits"], blank_idx=0)
            pred_texts = [loader.encoder.decode(seq) for seq in pred_tokens]
            gt_texts = batch.texts
            argmax_pred = metrics["logits"].argmax(dim=-1)
            total_argmax_tokens += int(argmax_pred.numel())
            total_non_blank_pred_tokens += int((argmax_pred != 0).sum().item())

            for pred_text, gt_text in zip(pred_texts, gt_texts):
                total_edit_distance += _levenshtein_distance(pred_text, gt_text)
                total_chars += max(len(gt_text), 1)
                total_word_distance += _word_error_distance(pred_text, gt_text)
                total_words += max(len(gt_text.split()), 1)
                total_seq_match += int(pred_text == gt_text)
                total_seqs += 1
                total_gt_chars += len(gt_text)
                total_pred_tokens += len(pred_text)

            num_steps += 1
            phase_step += 1
            running_cer = total_edit_distance / max(total_chars, 1)
            running_wer = total_word_distance / max(total_words, 1)
            running_seq_acc = total_seq_match / max(total_seqs, 1)
            progress.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                ctc=f"{metrics['ctc_loss']:.4f}",
                cer=f"{running_cer:.4f}",
                wer=f"{running_wer:.4f}",
                seq_acc=f"{running_seq_acc:.4f}",
            )
            if tb_hook is not None and log_prefix is not None:
                step_idx = tb_hook.global_step if training else phase_step
                tb_hook.writer.add_scalar(f"{log_prefix}/step_loss", metrics["loss"], step_idx)
                tb_hook.writer.add_scalar(f"{log_prefix}/step_ctc_loss", metrics["ctc_loss"], step_idx)
                tb_hook.writer.add_scalar(f"{log_prefix}/step_aux_loss", metrics["aux_loss"], step_idx)
                tb_hook.writer.add_scalar(f"{log_prefix}/step_cer", running_cer, step_idx)
                tb_hook.writer.add_scalar(f"{log_prefix}/step_wer", running_wer, step_idx)
                tb_hook.writer.add_scalar(f"{log_prefix}/step_seq_acc", running_seq_acc, step_idx)

        if num_steps == 0:
            raise ValueError("_run_epoch: dataloader has no batches")

        avg_loss = total_loss / num_steps
        avg_ctc_loss = total_ctc_loss / num_steps
        avg_aux_loss = total_aux_loss / num_steps
        avg_cer = total_edit_distance / max(total_chars, 1)
        avg_wer = total_word_distance / max(total_words, 1)
        avg_seq_acc = total_seq_match / max(total_seqs, 1)
        pred_non_blank_ratio = total_non_blank_pred_tokens / max(total_argmax_tokens, 1)
        pred_avg_length = total_pred_tokens / max(total_seqs, 1)
        gt_avg_length = total_gt_chars / max(total_seqs, 1)

        if tb_hook is not None and log_prefix is not None:
            tb_hook.writer.add_scalar(f"{log_prefix}/epoch_loss", avg_loss, epoch_num)
            tb_hook.writer.add_scalar(f"{log_prefix}/epoch_ctc_loss", avg_ctc_loss, epoch_num)
            tb_hook.writer.add_scalar(f"{log_prefix}/epoch_aux_loss", avg_aux_loss, epoch_num)
            tb_hook.writer.add_scalar(f"{log_prefix}/epoch_cer", avg_cer, epoch_num)
            tb_hook.writer.add_scalar(f"{log_prefix}/epoch_wer", avg_wer, epoch_num)
            tb_hook.writer.add_scalar(f"{log_prefix}/epoch_seq_acc", avg_seq_acc, epoch_num)
            tb_hook.writer.add_scalar(f"{log_prefix}/epoch_pred_non_blank_ratio", pred_non_blank_ratio, epoch_num)
            tb_hook.writer.add_scalar(f"{log_prefix}/epoch_pred_avg_length", pred_avg_length, epoch_num)
            tb_hook.writer.add_scalar(f"{log_prefix}/epoch_gt_avg_length", gt_avg_length, epoch_num)

        return {
            "loss": avg_loss,
            "ctc_loss": avg_ctc_loss,
            "aux_loss": avg_aux_loss,
            "cer": avg_cer,
            "wer": avg_wer,
            "seq_acc": avg_seq_acc,
            "pred_non_blank_ratio": pred_non_blank_ratio,
            "pred_avg_length": pred_avg_length,
            "gt_avg_length": gt_avg_length,
            "avg_loss": avg_loss,
            "avg_ctc_loss": avg_ctc_loss,
            "avg_aux_loss": avg_aux_loss,
        }

    def _find_resume_checkpoint(
        self,
        save_dir: Path,
        best_ckpt_path: Path,
        last_ckpt_path: Path | None = None,
    ) -> Path | None:
        if last_ckpt_path is not None and last_ckpt_path.is_file():
            return last_ckpt_path
        if best_ckpt_path.is_file():
            return best_ckpt_path
        candidates: list[Path] = []
        for pattern in ("*.pt", "*.pth", "*.ckpt", "*.bin"):
            candidates.extend(save_dir.glob(pattern))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _load_checkpoint(self, checkpoint_path: Path) -> tuple[int, float, float]:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as exc:
            LOGGER.warning("Skip checkpoint %s: load failed (%s)", checkpoint_path, exc)
            return 1, float("inf"), float("inf")

        if not isinstance(checkpoint, dict):
            LOGGER.warning("Skip checkpoint %s: invalid checkpoint format", checkpoint_path)
            return 1, float("inf"), float("inf")

        model_state = checkpoint.get("model_state_dict")
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if model_state is None or optimizer_state is None:
            LOGGER.warning("Skip checkpoint %s: missing model/optimizer states", checkpoint_path)
            return 1, float("inf"), float("inf")

        try:
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)
        except Exception as exc:
            LOGGER.warning("Skip checkpoint %s: incompatible state dict (%s)", checkpoint_path, exc)
            return 1, float("inf"), float("inf")

        if "epoch_idx" in checkpoint:
            start_epoch = int(checkpoint["epoch_idx"]) + 2
        else:
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        best_val_cer = float(checkpoint.get("best_val_cer", float("inf")))
        LOGGER.info("Loaded checkpoint %s (resume from epoch %s)", checkpoint_path, start_epoch)
        return start_epoch, best_val_loss, best_val_cer

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        epoch_idx: int,
        best_val_loss: float,
        best_val_cer: float,
        saved_by: str,
        scaler: torch.cuda.amp.GradScaler | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        encoder = self.train_loader.encoder
        torch.save(
            {
                "version": 1,
                "epoch_idx": epoch_idx,
                "epoch": epoch_idx + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_cer": best_val_cer,
                "encoder_vocab": encoder.vocab,
                "encoder_num_classes": encoder.num_classes,
                "saved_by": saved_by,
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            },
            checkpoint_path,
        )

    def _save_last_checkpoint(
        self,
        checkpoint_path: Path,
        epoch_idx: int,
        best_val_loss: float,
        best_val_cer: float,
        scaler: torch.cuda.amp.GradScaler | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        self._save_checkpoint(
            checkpoint_path=checkpoint_path,
            epoch_idx=epoch_idx,
            best_val_loss=best_val_loss,
            best_val_cer=best_val_cer,
            saved_by="last",
            scaler=scaler,
            scheduler=scheduler,
        )

    def train(self) -> dict[str, object]:
        self._validate_training_config()
        seed = self._setup_reproducibility()
        cfg = self.config.config_training
        io_cfg = cfg["io"]
        hooks_cfg = cfg["hooks"]
        tensortboard_cfg = hooks_cfg.get("tensorboard", {})
        loop_cfg = cfg["loop"]
        loss_cfg = cfg["loss"]
        history: list[dict[str, float]] = []
        best_val_loss = float("inf")
        best_val_cer = float("inf")
        save_dir = Path(io_cfg["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        legacy_best_name = io_cfg.get("best_checkpoint_name")
        best_loss_default = str(legacy_best_name) if legacy_best_name is not None else "best_loss.pt"
        best_loss_ckpt_path = save_dir / str(io_cfg.get("best_loss_checkpoint_name", best_loss_default))
        best_cer_ckpt_path = save_dir / str(io_cfg.get("best_cer_checkpoint_name", "best_cer.pt"))
        last_ckpt_path = save_dir / str(io_cfg.get("last_checkpoint_name", "last_checkpoint.pt"))
        epochs = int(loop_cfg["epochs"])
        aux_loss_weight = float(loss_cfg["aux_loss_weight"])
        start_epoch_idx = 0
        optimizer_cfg = dict(cfg.get("optimizer", {}))
        grad_accum_steps = max(1, int(optimizer_cfg.get("grad_accum_steps", 1)))
        amp_enabled = bool(optimizer_cfg.get("amp_enabled", False)) and self.device.type == "cuda"
        amp_dtype = self._resolve_amp_dtype(str(optimizer_cfg.get("amp_dtype", "float16")))
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        scheduler = self._build_scheduler(optimizer_cfg=optimizer_cfg, epochs=epochs)

        checkpoint_path = self._find_resume_checkpoint(
            save_dir=save_dir,
            best_ckpt_path=best_loss_ckpt_path,
            last_ckpt_path=last_ckpt_path,
        )
        if checkpoint_path is not None:
            start_epoch_num, best_val_loss, best_val_cer = self._load_checkpoint(checkpoint_path)
            start_epoch_idx = max(0, start_epoch_num - 1)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                scaler_state = checkpoint.get("scaler_state_dict")
                if scaler_state is not None:
                    try:
                        scaler.load_state_dict(scaler_state)
                    except Exception as exc:
                        LOGGER.warning("Skip scaler state from checkpoint %s: %s", checkpoint_path, exc)
                if scheduler is not None:
                    scheduler_state = checkpoint.get("scheduler_state_dict")
                    if scheduler_state is not None:
                        try:
                            scheduler.load_state_dict(scheduler_state)
                        except Exception as exc:
                            LOGGER.warning("Skip scheduler state from checkpoint %s: %s", checkpoint_path, exc)
            if start_epoch_idx >= epochs:
                LOGGER.info("Checkpoint epoch already reached target epochs (%s). Nothing to train.", epochs)
                return {
                    "model": self.model,
                    "train_dataset_size": len(self.train_loader.dataset),
                    "val_dataset_size": len(self.val_loader.dataset),
                    "encoder": self.train_loader.encoder,
                    "history": history,
                    "best_checkpoint": str(best_loss_ckpt_path),
                    "best_loss_checkpoint": str(best_loss_ckpt_path),
                    "best_cer_checkpoint": str(best_cer_ckpt_path),
                    "best_val_loss": best_val_loss,
                    "best_val_cer": best_val_cer,
                    "start_epoch": start_epoch_idx + 1,
                    "final_epoch": epochs,
                    "seed": seed,
                    "last_checkpoint": str(last_ckpt_path),
                }

        with ExitStack() as stack:
            tb_hook = stack.enter_context(
                TensorBoardScalarHook(
                    model=self.model,
                    log_dir=io_cfg["tensorboard_dir"],
                    module_names=None,
                    log_every_n_steps=tensortboard_cfg.get("log_every_n_steps", 1),
                    leaf_only=bool(tensortboard_cfg.get("leaf_only", True)),
                    log_histograms_every_n_steps=tensortboard_cfg.get("log_histograms_every_n_steps", 0),
                    flush_every_n_steps=tensortboard_cfg.get("flush_every_n_steps", 10),
                    max_modules=tensortboard_cfg.get("max_modules", 100)    
                )
            )
            if bool(hooks_cfg["nan_guard"]["enabled"]):
                stack.enter_context(
                    NaNInfGuardHook(
                        model=self.model    ,
                        check_inputs=bool(hooks_cfg["nan_guard"]["check_inputs"]),
                        check_outputs=bool(hooks_cfg["nan_guard"]["check_outputs"]),
                        module_names=None,
                    )
                )

            zero_seq_acc_streak = 0
            grad_hook: GradNormHook | None = None
            if bool(hooks_cfg["grad_norm"]["enabled"]):
                grad_hook = stack.enter_context(
                    GradNormHook(
                        model=self.model,
                        module_names=list(hooks_cfg["grad_norm"]["module_names"]),
                    )
                )

            for epoch_idx in range(start_epoch_idx, epochs):
                epoch_num = epoch_idx + 1
                train_metrics = self._run_epoch(
                    epoch_idx=epoch_idx,
                    training=True,
                    tb_hook=tb_hook,
                    log_prefix="train",
                    log_lr_every_n_steps=tensortboard_cfg.get("log_lr_every_n_steps", 0),
                    aux_loss_weight=aux_loss_weight,
                    grad_accum_steps=grad_accum_steps,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    scaler=scaler,
                    scheduler=scheduler,
                )
                val_metrics = self._run_epoch(
                    epoch_idx=epoch_idx,
                    training=False,
                    tb_hook=tb_hook,
                    log_prefix="val"
                )
                history.append({"train": train_metrics, "val": val_metrics})

                LOGGER.info(
                    "Epoch %s/%s - Train Loss: %.4f, Val Loss: %.4f, Val CER: %.4f, Val WER: %.4f, Val Seq Acc: %.4f",
                    epoch_num,
                    epochs,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    val_metrics["cer"],
                    val_metrics["wer"],
                    val_metrics["seq_acc"],
                )
                LOGGER.info(
                    "Epoch %s diagnostics - pred_non_blank_ratio: %.4f, pred_avg_length: %.4f, gt_avg_length: %.4f",
                    epoch_num,
                    val_metrics["pred_non_blank_ratio"],
                    val_metrics["pred_avg_length"],
                    val_metrics["gt_avg_length"],
                )

                tb_hook.writer.add_scalar("train/epoch_loss", train_metrics["loss"], epoch_num)
                tb_hook.writer.add_scalar("train/epoch_ctc_loss", train_metrics["ctc_loss"], epoch_num)
                tb_hook.writer.add_scalar("train/epoch_aux_loss", train_metrics["aux_loss"], epoch_num)
                tb_hook.writer.add_scalar("train/epoch_cer", train_metrics["cer"], epoch_num)
                tb_hook.writer.add_scalar("train/epoch_seq_acc", train_metrics["seq_acc"], epoch_num)
                tb_hook.writer.add_scalar("val/epoch_loss", val_metrics["loss"], epoch_num)
                tb_hook.writer.add_scalar("val/epoch_ctc_loss", val_metrics["ctc_loss"], epoch_num)
                tb_hook.writer.add_scalar("val/epoch_aux_loss", val_metrics["aux_loss"], epoch_num)
                tb_hook.writer.add_scalar("val/epoch_cer", val_metrics["cer"], epoch_num)
                tb_hook.writer.add_scalar("val/epoch_seq_acc", val_metrics["seq_acc"], epoch_num)
                tb_hook.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], epoch_num)

                if grad_hook is not None:
                    for name, value in grad_hook.grad_input_norms.items():
                        tb_hook.writer.add_scalar(f"grad_norm/{name}_input", value, epoch_num)
                    for name, value in grad_hook.grad_output_norms.items():
                        tb_hook.writer.add_scalar(f"grad_norm/{name}_output", value, epoch_num)

                improved_loss = False
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    improved_loss = True
                    self._save_checkpoint(
                        checkpoint_path=best_loss_ckpt_path,
                        epoch_idx=epoch_idx,
                        best_val_loss=best_val_loss,
                        best_val_cer=best_val_cer,
                        saved_by="loss",
                        scaler=scaler,
                        scheduler=scheduler,
                    )
                if val_metrics["cer"] < best_val_cer:
                    best_val_cer = val_metrics["cer"]
                    self._save_checkpoint(
                        checkpoint_path=best_cer_ckpt_path,
                        epoch_idx=epoch_idx,
                        best_val_loss=best_val_loss,
                        best_val_cer=best_val_cer,
                        saved_by="cer",
                        scaler=scaler,
                        scheduler=scheduler,
                    )
                if not improved_loss and val_metrics["seq_acc"] == 0.0:
                    zero_seq_acc_streak += 1
                else:
                    zero_seq_acc_streak = 0
                if zero_seq_acc_streak >= 2 or val_metrics["pred_non_blank_ratio"] < 0.05:
                    LOGGER.warning(
                        "CTC blank collapse suspected at epoch %s: seq_acc=%.4f, pred_non_blank_ratio=%.4f",
                        epoch_num,
                        val_metrics["seq_acc"],
                        val_metrics["pred_non_blank_ratio"],
                    )
                self._save_last_checkpoint(
                    checkpoint_path=last_ckpt_path,
                    epoch_idx=epoch_idx,
                    best_val_loss=best_val_loss,
                    best_val_cer=best_val_cer,
                    scaler=scaler,
                    scheduler=scheduler,
                )

        return {
            "model": self.model,
            "train_dataset_size": len(self.train_loader.dataset),
            "val_dataset_size": len(self.val_loader.dataset),
            "encoder": self.train_loader.encoder,
            "history": history,
            "best_checkpoint": str(best_loss_ckpt_path),
            "best_loss_checkpoint": str(best_loss_ckpt_path),
            "best_cer_checkpoint": str(best_cer_ckpt_path),
            "best_val_loss": best_val_loss,
            "best_val_cer": best_val_cer,
            "start_epoch": start_epoch_idx + 1,
            "final_epoch": epochs,
            "seed": seed,
            "last_checkpoint": str(last_ckpt_path),
        }
