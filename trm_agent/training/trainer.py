"""TRM Trainer with Deep Supervision.

Implements the training loop with:
- Deep supervision (N_sup steps)
- Adaptive Computational Time (ACT) for early stopping
- Exponential Moving Average (EMA)
- Learning rate warmup
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from trm_agent.data import TRMTokenizer

from rich.console import Console
from rich.table import Table
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from trm_agent.models.config import TRMConfig
from trm_agent.models.ema import EMA
from trm_agent.models.trm import TRMForToolCalling
from trm_agent.training.losses import DeepSupervisionLoss
from trm_agent.utils import get_logger, is_main_process, gather_metrics

logger = get_logger(__name__)
console = Console()


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0

    # Scheduler
    warmup_steps: int = 2000
    num_epochs: int = 100

    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # Memory optimization
    use_amp: bool = True  # Automatic Mixed Precision (fp16/bf16)
    amp_dtype: str = "float16"  # "float16" or "bfloat16"

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999

    # ACT
    use_act: bool = True

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    log_sample_interval: int = 0  # Log sample predictions every N steps (0 = disabled)

    # Paths
    output_dir: str = "outputs"


def get_cosine_warmup_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create linear warmup + cosine decay scheduler.

    Linearly increases LR from 0 to target during warmup,
    then decays with cosine schedule to min_lr_ratio * initial_lr.

    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of initial (default: 0.1 = 10%)
    """
    import math

    def lr_lambda(step: int) -> float:
        # Warmup phase
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        # Cosine decay phase
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Decay from 1.0 to min_lr_ratio
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


class TRMTrainer:
    """Trainer for TRM with deep supervision.

    Implements the paper's training algorithm:
    1. For each batch, iterate through N_sup supervision steps
    2. At each step, compute loss and update with gradients
    3. Use ACT to early stop when model is confident
    4. Apply EMA for stability
    """

    def __init__(
        self,
        model: TRMForToolCalling,
        config: TRMConfig,
        training_config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        tool_names: Optional[list[str]] = None,
        slot_fields: Optional[list[str]] = None,
        tokenizer: Optional["TRMTokenizer"] = None,
    ):
        """Initialize trainer.

        Args:
            model: TRM model
            config: Model configuration
            training_config: Training configuration
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader (optional)
            device: Device to use
            tool_names: List of tool names for logging
            slot_fields: List of slot field names for logging
            tokenizer: Tokenizer for decoding spans (optional)
        """
        self.model = model
        self.config = config
        self.training_config = training_config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tool_names = tool_names or []
        self.slot_fields = slot_fields or config.slot_fields
        self.tokenizer = tokenizer

        # Check if model is wrapped with DDP
        self.is_ddp = hasattr(model, "module")
        self.raw_model = model.module if self.is_ddp else model

        # Device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Don't move if already on device (DDP case)
        if not self.is_ddp:
            self.model.to(self.device)

        # Loss function
        self.loss_fn = DeepSupervisionLoss(config)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            betas=(training_config.beta1, training_config.beta2),
            weight_decay=training_config.weight_decay,
        )

        # Scheduler (cosine decay after warmup)
        total_steps = (
            len(train_dataloader)
            * training_config.num_epochs
            // training_config.gradient_accumulation_steps
        )
        self.scheduler = get_cosine_warmup_scheduler(
            self.optimizer,
            training_config.warmup_steps,
            total_steps,
        )

        # EMA (use raw model, not DDP wrapper)
        self.ema = None
        if training_config.use_ema:
            self.ema = EMA(self.raw_model, decay=training_config.ema_decay)

        # AMP (Automatic Mixed Precision)
        self.use_amp = training_config.use_amp and self.device.type == "cuda"
        if self.use_amp:
            amp_dtype = training_config.amp_dtype.lower()
            if amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler("cuda")
            logger.info(f"Using AMP with dtype={self.amp_dtype}")
        else:
            self.amp_dtype = torch.float32
            self.scaler = None

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Output directory (only create on main process)
        self.output_dir = Path(training_config.output_dir)
        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single training step with deep supervision.

        Args:
            batch: Batch of training samples

        Returns:
            Dictionary of loss values
        """
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass with deep supervision (with AMP if enabled)
        # Call forward() directly through DDP wrapper for proper gradient sync
        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
            all_outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                role_ids=batch["role_ids"],
                return_all_steps=True,  # Enable deep supervision
            )

            # Compute loss over all supervision steps
            losses = self.loss_fn(
                all_outputs,
                batch["decision_labels"],
                batch["tool_name_labels"],
                batch["slot_presence_labels"],
                batch["slot_start_labels"],
                batch["slot_end_labels"],
                batch["arg_start_labels"],
                batch["arg_end_labels"],
            )

        # Backward pass
        loss = losses["total_loss"] / self.training_config.gradient_accumulation_steps

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {k: v.item() for k, v in losses.items()}

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of average loss values
        """
        self.model.train()
        total_losses = {}
        num_batches = 0

        # Set epoch for distributed sampler (ensures different shuffling each epoch)
        if hasattr(self.train_dataloader.sampler, "set_epoch"):
            self.train_dataloader.sampler.set_epoch(self.epoch)

        # Calculate total optimizer steps per epoch
        accum_steps = self.training_config.gradient_accumulation_steps
        total_batches = len(self.train_dataloader)
        total_opt_steps = total_batches // accum_steps

        progress_bar = tqdm(
            total=total_opt_steps,
            desc=f"Epoch {self.epoch + 1}/{self.training_config.num_epochs}",
            disable=not is_main_process(),
        )

        for step, batch in enumerate(self.train_dataloader):
            # Training step
            losses = self.train_step(batch)

            # Accumulate losses
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v
            num_batches += 1

            # Gradient accumulation
            if (step + 1) % accum_steps == 0:
                if self.scaler is not None:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)

                # Clip gradients
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.max_grad_norm,
                )

                if self.scaler is not None:
                    # Update weights with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Update weights
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                # Update EMA
                if self.ema is not None:
                    self.ema.update()

                self.global_step += 1

                # Update progress bar
                progress_bar.update(1)

                # Update postfix every 10 optimizer steps
                if self.global_step % 10 == 0:
                    avg_loss = total_losses.get("total_loss", 0) / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        lr=f"{lr:.2e}",
                    )

                # Evaluation
                if (
                    self.eval_dataloader is not None
                    and self.global_step % self.training_config.eval_interval == 0
                ):
                    eval_metrics = self.evaluate()
                    logger.info(f"Eval metrics: {eval_metrics}")

                # Log sample predictions
                if (
                    self.training_config.log_sample_interval > 0
                    and self.eval_dataloader is not None
                    and self.global_step % self.training_config.log_sample_interval == 0
                ):
                    self._log_sample_predictions()

                # Save checkpoint
                if self.global_step % self.training_config.save_interval == 0:
                    self.save_checkpoint()

        progress_bar.close()

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate model on validation set with comprehensive metrics.

        Returns:
            Dictionary of evaluation metrics including:
            - Decision accuracy and confusion matrix
            - Tool name accuracy (for tool_call samples)
            - Slot presence accuracy (per slot and overall)
        """
        if self.eval_dataloader is None:
            return {}

        self.model.eval()

        # Use EMA weights for evaluation
        if self.ema is not None:
            self.ema.apply_shadow()

        # Accumulators for metrics
        # Decision confusion matrix: [TP, FP, FN, TN] for tool_call as positive
        decision_tp = 0  # Predicted tool_call, actual tool_call
        decision_fp = 0  # Predicted tool_call, actual direct_answer
        decision_fn = 0  # Predicted direct_answer, actual tool_call
        decision_tn = 0  # Predicted direct_answer, actual direct_answer

        # Tool name accuracy (only for correct tool_call predictions)
        tool_correct = 0
        tool_total = 0
        per_tool_correct = [0] * len(self.tool_names)
        per_tool_total = [0] * len(self.tool_names)

        # Slot presence accuracy
        slot_correct = 0
        slot_total = 0
        per_slot_correct = [0] * len(self.slot_fields)
        per_slot_total = [0] * len(self.slot_fields)

        # Span extraction metrics
        slot_span_correct = 0  # Both start and end match
        slot_span_total = 0  # Total slots with valid labels
        arg_span_correct = 0  # Both start and end match
        arg_span_total = 0  # Total args with valid labels

        total_samples = 0

        logger.info("Starting evaluation...")

        eval_progress = tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            disable=not is_main_process(),
        )

        for batch in eval_progress:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            # Use raw_model for inference (no gradient sync needed in eval)
            outputs = self.raw_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                role_ids=batch["role_ids"],
                return_all_steps=False,  # Inference mode
            )

            batch_size = batch["decision_labels"].size(0)
            total_samples += batch_size

            # Decision predictions
            decision_pred = (torch.sigmoid(outputs.decision_logits) > 0.5).long().view(-1)
            decision_true = batch["decision_labels"].long()

            # Update confusion matrix
            for pred, true in zip(decision_pred, decision_true):
                if pred == 1 and true == 1:
                    decision_tp += 1
                elif pred == 1 and true == 0:
                    decision_fp += 1
                elif pred == 0 and true == 1:
                    decision_fn += 1
                else:
                    decision_tn += 1

            # Tool name accuracy (only for true tool_call samples)
            tool_mask = batch["tool_name_labels"] >= 0
            if tool_mask.any():
                tool_pred = outputs.tool_logits.argmax(dim=-1)
                tool_true = batch["tool_name_labels"]
                tool_correct += (tool_pred[tool_mask] == tool_true[tool_mask]).sum().item()
                tool_total += tool_mask.sum().item()

                # Per-tool accuracy
                for i in range(len(self.tool_names)):
                    tool_i_mask = tool_true == i
                    if tool_i_mask.any():
                        per_tool_correct[i] += (tool_pred[tool_i_mask] == i).sum().item()
                        per_tool_total[i] += tool_i_mask.sum().item()

            # Slot presence accuracy
            slot_pred = (torch.sigmoid(outputs.slot_presence_logits) > 0.5).float()
            slot_true = batch["slot_presence_labels"]

            slot_matches = (slot_pred == slot_true).float()
            slot_correct += slot_matches.sum().item()
            slot_total += slot_true.numel()

            # Per-slot accuracy
            for i in range(len(self.slot_fields)):
                per_slot_correct[i] += slot_matches[:, i].sum().item()
                per_slot_total[i] += batch_size

            # Slot span extraction accuracy (only for slots with valid labels)
            slot_start_labels = batch["slot_start_labels"]  # [batch, num_slots]
            slot_end_labels = batch["slot_end_labels"]  # [batch, num_slots]
            slot_start_pred = outputs.slot_start_logits.argmax(dim=1)  # [batch, num_slots]
            slot_end_pred = outputs.slot_end_logits.argmax(dim=1)  # [batch, num_slots]

            # Valid slots have start_label >= 0
            slot_valid_mask = slot_start_labels >= 0  # [batch, num_slots]
            if slot_valid_mask.any():
                slot_start_match = slot_start_pred == slot_start_labels
                slot_end_match = slot_end_pred == slot_end_labels
                slot_span_match = slot_start_match & slot_end_match & slot_valid_mask
                slot_span_correct += slot_span_match.sum().item()
                slot_span_total += slot_valid_mask.sum().item()

            # Argument span extraction accuracy (only for tool_call samples with valid labels)
            if tool_mask.any():
                arg_start_labels = batch["arg_start_labels"][tool_mask]  # [num_tool_calls, max_args]
                arg_end_labels = batch["arg_end_labels"][tool_mask]
                arg_start_pred = outputs.arg_start_logits[tool_mask].argmax(dim=1)  # [num_tool_calls, max_args]
                arg_end_pred = outputs.arg_end_logits[tool_mask].argmax(dim=1)

                # Valid args have start_label >= 0
                arg_valid_mask = arg_start_labels >= 0
                if arg_valid_mask.any():
                    arg_start_match = arg_start_pred == arg_start_labels
                    arg_end_match = arg_end_pred == arg_end_labels
                    arg_span_match = arg_start_match & arg_end_match & arg_valid_mask
                    arg_span_correct += arg_span_match.sum().item()
                    arg_span_total += arg_valid_mask.sum().item()

        # Restore original weights
        if self.ema is not None:
            self.ema.restore()

        # Aggregate metrics across all GPUs (for distributed training)
        raw_metrics = {
            "total_samples": total_samples,
            "decision_tp": decision_tp,
            "decision_fp": decision_fp,
            "decision_fn": decision_fn,
            "decision_tn": decision_tn,
            "tool_correct": tool_correct,
            "tool_total": tool_total,
            "slot_correct": slot_correct,
            "slot_total": slot_total,
            "slot_span_correct": slot_span_correct,
            "slot_span_total": slot_span_total,
            "arg_span_correct": arg_span_correct,
            "arg_span_total": arg_span_total,
        }
        # Add per-slot metrics
        for i, field in enumerate(self.slot_fields):
            raw_metrics[f"per_slot_correct_{i}"] = per_slot_correct[i]
            raw_metrics[f"per_slot_total_{i}"] = per_slot_total[i]

        # Add per-tool metrics
        for i in range(len(self.tool_names)):
            raw_metrics[f"per_tool_correct_{i}"] = per_tool_correct[i]
            raw_metrics[f"per_tool_total_{i}"] = per_tool_total[i]

        # Reduce across GPUs
        agg = gather_metrics(raw_metrics, self.device)

        # Extract aggregated values
        total_samples = int(agg["total_samples"])
        decision_tp = int(agg["decision_tp"])
        decision_fp = int(agg["decision_fp"])
        decision_fn = int(agg["decision_fn"])
        decision_tn = int(agg["decision_tn"])
        tool_correct = agg["tool_correct"]
        tool_total = int(agg["tool_total"])
        slot_correct = agg["slot_correct"]
        slot_total = int(agg["slot_total"])
        slot_span_correct = agg["slot_span_correct"]
        slot_span_total = int(agg["slot_span_total"])
        arg_span_correct = agg["arg_span_correct"]
        arg_span_total = int(agg["arg_span_total"])

        for i in range(len(self.slot_fields)):
            per_slot_correct[i] = agg[f"per_slot_correct_{i}"]
            per_slot_total[i] = int(agg[f"per_slot_total_{i}"])

        for i in range(len(self.tool_names)):
            per_tool_correct[i] = agg[f"per_tool_correct_{i}"]
            per_tool_total[i] = int(agg[f"per_tool_total_{i}"])

        # Compute metrics
        metrics = {"eval_samples": total_samples}

        # Decision metrics
        decision_total = decision_tp + decision_fp + decision_fn + decision_tn
        metrics["decision_accuracy"] = (decision_tp + decision_tn) / decision_total if decision_total > 0 else 0.0

        # Precision, Recall, F1 for tool_call
        precision = decision_tp / (decision_tp + decision_fp) if (decision_tp + decision_fp) > 0 else 0.0
        recall = decision_tp / (decision_tp + decision_fn) if (decision_tp + decision_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics["decision_precision"] = precision
        metrics["decision_recall"] = recall
        metrics["decision_f1"] = f1

        # Tool name accuracy
        metrics["tool_accuracy"] = tool_correct / tool_total if tool_total > 0 else 0.0
        metrics["tool_samples"] = tool_total

        # Slot accuracy
        metrics["slot_accuracy"] = slot_correct / slot_total if slot_total > 0 else 0.0

        # Span extraction accuracy
        metrics["slot_span_exact_match"] = slot_span_correct / slot_span_total if slot_span_total > 0 else 0.0
        metrics["slot_span_samples"] = slot_span_total
        metrics["arg_span_exact_match"] = arg_span_correct / arg_span_total if arg_span_total > 0 else 0.0
        metrics["arg_span_samples"] = arg_span_total

        # Per-slot accuracy
        for i, field in enumerate(self.slot_fields):
            acc = per_slot_correct[i] / per_slot_total[i] if per_slot_total[i] > 0 else 0.0
            metrics[f"slot_{field}_acc"] = acc

        # Per-tool accuracy
        for i, tool_name in enumerate(self.tool_names):
            acc = per_tool_correct[i] / per_tool_total[i] if per_tool_total[i] > 0 else 0.0
            metrics[f"tool_{tool_name}_acc"] = acc
            metrics[f"tool_{tool_name}_samples"] = per_tool_total[i]

        # Log confusion matrix (only on main process, with aggregated values)
        self._log_confusion_matrix(decision_tp, decision_fp, decision_fn, decision_tn)

        return metrics

    def _log_confusion_matrix(self, tp: int, fp: int, fn: int, tn: int):
        """Log confusion matrix for decision classification using rich table."""
        if not is_main_process():
            return

        total = tp + fp + fn + tn
        if total == 0:
            return

        table = Table(title="Decision Confusion Matrix", show_header=True)
        table.add_column("Actual \\ Predicted", style="bold")
        table.add_column("tool_call", justify="center", style="cyan")
        table.add_column("direct_answer", justify="center", style="magenta")

        table.add_row("tool_call", f"[green]{tp}[/green] (TP)", f"[red]{fn}[/red] (FN)")
        table.add_row("direct_answer", f"[red]{fp}[/red] (FP)", f"[green]{tn}[/green] (TN)")

        console.print(table)

    def _decode_span(self, input_ids: torch.Tensor, start: int, end: int) -> str:
        """Decode a span from input_ids using tokenizer."""
        if self.tokenizer is None or start < 0 or end < start:
            return ""
        if end >= len(input_ids):
            end = len(input_ids) - 1
        span_ids = input_ids[start:end + 1].tolist()
        return self.tokenizer.decode(span_ids, skip_special_tokens=True)

    def _get_arg_idx_to_name(self, dataset, tool_name: str) -> dict[int, str]:
        """Get argument index to name mapping for a tool."""
        if tool_name is None:
            return {}

        # Unwrap Subset if needed
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset

        if not hasattr(dataset, 'arg_name_to_idx'):
            return {}

        arg_name_to_idx = dataset.arg_name_to_idx
        if tool_name not in arg_name_to_idx:
            return {}

        # Reverse the mapping: name->idx becomes idx->name
        return {v: k for k, v in arg_name_to_idx[tool_name].items()}

    @torch.no_grad()
    def _log_sample_predictions(self):
        """Log random tool_call and direct_answer predictions from eval dataset."""
        if not is_main_process() or self.eval_dataloader is None:
            return

        import random

        self.model.eval()

        # Use EMA weights if available
        if self.ema is not None:
            self.ema.apply_shadow()

        try:
            # Collect all samples by type
            tool_call_samples = []
            direct_answer_samples = []

            for batch in self.eval_dataloader:
                batch_device = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = batch["decision_labels"].size(0)

                for i in range(batch_size):
                    is_tool_call = batch["tool_name_labels"][i].item() >= 0
                    if is_tool_call:
                        tool_call_samples.append((batch, i))
                    else:
                        direct_answer_samples.append((batch, i))

            # Randomly select one of each type
            samples_to_log = []
            if tool_call_samples:
                samples_to_log.append(("tool_call", random.choice(tool_call_samples)))
            if direct_answer_samples:
                samples_to_log.append(("direct_answer", random.choice(direct_answer_samples)))

            console.print(f"\n[bold blue]═══ Sample Predictions (Step {self.global_step}) ═══[/bold blue]\n")

            for sample_type, (batch, sample_idx) in samples_to_log:
                batch_device = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.raw_model(
                    input_ids=batch_device["input_ids"],
                    attention_mask=batch_device["attention_mask"],
                    role_ids=batch_device["role_ids"],
                    return_all_steps=False,
                )

                self._log_single_sample(
                    outputs, batch, sample_idx, sample_type
                )

            console.print("[bold blue]═══ End Sample Predictions ═══[/bold blue]\n")

        finally:
            if self.ema is not None:
                self.ema.restore()

    def _log_single_sample(
        self,
        outputs,
        batch: dict[str, torch.Tensor],
        sample_idx: int,
        sample_type: str,
    ):
        """Log a single sample prediction."""
        input_ids = batch["input_ids"][sample_idx].cpu()

        # Extract predictions
        decision_prob = torch.sigmoid(outputs.decision_logits[sample_idx]).item()
        decision_pred = "tool_call" if decision_prob > 0.5 else "direct_answer"
        decision_true = "tool_call" if sample_type == "tool_call" else "direct_answer"

        # Build main table
        table = Table(title=f"Sample: {sample_type}", show_header=True)
        table.add_column("Field", style="bold")
        table.add_column("Predicted", style="cyan")
        table.add_column("Ground Truth", style="green")

        # Decision
        dec_match = "[green]" if decision_pred == decision_true else "[red]"
        table.add_row("Decision", f"{dec_match}{decision_pred}[/] ({decision_prob:.3f})", decision_true)

        # Tool name (for tool_call samples)
        tool_true_idx = batch["tool_name_labels"][sample_idx].item()
        tool_true_name = None
        if tool_true_idx >= 0:
            tool_pred_idx = outputs.tool_logits[sample_idx].argmax().item()
            tool_pred_name = self.tool_names[tool_pred_idx] if tool_pred_idx < len(self.tool_names) else f"tool_{tool_pred_idx}"
            tool_true_name = self.tool_names[tool_true_idx] if tool_true_idx < len(self.tool_names) else f"tool_{tool_true_idx}"
            tool_match = "[green]" if tool_pred_name == tool_true_name else "[red]"
            table.add_row("Tool Name", f"{tool_match}{tool_pred_name}[/]", tool_true_name)
        else:
            table.add_row("Tool Name", "-", "-")

        console.print(table)

        # Slot presence and spans
        slot_probs = torch.sigmoid(outputs.slot_presence_logits[sample_idx])
        slot_preds = (slot_probs > 0.5).tolist()
        slot_true = batch["slot_presence_labels"][sample_idx].tolist()
        slot_start_pred = outputs.slot_start_logits[sample_idx].argmax(dim=0).tolist()
        slot_end_pred = outputs.slot_end_logits[sample_idx].argmax(dim=0).tolist()
        slot_start_true = batch["slot_start_labels"][sample_idx].tolist()
        slot_end_true = batch["slot_end_labels"][sample_idx].tolist()

        slot_table = Table(title="Slot Values", show_header=True)
        slot_table.add_column("Slot", style="bold")
        slot_table.add_column("Present", justify="center")
        slot_table.add_column("Predicted Value", style="cyan")
        slot_table.add_column("True Value", style="green")

        for i, field in enumerate(self.slot_fields):
            pred_present = slot_preds[i] if i < len(slot_preds) else False
            true_present = slot_true[i] if i < len(slot_true) else 0

            if not pred_present and not true_present:
                continue

            present_str = f"P:{1 if pred_present else 0} / T:{int(true_present)}"

            # Decode predicted span
            if pred_present and i < len(slot_start_pred):
                pred_text = self._decode_span(input_ids, slot_start_pred[i], slot_end_pred[i])
                pred_span = f"[{slot_start_pred[i]}:{slot_end_pred[i]}]"
                pred_val = f"{pred_text} {pred_span}" if pred_text else f"? {pred_span}"
            else:
                pred_val = "-"

            # Decode ground truth span
            true_start = slot_start_true[i] if i < len(slot_start_true) else -1
            true_end = slot_end_true[i] if i < len(slot_end_true) else -1
            if true_start >= 0:
                true_text = self._decode_span(input_ids, true_start, true_end)
                true_span = f"[{true_start}:{true_end}]"
                true_val = f"{true_text} {true_span}" if true_text else f"? {true_span}"
            else:
                true_val = "-"

            # Color based on match
            if pred_present and true_start >= 0:
                if slot_start_pred[i] == true_start and slot_end_pred[i] == true_end:
                    pred_val = f"[green]{pred_val}[/]"
                else:
                    pred_val = f"[red]{pred_val}[/]"

            slot_table.add_row(field, present_str, pred_val, true_val)

        if slot_table.row_count > 0:
            console.print(slot_table)

        # Argument spans (only for tool_call samples)
        if sample_type == "tool_call" and tool_true_name:
            arg_start_pred = outputs.arg_start_logits[sample_idx].argmax(dim=0).tolist()
            arg_end_pred = outputs.arg_end_logits[sample_idx].argmax(dim=0).tolist()
            arg_start_true = batch["arg_start_labels"][sample_idx].tolist()
            arg_end_true = batch["arg_end_labels"][sample_idx].tolist()

            arg_idx_to_name = self._get_arg_idx_to_name(self.eval_dataloader.dataset, tool_true_name)

            arg_table = Table(title=f"Argument Values (tool: {tool_true_name})", show_header=True)
            arg_table.add_column("Arg Name", style="bold")
            arg_table.add_column("Predicted Value", style="cyan")
            arg_table.add_column("True Value", style="green")

            for i in range(len(arg_start_true)):
                true_start = arg_start_true[i]
                true_end = arg_end_true[i]

                if true_start < 0:
                    continue

                arg_name = arg_idx_to_name.get(i, f"arg_{i}")

                pred_text = self._decode_span(input_ids, arg_start_pred[i], arg_end_pred[i])
                pred_span = f"[{arg_start_pred[i]}:{arg_end_pred[i]}]"
                pred_val = f"{pred_text} {pred_span}" if pred_text else f"? {pred_span}"

                true_text = self._decode_span(input_ids, true_start, true_end)
                true_span = f"[{true_start}:{true_end}]"
                true_val = f"{true_text} {true_span}" if true_text else f"? {true_span}"

                if arg_start_pred[i] == true_start and arg_end_pred[i] == true_end:
                    pred_val = f"[green]{pred_val}[/]"
                else:
                    pred_val = f"[red]{pred_val}[/]"

                arg_table.add_row(arg_name, pred_val, true_val)

            if arg_table.row_count > 0:
                console.print(arg_table)

        console.print("")

    def _log_data_samples(self):
        """Log one tool_call and one direct_answer sample for debugging."""
        if not is_main_process():
            return

        console.print("\n[bold blue]═══ Data Sample Check ═══[/bold blue]\n")

        tool_call_found = False
        direct_answer_found = False

        for batch in self.train_dataloader:
            batch_size = batch["decision_labels"].size(0)

            for i in range(batch_size):
                decision = batch["decision_labels"][i].item()
                is_tool_call = decision == 1

                # Skip if we already logged this type
                if is_tool_call and tool_call_found:
                    continue
                if not is_tool_call and direct_answer_found:
                    continue

                sample_type = "tool_call" if is_tool_call else "direct_answer"
                table = Table(title=f"Sample: {sample_type}", show_header=True)
                table.add_column("Field", style="bold")
                table.add_column("Value")

                # Basic info
                input_ids = batch["input_ids"][i]
                seq_len = (batch["attention_mask"][i] == 1).sum().item()
                table.add_row("Sequence Length", str(seq_len))
                table.add_row("Decision Label", f"{decision} ({sample_type})")

                # Decode first 100 tokens of input
                if self.tokenizer:
                    text = self.tokenizer.decode(input_ids[:min(100, seq_len)].tolist())
                    text = text[:200] + "..." if len(text) > 200 else text
                    table.add_row("Input (first 100 tokens)", text)

                # Tool info
                tool_label = batch["tool_name_labels"][i].item()
                if tool_label >= 0:
                    tool_name = self.tool_names[tool_label] if tool_label < len(self.tool_names) else f"tool_{tool_label}"
                    table.add_row("Tool Name", tool_name)
                else:
                    table.add_row("Tool Name", "-")

                # Slot presence
                slot_presence = batch["slot_presence_labels"][i].tolist()
                slot_str = ", ".join(
                    f"{field}={int(slot_presence[j])}"
                    for j, field in enumerate(self.slot_fields)
                    if j < len(slot_presence)
                )
                table.add_row("Slot Presence", slot_str)

                console.print(table)

                # Slot spans table
                slot_start = batch["slot_start_labels"][i].tolist()
                slot_end = batch["slot_end_labels"][i].tolist()

                span_table = Table(title=f"Slot Spans ({sample_type})", show_header=True)
                span_table.add_column("Slot", style="bold")
                span_table.add_column("Present")
                span_table.add_column("Start")
                span_table.add_column("End")
                span_table.add_column("Decoded Value")

                for j, field in enumerate(self.slot_fields):
                    if j >= len(slot_start):
                        break
                    present = int(slot_presence[j]) if j < len(slot_presence) else 0
                    start = slot_start[j]
                    end = slot_end[j]

                    if start >= 0 and self.tokenizer:
                        decoded = self._decode_span(input_ids, start, end)
                    else:
                        decoded = "-"

                    span_table.add_row(
                        field,
                        str(present),
                        str(start) if start >= 0 else "-",
                        str(end) if end >= 0 else "-",
                        decoded,
                    )

                console.print(span_table)

                # Argument spans (only for tool_call)
                if is_tool_call and tool_label >= 0:
                    arg_start = batch["arg_start_labels"][i].tolist()
                    arg_end = batch["arg_end_labels"][i].tolist()

                    # Get argument name mapping for this tool
                    tool_name = self.tool_names[tool_label] if tool_label < len(self.tool_names) else None
                    arg_idx_to_name = self._get_arg_idx_to_name(self.train_dataloader.dataset, tool_name)

                    arg_table = Table(title=f"Argument Spans (tool: {tool_name})", show_header=True)
                    arg_table.add_column("Arg Name", style="bold")
                    arg_table.add_column("Index")
                    arg_table.add_column("Start")
                    arg_table.add_column("End")
                    arg_table.add_column("Decoded Value")

                    has_args = False
                    for j in range(len(arg_start)):
                        if arg_start[j] >= 0:
                            has_args = True
                            arg_name = arg_idx_to_name.get(j, f"arg_{j}")
                            if self.tokenizer:
                                decoded = self._decode_span(input_ids, arg_start[j], arg_end[j])
                            else:
                                decoded = "-"
                            arg_table.add_row(arg_name, str(j), str(arg_start[j]), str(arg_end[j]), decoded)

                    if has_args:
                        console.print(arg_table)
                    else:
                        console.print("[dim]No argument spans found[/dim]")

                console.print("")

                # Mark as found
                if is_tool_call:
                    tool_call_found = True
                else:
                    direct_answer_found = True

                # Stop if we found both
                if tool_call_found and direct_answer_found:
                    break

            if tool_call_found and direct_answer_found:
                break

        console.print("[bold blue]═══ End Data Sample Check ═══[/bold blue]\n")

    def train(self):
        """Full training loop."""
        # Log sample data for debugging
        self._log_data_samples()

        logger.info(f"Starting training for {self.training_config.num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(self.train_dataloader.dataset)}")
        if self.eval_dataloader is not None:
            logger.info(f"Validation samples: {len(self.eval_dataloader.dataset)}")

        for epoch in range(self.training_config.num_epochs):
            self.epoch = epoch

            # Train epoch
            avg_losses = self.train_epoch()
            logger.info(f"Epoch {epoch + 1} complete. Avg loss: {avg_losses['total_loss']:.4f}")

            # Evaluate
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                self._log_eval_metrics(epoch + 1, eval_metrics)

            # Save checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")

        # Save final model
        self.save_model()
        logger.info("Training complete!")

    def _log_eval_metrics(self, epoch: int, metrics: dict[str, float]):
        """Log evaluation metrics using rich tables."""
        if not is_main_process():
            return

        # Main metrics table
        main_table = Table(
            title=f"Epoch {epoch} Evaluation Results",
            show_header=True,
            header_style="bold blue",
        )
        main_table.add_column("Metric", style="bold")
        main_table.add_column("Value", justify="right")

        # Decision metrics
        main_table.add_row("Decision Accuracy", f"{metrics.get('decision_accuracy', 0):.4f}")
        main_table.add_row("Decision Precision", f"{metrics.get('decision_precision', 0):.4f}")
        main_table.add_row("Decision Recall", f"{metrics.get('decision_recall', 0):.4f}")
        main_table.add_row("Decision F1", f"[bold cyan]{metrics.get('decision_f1', 0):.4f}[/bold cyan]")
        main_table.add_row("", "")  # Empty row as separator

        # Tool metrics
        tool_acc = metrics.get('tool_accuracy', 0)
        tool_samples = int(metrics.get('tool_samples', 0))
        main_table.add_row("Tool Accuracy", f"{tool_acc:.4f} ({tool_samples} samples)")
        main_table.add_row("", "")  # Empty row as separator

        # Overall slot accuracy
        main_table.add_row("Slot Accuracy (Overall)", f"{metrics.get('slot_accuracy', 0):.4f}")
        main_table.add_row("", "")  # Empty row as separator

        # Span extraction metrics
        slot_span_em = metrics.get('slot_span_exact_match', 0)
        slot_span_samples = int(metrics.get('slot_span_samples', 0))
        arg_span_em = metrics.get('arg_span_exact_match', 0)
        arg_span_samples = int(metrics.get('arg_span_samples', 0))
        main_table.add_row("Slot Span EM", f"{slot_span_em:.4f} ({slot_span_samples} spans)")
        main_table.add_row("Arg Span EM", f"{arg_span_em:.4f} ({arg_span_samples} spans)")

        console.print(main_table)

        # Per-slot accuracy table
        slot_table = Table(
            title="Per-Slot Accuracy",
            show_header=True,
            header_style="bold green",
        )
        slot_table.add_column("Slot Field", style="bold")
        slot_table.add_column("Accuracy", justify="right")

        for field in self.slot_fields:
            acc = metrics.get(f"slot_{field}_acc", 0)
            # Color code based on accuracy
            if acc >= 0.9:
                acc_str = f"[green]{acc:.4f}[/green]"
            elif acc >= 0.7:
                acc_str = f"[yellow]{acc:.4f}[/yellow]"
            else:
                acc_str = f"[red]{acc:.4f}[/red]"
            slot_table.add_row(field, acc_str)

        console.print(slot_table)

        # Per-tool accuracy table
        if self.tool_names:
            tool_table = Table(
                title="Per-Tool Accuracy",
                show_header=True,
                header_style="bold magenta",
            )
            tool_table.add_column("Tool Name", style="bold")
            tool_table.add_column("Accuracy", justify="right")
            tool_table.add_column("Samples", justify="right")

            for tool_name in self.tool_names:
                acc = metrics.get(f"tool_{tool_name}_acc", 0)
                samples = int(metrics.get(f"tool_{tool_name}_samples", 0))
                # Color code based on accuracy
                if acc >= 0.9:
                    acc_str = f"[green]{acc:.4f}[/green]"
                elif acc >= 0.7:
                    acc_str = f"[yellow]{acc:.4f}[/yellow]"
                else:
                    acc_str = f"[red]{acc:.4f}[/red]"
                tool_table.add_row(tool_name, acc_str, str(samples))

            console.print(tool_table)

    def save_checkpoint(self, name: Optional[str] = None):
        """Save training checkpoint to output directory.

        Saves:
        - checkpoint_{name}.pt: Model weights, optimizer state, training state
        - config.yaml: Model configuration for inference
        """
        if not is_main_process():
            return

        name = name or f"step_{self.global_step}"
        checkpoint_path = self.output_dir / f"checkpoint_{name}.pt"

        state = {
            "model_state_dict": self.raw_model.state_dict(),  # Use raw model for DDP
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config.to_dict(),
        }

        if self.ema is not None:
            state["ema_state_dict"] = self.ema.state_dict()

        torch.save(state, checkpoint_path)

        # Also save config as YAML for easy loading during inference
        config_path = self.output_dir / "config.yaml"
        self.config.to_yaml(config_path)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str | Path):
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        state = torch.load(checkpoint_path, map_location=self.device)

        self.raw_model.load_state_dict(state["model_state_dict"])  # Use raw model for DDP
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]

        if self.ema is not None and "ema_state_dict" in state:
            self.ema.load_state_dict(state["ema_state_dict"])

        logger.info(f"Loaded checkpoint: {checkpoint_path}")

    def save_model(self):
        """Save final model."""
        if not is_main_process():
            return

        model_path = self.output_dir / "model.pt"
        torch.save(self.raw_model.state_dict(), model_path)  # Use raw model for DDP

        # Save EMA model separately
        if self.ema is not None:
            ema_path = self.output_dir / "model_ema.pt"
            self.ema.apply_shadow()
            torch.save(self.raw_model.state_dict(), ema_path)  # Use raw model for DDP
            self.ema.restore()

        # Save config
        config_path = self.output_dir / "config.yaml"
        self.config.to_yaml(config_path)

        logger.info(f"Saved model to {self.output_dir}")
