"""TRM Trainer with Deep Supervision.

Implements the training loop with:
- Deep supervision (N_sup steps)
- Adaptive Computational Time (ACT) for early stopping
- Exponential Moving Average (EMA)
- Learning rate warmup
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
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

    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"


def get_linear_warmup_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Create linear warmup scheduler.

    Linearly increases LR from 0 to target during warmup,
    then keeps it constant.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

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
        """
        self.model = model
        self.config = config
        self.training_config = training_config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tool_names = tool_names or []
        self.slot_fields = slot_fields or config.slot_fields

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

        # Scheduler
        total_steps = (
            len(train_dataloader)
            * training_config.num_epochs
            // training_config.gradient_accumulation_steps
        )
        self.scheduler = get_linear_warmup_scheduler(
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

        # Output directories (only create on main process)
        self.output_dir = Path(training_config.output_dir)
        self.checkpoint_dir = Path(training_config.checkpoint_dir)
        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
                    avg_decision = total_losses.get("decision_loss", 0) / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        decision=f"{avg_decision:.4f}",
                        lr=f"{lr:.2e}",
                    )

                # Logging
                if self.global_step % self.training_config.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {self.global_step} | "
                        f"Loss: {losses['total_loss']:.4f} | "
                        f"Decision: {losses['decision_loss']:.4f} | "
                        f"LR: {lr:.2e}"
                    )

                # Evaluation
                if (
                    self.eval_dataloader is not None
                    and self.global_step % self.training_config.eval_interval == 0
                ):
                    eval_metrics = self.evaluate()
                    logger.info(f"Eval metrics: {eval_metrics}")

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

        # Slot presence accuracy
        slot_correct = 0
        slot_total = 0
        per_slot_correct = [0] * len(self.slot_fields)
        per_slot_total = [0] * len(self.slot_fields)

        total_samples = 0

        for batch in self.eval_dataloader:
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
        }
        # Add per-slot metrics
        for i, field in enumerate(self.slot_fields):
            raw_metrics[f"per_slot_correct_{i}"] = per_slot_correct[i]
            raw_metrics[f"per_slot_total_{i}"] = per_slot_total[i]

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

        for i in range(len(self.slot_fields)):
            per_slot_correct[i] = agg[f"per_slot_correct_{i}"]
            per_slot_total[i] = int(agg[f"per_slot_total_{i}"])

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

        # Per-slot accuracy
        for i, field in enumerate(self.slot_fields):
            acc = per_slot_correct[i] / per_slot_total[i] if per_slot_total[i] > 0 else 0.0
            metrics[f"slot_{field}_acc"] = acc

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

    def train(self):
        """Full training loop."""
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

    def save_checkpoint(self, name: Optional[str] = None):
        """Save training checkpoint."""
        if not is_main_process():
            return

        name = name or f"step_{self.global_step}"
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

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
