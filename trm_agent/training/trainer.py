"""TRM Trainer with Deep Supervision.

Implements the training loop with:
- Deep supervision (N_sup steps)
- Exponential Moving Average (EMA)
- Learning rate warmup
- Gradient accumulation with DDP no_sync optimization

Note: Span extraction (slots/params) is handled by GLiNER2.
TRM only handles decision classification and tool selection.
"""

from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from trm_agent.models.config import TRMConfig
from trm_agent.models.ema import EMA
from trm_agent.models.trm import TRMForToolCalling
from trm_agent.utils import get_logger, is_main_process

from .config import TrainingConfig
from .losses import DeepSupervisionLoss
from .logging import log_eval_metrics, log_training_start
from .metrics import (
    EvalAccumulators,
    compute_final_metrics,
    update_decision_metrics,
    update_tool_metrics,
)
from .scheduler import get_cosine_warmup_scheduler

logger = get_logger(__name__)


class TRMTrainer:
    """Trainer for TRM with deep supervision."""

    def __init__(
        self,
        model: TRMForToolCalling,
        config: TRMConfig,
        training_config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        tool_names: Optional[list[str]] = None,
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
        """
        self.model = model
        self.config = config
        self.training_config = training_config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tool_names = tool_names or []

        # Check if model is wrapped with DDP
        self.is_ddp = hasattr(model, "module")
        self.raw_model = model.module if self.is_ddp else model

        # Device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
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
        self.scheduler = get_cosine_warmup_scheduler(
            self.optimizer,
            training_config.warmup_steps,
            total_steps,
        )

        # EMA
        self.ema = None
        if training_config.use_ema:
            self.ema = EMA(self.raw_model, decay=training_config.ema_decay)

        # AMP
        self._setup_amp()

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Output directory
        self.output_dir = Path(training_config.output_dir)
        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_amp(self):
        """Setup automatic mixed precision."""
        self.use_amp = (
            self.training_config.use_amp and self.device.type == "cuda"
        )
        if self.use_amp:
            amp_dtype = self.training_config.amp_dtype.lower()
            if amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler("cuda")
            logger.info(f"Using AMP with dtype={self.amp_dtype}")
        else:
            self.amp_dtype = torch.float32
            self.scaler = None

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single training step with deep supervision."""
        self.model.train()
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
            all_outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                role_ids=batch["role_ids"],
                return_all_steps=True,
            )

            losses = self.loss_fn(
                all_outputs,
                batch["decision_labels"],
                batch["tool_name_labels"],
            )

        loss = losses["total_loss"] / self.training_config.gradient_accumulation_steps

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {k: v.item() for k, v in losses.items()}

    def train_epoch(self, progress_bar: Optional[tqdm] = None) -> dict[str, float]:
        """Train for one epoch.

        Args:
            progress_bar: Optional global progress bar to update
        """
        self.model.train()
        total_losses = {}
        num_batches = 0

        if hasattr(self.train_dataloader.sampler, "set_epoch"):
            self.train_dataloader.sampler.set_epoch(self.epoch)

        accum_steps = self.training_config.gradient_accumulation_steps

        for step, batch in enumerate(self.train_dataloader):
            # Use no_sync for intermediate accumulation steps in DDP
            # This avoids gradient synchronization until the final accumulation step
            is_accumulating = (step + 1) % accum_steps != 0
            sync_context = (
                self.model.no_sync() if (self.is_ddp and is_accumulating) else nullcontext()
            )

            with sync_context:
                losses = self.train_step(batch)

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v
            num_batches += 1

            if not is_accumulating:
                self._optimizer_step()
                self.global_step += 1

                # Update global progress bar
                if progress_bar is not None:
                    progress_bar.update(1)
                    if self.global_step % 10 == 0:
                        avg_loss = total_losses.get("total_loss", 0) / num_batches
                        lr = self.scheduler.get_last_lr()[0]
                        progress_bar.set_postfix(
                            epoch=f"{self.epoch + 1}/{self.training_config.num_epochs}",
                            loss=f"{avg_loss:.4f}",
                            lr=f"{lr:.2e}",
                        )

                # Step-based checkpoint saving (optional, for long epochs)
                if (
                    self.training_config.save_interval > 0
                    and self.global_step % self.training_config.save_interval == 0
                ):
                    self.save_checkpoint()

        return {k: v / num_batches for k, v in total_losses.items()}

    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.training_config.max_grad_norm,
        )

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

        if self.ema is not None:
            self.ema.update()

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate model on validation set."""
        if self.eval_dataloader is None:
            return {}

        self.model.eval()

        if self.ema is not None:
            self.ema.apply_shadow()

        # Initialize accumulators
        accum = EvalAccumulators()
        accum.init_lists(len(self.tool_names))

        eval_progress = tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            disable=not is_main_process(),
        )

        for batch in eval_progress:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.raw_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                role_ids=batch["role_ids"],
                return_all_steps=False,
            )

            batch_size = batch["decision_labels"].size(0)
            accum.total_samples += batch_size

            # Decision predictions
            decision_pred = (
                (torch.sigmoid(outputs.decision_logits) > 0.5).long().view(-1)
            )
            decision_true = batch["decision_labels"].long()

            update_decision_metrics(accum, decision_pred, decision_true)
            update_tool_metrics(accum, outputs, batch["tool_name_labels"], len(self.tool_names))

        if self.ema is not None:
            self.ema.restore()

        # Compute final metrics
        metrics = compute_final_metrics(
            accum,
            self.tool_names,
            self.device,
        )

        return metrics

    def train(self):
        """Full training loop with global step progress bar."""
        # Calculate total steps and starting point
        accum_steps = self.training_config.gradient_accumulation_steps
        steps_per_epoch = len(self.train_dataloader) // accum_steps
        total_steps = steps_per_epoch * self.training_config.num_epochs

        # Determine starting epoch (for resuming from checkpoint)
        # If global_step > 0, we're resuming, so start from next epoch
        start_epoch = self.epoch + 1 if self.global_step > 0 else 0

        log_training_start(
            self.training_config.num_epochs,
            str(self.device),
            len(self.train_dataloader.dataset),
            len(self.eval_dataloader.dataset) if self.eval_dataloader else 0,
            self.tool_names,
        )

        if start_epoch > 0:
            logger.info(
                f"Resuming training from epoch {start_epoch + 1}/{self.training_config.num_epochs}, "
                f"step {self.global_step}/{total_steps}"
            )
        else:
            logger.info(
                f"Starting training: {self.training_config.num_epochs} epochs, "
                f"{total_steps} total steps ({steps_per_epoch} steps/epoch)"
            )

        # Create global progress bar
        progress_bar = tqdm(
            total=total_steps,
            initial=self.global_step,  # Resume from current step
            desc="Training",
            disable=not is_main_process(),
        )

        for epoch in range(start_epoch, self.training_config.num_epochs):
            self.epoch = epoch

            avg_losses = self.train_epoch(progress_bar)
            logger.info(f"Epoch {epoch + 1} complete. Avg loss: {avg_losses['total_loss']:.4f}")

            # Epoch-end evaluation
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                log_eval_metrics(
                    epoch + 1,
                    eval_metrics,
                    self.tool_names,
                )

            # Epoch-end checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")

        progress_bar.close()
        self.save_model()
        logger.info("Training complete!")

    def save_checkpoint(self, name: Optional[str] = None):
        """Save training checkpoint."""
        if not is_main_process():
            return

        name = name or f"step_{self.global_step}"
        checkpoint_path = self.output_dir / f"checkpoint_{name}.pt"

        state = {
            "model_state_dict": self.raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config.to_dict(),
        }

        if self.ema is not None:
            state["ema_state_dict"] = self.ema.state_dict()

        torch.save(state, checkpoint_path)
        self.config.to_yaml(self.output_dir / "config.yaml")

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str | Path):
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        state = torch.load(checkpoint_path, map_location=self.device)

        self.raw_model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]

        if self.ema is not None and "ema_state_dict" in state:
            self.ema.load_state_dict(state["ema_state_dict"])

        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        logger.info(f"  Resumed at epoch {self.epoch + 1}, global_step {self.global_step}")

    def save_model(self):
        """Save final model."""
        if not is_main_process():
            return

        torch.save(self.raw_model.state_dict(), self.output_dir / "model.pt")

        if self.ema is not None:
            self.ema.apply_shadow()
            torch.save(self.raw_model.state_dict(), self.output_dir / "model_ema.pt")
            self.ema.restore()

        self.config.to_yaml(self.output_dir / "config.yaml")

        logger.info(f"Saved model to {self.output_dir}")
