"""TRM Trainer with Deep Supervision.

Implements the training loop with:
- Deep supervision (N_sup steps)
- Exponential Moving Average (EMA)
- Learning rate warmup
- Unified field extraction (slots + tool params)
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from trm_agent.data import TRMTokenizer

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
    update_unified_field_metrics,
)
from .scheduler import get_cosine_warmup_scheduler

logger = get_logger(__name__)


class TRMTrainer:
    """Trainer for TRM with deep supervision and unified field extraction."""

    def __init__(
        self,
        model: TRMForToolCalling,
        config: TRMConfig,
        training_config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        tool_names: Optional[list[str]] = None,
        unified_fields: Optional[list[str]] = None,
        num_slots: int = 0,
        tool_param_mask: Optional[torch.Tensor] = None,
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
            unified_fields: List of unified field names (slots + tool_params)
            num_slots: Number of slot fields (first num_slots in unified_fields)
            tool_param_mask: [num_tools, num_tool_params] mask tensor
            tokenizer: Tokenizer for decoding spans (optional)
        """
        self.model = model
        self.config = config
        self.training_config = training_config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tool_names = tool_names or []
        self.unified_fields = unified_fields or config.get_unified_fields()
        self.num_slots = num_slots or config.num_slots
        self.num_tool_params = len(self.unified_fields) - self.num_slots
        self.tokenizer = tokenizer

        # Check if model is wrapped with DDP
        self.is_ddp = hasattr(model, "module")
        self.raw_model = model.module if self.is_ddp else model

        # Device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if not self.is_ddp:
            self.model.to(self.device)

        # Store tool_param_mask
        if tool_param_mask is not None:
            self.tool_param_mask = tool_param_mask.to(self.device)
        else:
            self.tool_param_mask = None

        # Loss function
        self.loss_fn = DeepSupervisionLoss(config, tool_param_mask=self.tool_param_mask)

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
                batch["unified_start_labels"],
                batch["unified_end_labels"],
                batch["unified_presence_labels"],
            )

        loss = losses["total_loss"] / self.training_config.gradient_accumulation_steps

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {k: v.item() for k, v in losses.items()}

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_losses = {}
        num_batches = 0

        if hasattr(self.train_dataloader.sampler, "set_epoch"):
            self.train_dataloader.sampler.set_epoch(self.epoch)

        accum_steps = self.training_config.gradient_accumulation_steps
        total_batches = len(self.train_dataloader)
        total_opt_steps = total_batches // accum_steps

        progress_bar = tqdm(
            total=total_opt_steps,
            desc=f"Epoch {self.epoch + 1}/{self.training_config.num_epochs}",
            disable=not is_main_process(),
        )

        for step, batch in enumerate(self.train_dataloader):
            losses = self.train_step(batch)

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v
            num_batches += 1

            if (step + 1) % accum_steps == 0:
                self._optimizer_step()
                self.global_step += 1
                progress_bar.update(1)

                if self.global_step % 10 == 0:
                    avg_loss = total_losses.get("total_loss", 0) / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                if (
                    self.eval_dataloader is not None
                    and self.global_step % self.training_config.eval_interval == 0
                ):
                    eval_metrics = self.evaluate()
                    logger.info(f"Eval metrics: {eval_metrics}")

                if self.global_step % self.training_config.save_interval == 0:
                    self.save_checkpoint()

        progress_bar.close()
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
        accum.init_lists(len(self.tool_names), len(self.unified_fields))

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
            update_unified_field_metrics(
                accum, outputs, batch, self.num_slots, self.tool_param_mask
            )

        if self.ema is not None:
            self.ema.restore()

        # Compute final metrics
        metrics = compute_final_metrics(
            accum,
            self.unified_fields,
            self.num_slots,
            self.tool_names,
            self.device,
        )

        return metrics

    def train(self):
        """Full training loop."""
        log_training_start(
            self.training_config.num_epochs,
            str(self.device),
            len(self.train_dataloader.dataset),
            len(self.eval_dataloader.dataset) if self.eval_dataloader else 0,
            self.unified_fields,
            self.num_slots,
        )

        logger.info(f"Starting training for {self.training_config.num_epochs} epochs")

        for epoch in range(self.training_config.num_epochs):
            self.epoch = epoch

            avg_losses = self.train_epoch()
            logger.info(f"Epoch {epoch + 1} complete. Avg loss: {avg_losses['total_loss']:.4f}")

            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                log_eval_metrics(
                    epoch + 1,
                    eval_metrics,
                    self.unified_fields,
                    self.num_slots,
                    self.tool_names,
                )

            self.save_checkpoint(f"epoch_{epoch + 1}")

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

        if self.tool_param_mask is not None:
            state["tool_param_mask"] = self.tool_param_mask.cpu()

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

        if "tool_param_mask" in state:
            self.tool_param_mask = state["tool_param_mask"].to(self.device)

        logger.info(f"Loaded checkpoint: {checkpoint_path}")

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

        if self.tool_param_mask is not None:
            torch.save(self.tool_param_mask.cpu(), self.output_dir / "tool_param_mask.pt")

        logger.info(f"Saved model to {self.output_dir}")
