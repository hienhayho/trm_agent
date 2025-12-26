"""Exponential Moving Average (EMA) for model weights.

EMA helps stabilize training and prevents overfitting on small datasets.
Reference: Paper finding that EMA improves from 79.9% to 87.4% accuracy.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average of model weights.

    Maintains shadow weights that are an exponential moving average
    of the model weights. This helps stabilize training and often
    leads to better generalization.

    Usage:
        ema = EMA(model, decay=0.999)
        for batch in dataloader:
            optimizer.step()
            ema.update()  # Update shadow weights

        # For evaluation, use shadow weights
        ema.apply_shadow()
        evaluate(model)
        ema.restore()  # Restore original weights
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
    ):
        """Initialize EMA.

        Args:
            model: The model to track
            decay: EMA decay rate (default: 0.999)
            device: Device for shadow weights (default: same as model)
        """
        self.model = model
        self.decay = decay
        self.device = device

        # Create shadow copy of model weights
        self.shadow = {}
        self.backup = {}

        self._init_shadow()

    def _init_shadow(self):
        """Initialize shadow weights as copy of model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.device is not None:
                    self.shadow[name] = param.data.clone().to(self.device)
                else:
                    self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """Update shadow weights with current model weights.

        shadow = decay * shadow + (1 - decay) * current
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Apply shadow weights to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        """Restore original weights after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}

    def state_dict(self) -> dict:
        """Return EMA state for checkpointing."""
        return {
            "shadow": self.shadow,
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict: dict):
        """Load EMA state from checkpoint."""
        self.shadow = state_dict["shadow"]
        self.decay = state_dict.get("decay", self.decay)


class ModelEMA(nn.Module):
    """EMA wrapper that can be used as a model replacement.

    This creates a complete copy of the model for EMA,
    which can be used directly for inference.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        self.decay = decay

        # Create EMA model as deep copy
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        # Disable gradients for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA model weights."""
        for ema_param, model_param in zip(
            self.ema_model.parameters(), model.parameters()
        ):
            ema_param.data = (
                self.decay * ema_param.data + (1 - self.decay) * model_param.data
            )

    def forward(self, *args, **kwargs):
        """Forward pass through EMA model."""
        return self.ema_model(*args, **kwargs)

    def state_dict(self) -> dict:
        """Return EMA model state dict."""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: dict):
        """Load EMA model state dict."""
        self.ema_model.load_state_dict(state_dict)
