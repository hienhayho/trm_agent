"""Loss functions for TRM training.

Includes:
- Focal Loss for imbalanced decision classification
- TRM Loss combining decision, tool, slots, and Q head losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from trm_agent.models.config import TRMConfig
from trm_agent.models.trm import TRMOutput


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.

    Focal Loss down-weights easy examples and focuses on hard ones.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Reference: Lin et al., "Focal Loss for Dense Object Detection"

    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('none', 'mean', 'sum')
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Predicted logits [batch, 1] or [batch]
            targets: Binary targets [batch] (0 or 1)

        Returns:
            Focal loss value
        """
        # Ensure logits is 1D
        logits = logits.view(-1)
        targets = targets.view(-1)

        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Compute pt (probability of correct class)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting
        alpha_weight = torch.where(
            targets == 1,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1 - self.alpha),
        )

        # Combine
        focal_loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class TRMLoss(nn.Module):
    """Combined loss for TRM training.

    Phase 1 training focuses on:
    - Decision accuracy (tool_call vs direct_answer) with Focal Loss
    - Tool name prediction (when decision is tool_call)
    - Slot presence prediction
    - Q head for halting (ACT)

    Content generation is NOT trained in Phase 1.
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        # Focal loss for imbalanced decision classification
        self.decision_loss = FocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
        )

        # Cross entropy for tool name
        self.tool_loss = nn.CrossEntropyLoss(ignore_index=-1)

        # BCE for slot presence
        self.slot_loss = nn.BCEWithLogitsLoss()

        # BCE for Q head (halting)
        self.q_loss = nn.BCEWithLogitsLoss()

        # Loss weights
        self.decision_weight = config.decision_loss_weight
        self.tool_weight = config.tool_loss_weight
        self.slot_weight = config.slots_loss_weight
        self.q_weight = config.q_loss_weight

    def forward(
        self,
        outputs: TRMOutput,
        decision_labels: torch.Tensor,
        tool_name_labels: torch.Tensor,
        slot_presence_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            outputs: TRMOutput from model forward pass
            decision_labels: Binary decision labels [batch]
            tool_name_labels: Tool name labels [batch] (-1 for non-tool_call)
            slot_presence_labels: Slot presence labels [batch, num_slots]

        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}

        # 1. Decision loss (Focal Loss)
        decision_loss = self.decision_loss(
            outputs.decision_logits, decision_labels
        )
        losses["decision_loss"] = decision_loss

        # 2. Tool name loss (only for tool_call samples)
        tool_mask = tool_name_labels >= 0
        if tool_mask.any():
            tool_loss = self.tool_loss(
                outputs.tool_logits[tool_mask],
                tool_name_labels[tool_mask],
            )
            losses["tool_loss"] = tool_loss
        else:
            # No tool_call samples in batch - use dummy loss to ensure gradient flow
            losses["tool_loss"] = 0.0 * outputs.tool_logits.sum()

        # 3. Slot presence loss
        slot_loss = self.slot_loss(
            outputs.slot_presence_logits,
            slot_presence_labels,
        )
        losses["slot_loss"] = slot_loss

        # 4. Q loss (halting)
        # Target: whether prediction matches ground truth
        with torch.no_grad():
            pred_decision = (torch.sigmoid(outputs.decision_logits) > 0.5).float()
            is_correct = (pred_decision.view(-1) == decision_labels).float()

        q_loss = self.q_loss(outputs.q_logits.view(-1), is_correct)
        losses["q_loss"] = q_loss

        # 5. Dummy loss for unused outputs (required for DDP gradient sync)
        # These outputs are not trained in Phase 1 but need gradients for DDP
        # Using 0.0 weight so they don't affect training, just enable gradient flow
        dummy_loss = 0.0 * (
            outputs.arg_start_logits.sum()
            + outputs.arg_end_logits.sum()
            + outputs.slot_start_logits.sum()
            + outputs.slot_end_logits.sum()
        )

        # 6. Total loss
        total_loss = (
            self.decision_weight * losses["decision_loss"]
            + self.tool_weight * losses["tool_loss"]
            + self.slot_weight * losses["slot_loss"]
            + self.q_weight * losses["q_loss"]
            + dummy_loss  # Enables gradient flow for all parameters
        )
        losses["total_loss"] = total_loss

        return losses


class DeepSupervisionLoss(nn.Module):
    """Loss for deep supervision training.

    Computes loss over multiple supervision steps and aggregates them.
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.step_loss = TRMLoss(config)

    def forward(
        self,
        all_outputs: list[TRMOutput],
        decision_labels: torch.Tensor,
        tool_name_labels: torch.Tensor,
        slot_presence_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute loss over all supervision steps.

        Args:
            all_outputs: List of TRMOutput from each supervision step
            decision_labels: Binary decision labels [batch]
            tool_name_labels: Tool name labels [batch]
            slot_presence_labels: Slot presence labels [batch, num_slots]

        Returns:
            Dictionary with aggregated losses
        """
        total_losses = {
            "decision_loss": 0.0,
            "tool_loss": 0.0,
            "slot_loss": 0.0,
            "q_loss": 0.0,
            "total_loss": 0.0,
        }

        num_steps = len(all_outputs)

        for outputs in all_outputs:
            step_losses = self.step_loss(
                outputs,
                decision_labels,
                tool_name_labels,
                slot_presence_labels,
            )

            for key in total_losses:
                total_losses[key] = total_losses[key] + step_losses[key]

        # Average over steps
        for key in total_losses:
            total_losses[key] = total_losses[key] / num_steps

        return total_losses
