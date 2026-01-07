"""Loss functions for TRM training.

Includes:
- Focal Loss for imbalanced decision classification
- TRM Loss combining decision, tool, and Q head losses

Note: Span extraction (slots/params) is handled by GLiNER2, not TRM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from trm_agent.models.config import TRMConfig
from trm_agent.models.trm import TRMOutput


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced binary classification.

    Focal Loss down-weights easy examples and focuses on hard ones.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Reference: Lin et al., "Focal Loss for Dense Object Detection"
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
        """Compute focal loss."""
        logits = logits.view(-1)
        targets = targets.view(-1)

        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        alpha_weight = torch.where(
            targets == 1,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1 - self.alpha),
        )

        focal_loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class FocalCrossEntropyLoss(nn.Module):
    """Focal Loss for multi-class classification.

    Extends focal loss to multi-class using cross-entropy.
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    """

    def __init__(
        self,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -1,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal cross-entropy loss.

        Args:
            logits: [batch, num_classes] - raw logits
            targets: [batch] - class indices

        Returns:
            Focal loss scalar
        """
        # Filter out ignored indices
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            if not mask.any():
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            logits = logits[mask]
            targets = targets[mask]

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # Get probability of target class
        batch_size = targets.shape[0]
        pt = probs[torch.arange(batch_size, device=logits.device), targets]

        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class TRMLoss(nn.Module):
    """Combined loss for TRM training.

    Includes:
    - Decision accuracy (tool_call vs direct_answer) with Focal Loss
    - Tool name prediction (when decision is tool_call)
    - Intent prediction (next assistant intent) with Focal Cross-Entropy
    - Q head for halting (ACT)

    Note: Span extraction (slots/params) is handled by GLiNER2.
    """

    def __init__(self, config: TRMConfig):
        """Initialize TRM loss.

        Args:
            config: TRM configuration
        """
        super().__init__()
        self.config = config

        # Focal loss for imbalanced decision classification
        self.decision_loss = FocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
        )

        # Cross entropy for tool name
        self.tool_loss = nn.CrossEntropyLoss(ignore_index=-1)

        # Focal cross-entropy for intent prediction (multi-class)
        self.intent_loss = FocalCrossEntropyLoss(
            gamma=config.focal_gamma,
            ignore_index=-1,
        )

        # BCE for Q head (halting)
        self.q_loss = nn.BCEWithLogitsLoss()

        # Loss weights
        self.decision_weight = config.decision_loss_weight
        self.tool_weight = config.tool_loss_weight
        self.intent_weight = getattr(config, "intent_loss_weight", 1.0)
        self.q_weight = config.q_loss_weight

        # Check if intent training is enabled
        self.num_intents = getattr(config, "num_intents", 0)

    def forward(
        self,
        outputs: TRMOutput,
        decision_labels: torch.Tensor,
        tool_name_labels: torch.Tensor,
        intent_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            outputs: TRMOutput from model forward pass
            decision_labels: Binary decision labels [batch]
            tool_name_labels: Tool name labels [batch] (-1 for non-tool_call)
            intent_labels: Intent labels [batch] (-1 for unknown/ignored)

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
            # Add 0 * sum of unused logits to ensure all outputs participate in graph
            # This is needed for DDP compatibility
            if (~tool_mask).any():
                tool_loss = tool_loss + 0.0 * outputs.tool_logits[~tool_mask].sum()
            losses["tool_loss"] = tool_loss
        else:
            # No tool_call samples, but still need gradient flow for DDP
            losses["tool_loss"] = 0.0 * outputs.tool_logits.sum()

        # 3. Intent loss (Focal Cross-Entropy for multi-class)
        if (
            self.num_intents > 0
            and outputs.intent_logits is not None
            and intent_labels is not None
        ):
            intent_mask = intent_labels >= 0
            if intent_mask.any():
                intent_loss = self.intent_loss(
                    outputs.intent_logits[intent_mask],
                    intent_labels[intent_mask],
                )
                # Add 0 * sum of unused logits to ensure all outputs participate in graph
                # This is needed for DDP compatibility
                if (~intent_mask).any():
                    intent_loss = intent_loss + 0.0 * outputs.intent_logits[~intent_mask].sum()
                losses["intent_loss"] = intent_loss
            else:
                # No valid intent labels, but still need gradient flow for DDP
                losses["intent_loss"] = 0.0 * outputs.intent_logits.sum()
        else:
            losses["intent_loss"] = torch.tensor(0.0, device=outputs.tool_logits.device)

        # 4. Q loss (halting)
        with torch.no_grad():
            pred_decision = (torch.sigmoid(outputs.decision_logits) > 0.5).float()
            is_correct = (pred_decision.view(-1) == decision_labels).float()

        q_loss = self.q_loss(outputs.q_logits.view(-1), is_correct)
        losses["q_loss"] = q_loss

        # 5. Total loss
        total_loss = (
            self.decision_weight * losses["decision_loss"]
            + self.tool_weight * losses["tool_loss"]
            + self.intent_weight * losses["intent_loss"]
            + self.q_weight * losses["q_loss"]
        )
        losses["total_loss"] = total_loss

        return losses


class DeepSupervisionLoss(nn.Module):
    """Loss for deep supervision training.

    Computes loss over multiple supervision steps and aggregates them.
    """

    def __init__(self, config: TRMConfig):
        """Initialize deep supervision loss.

        Args:
            config: TRM configuration
        """
        super().__init__()
        self.config = config
        self.step_loss = TRMLoss(config)

    def forward(
        self,
        all_outputs: list[TRMOutput],
        decision_labels: torch.Tensor,
        tool_name_labels: torch.Tensor,
        intent_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute loss over all supervision steps.

        Args:
            all_outputs: List of TRMOutput from each supervision step
            decision_labels: Binary decision labels [batch]
            tool_name_labels: Tool name labels [batch]
            intent_labels: Intent labels [batch] (-1 for unknown/ignored)

        Returns:
            Dictionary with aggregated losses
        """
        total_losses = {
            "decision_loss": 0.0,
            "tool_loss": 0.0,
            "intent_loss": 0.0,
            "q_loss": 0.0,
            "total_loss": 0.0,
        }

        num_steps = len(all_outputs)

        for outputs in all_outputs:
            step_losses = self.step_loss(
                outputs,
                decision_labels,
                tool_name_labels,
                intent_labels,
            )

            for key in total_losses:
                total_losses[key] = total_losses[key] + step_losses[key]

        # Average over steps
        for key in total_losses:
            total_losses[key] = total_losses[key] / num_steps

        return total_losses
