"""Loss functions for TRM training.

Includes:
- Focal Loss for imbalanced decision classification
- UnifiedSpanLoss for parameter extraction with decision+tool masking
- TRM Loss combining decision, tool, unified params, and Q head losses
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


class UnifiedSpanLoss(nn.Module):
    """Span loss with decision + tool-based masking.

    Handles unified fields (slots + tool_params):
    - For direct_answer: only slot fields are valid
    - For tool_call: slot fields + tool-specific params are valid
    """

    def __init__(self, num_slots: int, ignore_index: int = -1):
        """Initialize unified span loss.

        Args:
            num_slots: Number of slot fields (always valid)
            ignore_index: Label value to ignore (-1 for missing spans)
        """
        super().__init__()
        self.num_slots = num_slots
        self.ignore_index = ignore_index

    def _build_unified_mask(
        self,
        decision_labels: torch.Tensor,  # [batch] 0=direct, 1=tool_call
        tool_name_labels: torch.Tensor,  # [batch] tool index or -1
        tool_param_mask: torch.Tensor,  # [num_tools, num_tool_params]
        num_unified_fields: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build per-sample unified mask based on decision and tool.

        Args:
            decision_labels: 0=direct_answer, 1=tool_call
            tool_name_labels: Tool index or -1 for direct_answer
            tool_param_mask: [num_tools, num_tool_params] which params each tool uses
            num_unified_fields: Total number of unified fields
            device: Tensor device

        Returns:
            mask: [batch, num_unified_fields] where 1=valid, 0=masked
        """
        batch_size = decision_labels.size(0)
        mask = torch.zeros(batch_size, num_unified_fields, device=device)

        for i in range(batch_size):
            # Slots are always valid
            mask[i, :self.num_slots] = 1.0

            # Tool params only valid for tool_call with valid tool
            if decision_labels[i] == 1 and tool_name_labels[i] >= 0:
                tool_idx = tool_name_labels[i].item()
                if tool_idx < tool_param_mask.size(0):
                    mask[i, self.num_slots:] = tool_param_mask[tool_idx]

        return mask

    def forward(
        self,
        start_logits: torch.Tensor,  # [batch, seq_len, num_unified]
        end_logits: torch.Tensor,  # [batch, seq_len, num_unified]
        start_labels: torch.Tensor,  # [batch, num_unified]
        end_labels: torch.Tensor,  # [batch, num_unified]
        decision_labels: torch.Tensor,  # [batch]
        tool_name_labels: torch.Tensor,  # [batch]
        tool_param_mask: torch.Tensor,  # [num_tools, num_tool_params]
    ) -> torch.Tensor:
        """Compute span extraction loss with decision+tool masking.

        Args:
            start_logits: Start position logits [batch, seq_len, num_unified]
            end_logits: End position logits [batch, seq_len, num_unified]
            start_labels: Start position labels [batch, num_unified]
            end_labels: End position labels [batch, num_unified]
            decision_labels: Decision labels [batch] (0=direct, 1=tool_call)
            tool_name_labels: Tool index labels [batch]
            tool_param_mask: [num_tools, num_tool_params] mask

        Returns:
            Masked span extraction loss
        """
        batch_size, seq_len, num_unified = start_logits.shape
        device = start_logits.device

        # Build unified mask
        unified_mask = self._build_unified_mask(
            decision_labels, tool_name_labels, tool_param_mask,
            num_unified, device
        )

        # Apply mask to labels: set masked fields to -1 (ignore)
        start_labels = torch.where(
            unified_mask.bool(),
            start_labels,
            torch.full_like(start_labels, self.ignore_index),
        )
        end_labels = torch.where(
            unified_mask.bool(),
            end_labels,
            torch.full_like(end_labels, self.ignore_index),
        )

        # Check if there are any valid labels
        valid_mask = start_labels >= 0
        if not valid_mask.any():
            return 0.0 * (start_logits.sum() + end_logits.sum())

        # Clamp labels to valid range
        start_labels = torch.where(
            start_labels >= 0,
            start_labels.clamp(0, seq_len - 1),
            start_labels,
        )
        end_labels = torch.where(
            end_labels >= 0,
            end_labels.clamp(0, seq_len - 1),
            end_labels,
        )

        # Transpose: [batch, seq_len, num_unified] -> [batch, num_unified, seq_len]
        start_logits = start_logits.transpose(1, 2)
        end_logits = end_logits.transpose(1, 2)

        # Reshape for cross entropy: [batch * num_unified, seq_len]
        start_logits = start_logits.reshape(-1, seq_len)
        end_logits = end_logits.reshape(-1, seq_len)

        # Flatten labels: [batch * num_unified]
        start_labels = start_labels.view(-1)
        end_labels = end_labels.view(-1)

        # Compute losses (ignore_index=-1 handles masked fields)
        ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        start_loss = ce_loss(start_logits, start_labels)
        end_loss = ce_loss(end_logits, end_labels)

        return (start_loss + end_loss) / 2


class UnifiedPresenceLoss(nn.Module):
    """Presence loss with decision + tool-based masking."""

    def __init__(self, num_slots: int):
        """Initialize unified presence loss.

        Args:
            num_slots: Number of slot fields (always valid)
        """
        super().__init__()
        self.num_slots = num_slots
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        presence_logits: torch.Tensor,  # [batch, num_unified]
        presence_labels: torch.Tensor,  # [batch, num_unified]
        decision_labels: torch.Tensor,  # [batch]
        tool_name_labels: torch.Tensor,  # [batch]
        tool_param_mask: torch.Tensor,  # [num_tools, num_tool_params]
    ) -> torch.Tensor:
        """Compute presence loss with masking.

        Args:
            presence_logits: Presence logits [batch, num_unified]
            presence_labels: Presence labels [batch, num_unified]
            decision_labels: Decision labels [batch]
            tool_name_labels: Tool index labels [batch]
            tool_param_mask: [num_tools, num_tool_params] mask

        Returns:
            Masked presence loss
        """
        batch_size, num_unified = presence_logits.shape
        device = presence_logits.device

        # Build unified mask
        mask = torch.zeros(batch_size, num_unified, device=device)
        for i in range(batch_size):
            mask[i, :self.num_slots] = 1.0
            if decision_labels[i] == 1 and tool_name_labels[i] >= 0:
                tool_idx = tool_name_labels[i].item()
                if tool_idx < tool_param_mask.size(0):
                    mask[i, self.num_slots:] = tool_param_mask[tool_idx]

        # Compute per-element loss
        loss = self.bce_loss(presence_logits, presence_labels)

        # Apply mask and average
        masked_loss = loss * mask
        num_valid = mask.sum()

        if num_valid > 0:
            return masked_loss.sum() / num_valid
        return 0.0 * loss.sum()


class TRMLoss(nn.Module):
    """Combined loss for TRM training.

    Includes:
    - Decision accuracy (tool_call vs direct_answer) with Focal Loss
    - Tool name prediction (when decision is tool_call)
    - Unified parameter presence and span extraction with decision+tool masking
    - Q head for halting (ACT)
    """

    def __init__(
        self,
        config: TRMConfig,
        tool_param_mask: torch.Tensor | None = None,
    ):
        """Initialize TRM loss.

        Args:
            config: TRM configuration
            tool_param_mask: [num_tools, num_tool_params] mask tensor
        """
        super().__init__()
        self.config = config

        # Register tool_param_mask as buffer
        if tool_param_mask is not None:
            self.register_buffer("tool_param_mask", tool_param_mask)
        else:
            # Default: all params valid for all tools
            self.register_buffer(
                "tool_param_mask",
                torch.ones(config.num_tools, config.num_tool_params)
            )

        # Focal loss for imbalanced decision classification
        self.decision_loss = FocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
        )

        # Cross entropy for tool name
        self.tool_loss = nn.CrossEntropyLoss(ignore_index=-1)

        # Unified span and presence losses
        self.unified_span_loss = UnifiedSpanLoss(
            num_slots=config.num_slots,
            ignore_index=-1,
        )
        self.unified_presence_loss = UnifiedPresenceLoss(
            num_slots=config.num_slots,
        )

        # BCE for Q head (halting)
        self.q_loss = nn.BCEWithLogitsLoss()

        # Loss weights
        self.decision_weight = config.decision_loss_weight
        self.tool_weight = config.tool_loss_weight
        self.unified_span_weight = config.unified_span_loss_weight
        self.unified_presence_weight = config.unified_presence_loss_weight
        self.q_weight = config.q_loss_weight

    def forward(
        self,
        outputs: TRMOutput,
        decision_labels: torch.Tensor,
        tool_name_labels: torch.Tensor,
        unified_start_labels: torch.Tensor,
        unified_end_labels: torch.Tensor,
        unified_presence_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            outputs: TRMOutput from model forward pass
            decision_labels: Binary decision labels [batch]
            tool_name_labels: Tool name labels [batch] (-1 for non-tool_call)
            unified_start_labels: Unified start labels [batch, num_unified]
            unified_end_labels: Unified end labels [batch, num_unified]
            unified_presence_labels: Unified presence labels [batch, num_unified]

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
            losses["tool_loss"] = 0.0 * outputs.tool_logits.sum()

        # 3. Unified span loss (with decision+tool masking)
        span_loss = self.unified_span_loss(
            outputs.param_start_logits,
            outputs.param_end_logits,
            unified_start_labels,
            unified_end_labels,
            decision_labels,
            tool_name_labels,
            self.tool_param_mask,
        )
        losses["unified_span_loss"] = span_loss

        # 4. Unified presence loss (with decision+tool masking)
        presence_loss = self.unified_presence_loss(
            outputs.param_presence_logits,
            unified_presence_labels,
            decision_labels,
            tool_name_labels,
            self.tool_param_mask,
        )
        losses["unified_presence_loss"] = presence_loss

        # 5. Q loss (halting)
        with torch.no_grad():
            pred_decision = (torch.sigmoid(outputs.decision_logits) > 0.5).float()
            is_correct = (pred_decision.view(-1) == decision_labels).float()

        q_loss = self.q_loss(outputs.q_logits.view(-1), is_correct)
        losses["q_loss"] = q_loss

        # 6. Total loss
        total_loss = (
            self.decision_weight * losses["decision_loss"]
            + self.tool_weight * losses["tool_loss"]
            + self.unified_span_weight * losses["unified_span_loss"]
            + self.unified_presence_weight * losses["unified_presence_loss"]
            + self.q_weight * losses["q_loss"]
        )
        losses["total_loss"] = total_loss

        return losses


class DeepSupervisionLoss(nn.Module):
    """Loss for deep supervision training.

    Computes loss over multiple supervision steps and aggregates them.
    """

    def __init__(
        self,
        config: TRMConfig,
        tool_param_mask: torch.Tensor | None = None,
    ):
        """Initialize deep supervision loss.

        Args:
            config: TRM configuration
            tool_param_mask: [num_tools, num_tool_params] mask tensor
        """
        super().__init__()
        self.config = config
        self.step_loss = TRMLoss(config, tool_param_mask=tool_param_mask)

    def forward(
        self,
        all_outputs: list[TRMOutput],
        decision_labels: torch.Tensor,
        tool_name_labels: torch.Tensor,
        unified_start_labels: torch.Tensor,
        unified_end_labels: torch.Tensor,
        unified_presence_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute loss over all supervision steps.

        Args:
            all_outputs: List of TRMOutput from each supervision step
            decision_labels: Binary decision labels [batch]
            tool_name_labels: Tool name labels [batch]
            unified_start_labels: Unified start labels [batch, num_unified]
            unified_end_labels: Unified end labels [batch, num_unified]
            unified_presence_labels: Unified presence labels [batch, num_unified]

        Returns:
            Dictionary with aggregated losses
        """
        total_losses = {
            "decision_loss": 0.0,
            "tool_loss": 0.0,
            "unified_span_loss": 0.0,
            "unified_presence_loss": 0.0,
            "q_loss": 0.0,
            "total_loss": 0.0,
        }

        num_steps = len(all_outputs)

        for outputs in all_outputs:
            step_losses = self.step_loss(
                outputs,
                decision_labels,
                tool_name_labels,
                unified_start_labels,
                unified_end_labels,
                unified_presence_labels,
            )

            for key in total_losses:
                total_losses[key] = total_losses[key] + step_losses[key]

        # Average over steps
        for key in total_losses:
            total_losses[key] = total_losses[key] / num_steps

        return total_losses
