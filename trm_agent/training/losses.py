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


class SpanExtractionLoss(nn.Module):
    """Loss for span extraction (start/end position prediction).

    Each span is treated as a classification problem over sequence positions.
    Uses CrossEntropyLoss with ignore_index=-1 for missing spans.
    """

    def __init__(self, ignore_index: int = -1):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        start_logits: torch.Tensor,  # [batch, seq_len, num_spans]
        end_logits: torch.Tensor,  # [batch, seq_len, num_spans]
        start_labels: torch.Tensor,  # [batch, num_spans]
        end_labels: torch.Tensor,  # [batch, num_spans]
    ) -> torch.Tensor:
        """Compute span extraction loss.

        Args:
            start_logits: Start position logits [batch, seq_len, num_spans]
            end_logits: End position logits [batch, seq_len, num_spans]
            start_labels: Start position labels [batch, num_spans]
            end_labels: End position labels [batch, num_spans]

        Returns:
            Average of start and end losses
        """
        batch_size, seq_len, num_spans = start_logits.shape

        # Check if there are any valid labels (not all -1)
        valid_mask = start_labels >= 0
        if not valid_mask.any():
            # No valid spans - return zero loss with gradient connection
            return 0.0 * (start_logits.sum() + end_logits.sum())

        # Clamp labels to valid range (0 to seq_len-1) or -1 for ignore
        # This prevents out-of-bounds errors
        start_labels = start_labels.clone()
        end_labels = end_labels.clone()
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

        # Transpose: [batch, seq_len, num_spans] -> [batch, num_spans, seq_len]
        start_logits = start_logits.transpose(1, 2)
        end_logits = end_logits.transpose(1, 2)

        # Reshape for cross entropy: [batch * num_spans, seq_len]
        start_logits = start_logits.reshape(-1, seq_len)
        end_logits = end_logits.reshape(-1, seq_len)

        # Flatten labels: [batch * num_spans]
        start_labels = start_labels.view(-1)
        end_labels = end_labels.view(-1)

        # Compute losses (ignore_index=-1 handles missing spans)
        start_loss = self.ce_loss(start_logits, start_labels)
        end_loss = self.ce_loss(end_logits, end_labels)

        return (start_loss + end_loss) / 2


class TRMLoss(nn.Module):
    """Combined loss for TRM training.

    Includes:
    - Decision accuracy (tool_call vs direct_answer) with Focal Loss
    - Tool name prediction (when decision is tool_call)
    - Slot presence prediction
    - Slot span extraction
    - Tool argument span extraction
    - Q head for halting (ACT)
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

        # Span extraction losses
        self.slot_span_loss = SpanExtractionLoss(ignore_index=-1)
        self.arg_span_loss = SpanExtractionLoss(ignore_index=-1)

        # Loss weights
        self.decision_weight = config.decision_loss_weight
        self.tool_weight = config.tool_loss_weight
        self.slot_weight = config.slots_loss_weight
        self.q_weight = config.q_loss_weight
        self.slot_span_weight = config.slot_span_loss_weight
        self.arg_span_weight = config.arg_span_loss_weight

    def forward(
        self,
        outputs: TRMOutput,
        decision_labels: torch.Tensor,
        tool_name_labels: torch.Tensor,
        slot_presence_labels: torch.Tensor,
        slot_start_labels: torch.Tensor,
        slot_end_labels: torch.Tensor,
        arg_start_labels: torch.Tensor,
        arg_end_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            outputs: TRMOutput from model forward pass
            decision_labels: Binary decision labels [batch]
            tool_name_labels: Tool name labels [batch] (-1 for non-tool_call)
            slot_presence_labels: Slot presence labels [batch, num_slots]
            slot_start_labels: Slot span start labels [batch, num_slots]
            slot_end_labels: Slot span end labels [batch, num_slots]
            arg_start_labels: Argument span start labels [batch, max_args]
            arg_end_labels: Argument span end labels [batch, max_args]

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

        # 5. Slot span extraction loss
        slot_span_loss = self.slot_span_loss(
            outputs.slot_start_logits,
            outputs.slot_end_logits,
            slot_start_labels,
            slot_end_labels,
        )
        losses["slot_span_loss"] = slot_span_loss

        # 6. Argument span extraction loss (only for tool_call samples)
        if tool_mask.any():
            arg_span_loss = self.arg_span_loss(
                outputs.arg_start_logits[tool_mask],
                outputs.arg_end_logits[tool_mask],
                arg_start_labels[tool_mask],
                arg_end_labels[tool_mask],
            )
            losses["arg_span_loss"] = arg_span_loss
        else:
            # No tool_call samples - dummy loss for gradient flow
            losses["arg_span_loss"] = 0.0 * (
                outputs.arg_start_logits.sum() + outputs.arg_end_logits.sum()
            )

        # 7. Total loss
        total_loss = (
            self.decision_weight * losses["decision_loss"]
            + self.tool_weight * losses["tool_loss"]
            + self.slot_weight * losses["slot_loss"]
            + self.q_weight * losses["q_loss"]
            + self.slot_span_weight * losses["slot_span_loss"]
            + self.arg_span_weight * losses["arg_span_loss"]
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
        slot_start_labels: torch.Tensor,
        slot_end_labels: torch.Tensor,
        arg_start_labels: torch.Tensor,
        arg_end_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute loss over all supervision steps.

        Args:
            all_outputs: List of TRMOutput from each supervision step
            decision_labels: Binary decision labels [batch]
            tool_name_labels: Tool name labels [batch]
            slot_presence_labels: Slot presence labels [batch, num_slots]
            slot_start_labels: Slot span start labels [batch, num_slots]
            slot_end_labels: Slot span end labels [batch, num_slots]
            arg_start_labels: Argument span start labels [batch, max_args]
            arg_end_labels: Argument span end labels [batch, max_args]

        Returns:
            Dictionary with aggregated losses
        """
        total_losses = {
            "decision_loss": 0.0,
            "tool_loss": 0.0,
            "slot_loss": 0.0,
            "q_loss": 0.0,
            "slot_span_loss": 0.0,
            "arg_span_loss": 0.0,
            "total_loss": 0.0,
        }

        num_steps = len(all_outputs)

        for outputs in all_outputs:
            step_losses = self.step_loss(
                outputs,
                decision_labels,
                tool_name_labels,
                slot_presence_labels,
                slot_start_labels,
                slot_end_labels,
                arg_start_labels,
                arg_end_labels,
            )

            for key in total_losses:
                total_losses[key] = total_losses[key] + step_losses[key]

        # Average over steps
        for key in total_losses:
            total_losses[key] = total_losses[key] / num_steps

        return total_losses
