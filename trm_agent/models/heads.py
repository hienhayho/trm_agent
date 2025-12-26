"""TRM Output Heads.

Output heads for the TRM model:
- DecisionHead: tool_call vs direct_answer (binary classification)
- ToolHead: Tool name and arguments prediction
- SlotsHead: Slot extraction from context
- QHead: Halting probability for ACT (Adaptive Computational Time)
- ContentHead: Response generation (not trained in Phase 1)
"""

import torch
import torch.nn as nn

from .config import TRMConfig
from .layers import RMSNorm


class DecisionHead(nn.Module):
    """Decision head for tool_call vs direct_answer classification.

    Binary classification: 0 = direct_answer, 1 = tool_call
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        # Pool over sequence and predict binary decision
        self.norm = RMSNorm(config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1, bias=False),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Predict decision from y embedding.

        Args:
            y: Answer embedding [batch, seq_len, hidden_size]

        Returns:
            Decision logits [batch, 1]
        """
        # Global average pooling over sequence
        y = self.norm(y)
        y_pooled = y.mean(dim=1)  # [batch, hidden_size]
        return self.classifier(y_pooled)


class ToolHead(nn.Module):
    """Tool prediction head.

    Predicts:
    - Tool name (classification over available tools)
    - Tool arguments (sequence generation or extraction)
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        self.norm = RMSNorm(config.hidden_size)

        # Tool name classifier
        self.tool_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, config.num_tools, bias=False),
        )

        # Argument extractor (predicts argument values as token indices)
        # For simplicity, we predict start/end positions for each argument
        self.arg_start_classifier = nn.Linear(
            config.hidden_size, config.max_tool_args, bias=False
        )
        self.arg_end_classifier = nn.Linear(
            config.hidden_size, config.max_tool_args, bias=False
        )

    def forward(
        self, y: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Predict tool name and arguments.

        Args:
            y: Answer embedding [batch, seq_len, hidden_size]

        Returns:
            Dictionary with:
            - tool_logits: [batch, num_tools]
            - arg_start_logits: [batch, seq_len, max_tool_args]
            - arg_end_logits: [batch, seq_len, max_tool_args]
        """
        y = self.norm(y)

        # Tool name prediction (pooled)
        y_pooled = y.mean(dim=1)  # [batch, hidden_size]
        tool_logits = self.tool_classifier(y_pooled)

        # Argument position prediction (per token)
        arg_start_logits = self.arg_start_classifier(y)
        arg_end_logits = self.arg_end_classifier(y)

        return {
            "tool_logits": tool_logits,
            "arg_start_logits": arg_start_logits,
            "arg_end_logits": arg_end_logits,
        }


class SlotsHead(nn.Module):
    """Slot extraction head.

    Extracts slot values from the conversation context.
    Each slot is predicted as a span (start, end) in the input.
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.num_slots = config.num_slots

        self.norm = RMSNorm(config.hidden_size)

        # For each slot, predict start and end positions
        self.slot_start_classifier = nn.Linear(
            config.hidden_size, config.num_slots, bias=False
        )
        self.slot_end_classifier = nn.Linear(
            config.hidden_size, config.num_slots, bias=False
        )

        # Slot presence classifier (whether slot is filled)
        self.slot_presence = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, config.num_slots, bias=False),
        )

    def forward(
        self, y: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Predict slot positions and presence.

        Args:
            y: Answer embedding [batch, seq_len, hidden_size]

        Returns:
            Dictionary with:
            - slot_start_logits: [batch, seq_len, num_slots]
            - slot_end_logits: [batch, seq_len, num_slots]
            - slot_presence_logits: [batch, num_slots]
        """
        y = self.norm(y)

        # Slot position prediction
        slot_start_logits = self.slot_start_classifier(y)
        slot_end_logits = self.slot_end_classifier(y)

        # Slot presence prediction (pooled)
        y_pooled = y.mean(dim=1)
        slot_presence_logits = self.slot_presence(y_pooled)

        return {
            "slot_start_logits": slot_start_logits,
            "slot_end_logits": slot_end_logits,
            "slot_presence_logits": slot_presence_logits,
        }


class QHead(nn.Module):
    """Q-head for Adaptive Computational Time (ACT).

    Predicts halting probability: whether the current answer is correct.
    Simplified from HRM's Q-learning approach to a single BCE loss.
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        self.norm = RMSNorm(config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1, bias=False),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Predict halting probability.

        Args:
            y: Answer embedding [batch, seq_len, hidden_size]

        Returns:
            Halting logits [batch, 1]
        """
        y = self.norm(y)
        y_pooled = y.mean(dim=1)
        return self.classifier(y_pooled)


class ContentHead(nn.Module):
    """Content generation head (for direct_answer responses).

    NOT trained in Phase 1. Defined for future use.
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Predict next token logits.

        Args:
            y: Answer embedding [batch, seq_len, hidden_size]

        Returns:
            Token logits [batch, seq_len, vocab_size]
        """
        y = self.norm(y)
        return self.lm_head(y)


class OutputHead(nn.Module):
    """Combined output head for TRM.

    Combines all prediction heads into a single module.
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        self.decision_head = DecisionHead(config)
        self.tool_head = ToolHead(config)
        self.slots_head = SlotsHead(config)
        self.content_head = ContentHead(config)  # Not trained in Phase 1

    def forward(
        self, y: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Predict all outputs from y embedding.

        Args:
            y: Answer embedding [batch, seq_len, hidden_size]

        Returns:
            Dictionary with all prediction outputs
        """
        decision_logits = self.decision_head(y)
        tool_outputs = self.tool_head(y)
        slots_outputs = self.slots_head(y)

        return {
            "decision_logits": decision_logits,
            **tool_outputs,
            **slots_outputs,
        }
