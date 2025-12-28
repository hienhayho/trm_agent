"""TRM Output Heads.

Output heads for the TRM model:
- DecisionHead: tool_call vs direct_answer (binary classification)
- ToolHead: Tool name prediction
- UnifiedParamHead: Unified extraction for slots + tool params with masking
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

    Predicts tool name (classification over available tools).
    Argument extraction is now handled by SharedParamHead.
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

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Predict tool name.

        Args:
            y: Answer embedding [batch, seq_len, hidden_size]

        Returns:
            Tool logits [batch, num_tools]
        """
        y = self.norm(y)

        # Tool name prediction (pooled)
        y_pooled = y.mean(dim=1)  # [batch, hidden_size]
        return self.tool_classifier(y_pooled)


class UnifiedParamHead(nn.Module):
    """Unified parameter extraction head.

    Combines slot extraction and tool param extraction into one head.
    Uses decision-based + tool-based masking during loss computation:
    - direct_answer: only slot fields valid (first num_slots)
    - tool_call: slot fields + tool-specific params valid

    The unified_fields = slot_fields + tool_param_fields (deduplicated).
    Slots are always extracted regardless of decision.
    Tool params are only valid for tool_call with tool-specific masking.
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.num_unified = config.num_unified_fields
        self.num_slots = config.num_slots

        self.norm = RMSNorm(config.hidden_size)

        # Single set of span extractors for all unified fields
        self.start_classifier = nn.Linear(
            config.hidden_size, self.num_unified, bias=False
        )
        self.end_classifier = nn.Linear(
            config.hidden_size, self.num_unified, bias=False
        )

        # Presence classifier (whether field is filled)
        self.presence_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, self.num_unified, bias=False),
        )

    def forward(self, y: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict unified field spans and presence.

        Args:
            y: Answer embedding [batch, seq_len, hidden_size]

        Returns:
            Dictionary with:
            - param_start_logits: [batch, seq_len, num_unified_fields]
            - param_end_logits: [batch, seq_len, num_unified_fields]
            - param_presence_logits: [batch, num_unified_fields]
        """
        y = self.norm(y)

        # Span prediction (per token)
        param_start_logits = self.start_classifier(y)
        param_end_logits = self.end_classifier(y)

        # Presence prediction (pooled)
        y_pooled = y.mean(dim=1)
        param_presence_logits = self.presence_classifier(y_pooled)

        return {
            "param_start_logits": param_start_logits,
            "param_end_logits": param_end_logits,
            "param_presence_logits": param_presence_logits,
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

    Combines all prediction heads into a single module:
    - DecisionHead: tool_call vs direct_answer
    - ToolHead: which tool to call
    - UnifiedParamHead: slots + tool params (with decision+tool masking)
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        self.decision_head = DecisionHead(config)
        self.tool_head = ToolHead(config)
        self.unified_param_head = UnifiedParamHead(config)
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
        tool_logits = self.tool_head(y)
        param_outputs = self.unified_param_head(y)

        return {
            "decision_logits": decision_logits,
            "tool_logits": tool_logits,
            **param_outputs,
        }
