"""TRM Output Heads.

Output heads for the TRM model:
- DecisionHead: tool_call vs direct_answer (binary classification)
- ToolHead: Tool name prediction
- QHead: Halting probability for ACT (Adaptive Computational Time)
- ContentHead: Response generation (not trained in Phase 1)

Note: Span extraction (slots/params) is handled by GLiNER2, not TRM.
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


class IntentHead(nn.Module):
    """Intent prediction head.

    Predicts the next intent/action of the assistant.
    Uses classification over available intents loaded from intent mapping.
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.num_intents = getattr(config, "num_intents", 0)

        if self.num_intents > 0:
            self.norm = RMSNorm(config.hidden_size)
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2, bias=False),
                nn.GELU(),
                nn.Linear(config.hidden_size // 2, self.num_intents, bias=False),
            )
        else:
            self.norm = None
            self.classifier = None

    def forward(self, y: torch.Tensor) -> torch.Tensor | None:
        """Predict intent.

        Args:
            y: Answer embedding [batch, seq_len, hidden_size]

        Returns:
            Intent logits [batch, num_intents] or None if num_intents=0
        """
        if self.num_intents == 0 or self.classifier is None:
            return None

        y = self.norm(y)
        y_pooled = y.mean(dim=1)  # [batch, hidden_size]
        return self.classifier(y_pooled)


class OutputHead(nn.Module):
    """Combined output head for TRM.

    Combines all prediction heads into a single module:
    - DecisionHead: tool_call vs direct_answer
    - ToolHead: which tool to call
    - IntentHead: next intent prediction (optional)

    Note: Span extraction (slots/params) is handled by GLiNER2.
    Note: ContentHead is not included here (Phase 2 feature).
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        self.decision_head = DecisionHead(config)
        self.tool_head = ToolHead(config)

        # Only create IntentHead if num_intents > 0
        self.num_intents = getattr(config, "num_intents", 0)
        if self.num_intents > 0:
            self.intent_head = IntentHead(config)
        else:
            self.intent_head = None

    def forward(
        self, y: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Predict all outputs from y embedding.

        Args:
            y: Answer embedding [batch, seq_len, hidden_size]

        Returns:
            Dictionary with decision_logits, tool_logits, and intent_logits
        """
        decision_logits = self.decision_head(y)
        tool_logits = self.tool_head(y)

        result = {
            "decision_logits": decision_logits,
            "tool_logits": tool_logits,
        }

        if self.intent_head is not None:
            intent_logits = self.intent_head(y)
            if intent_logits is not None:
                result["intent_logits"] = intent_logits

        return result
