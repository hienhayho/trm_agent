"""TRM Model Configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class TRMConfig:
    """Configuration for TRM (Tiny Recursive Model) for tool-calling.

    Architecture based on the paper "Less is More: Recursive Reasoning with Tiny Networks"

    Unified Parameter Extraction:
    - slot_fields: Always extracted regardless of decision (direct_answer or tool_call)
    - tool_param_fields: Only valid for tool_call, masked by tool-specific mask
    - unified_fields = slot_fields + tool_param_fields (deduplicated)
    """

    # Architecture
    hidden_size: int = 512
    num_layers: int = 2  # Paper finding: 2 layers is optimal
    num_heads: int = 8
    intermediate_size: int = 2048
    dropout: float = 0.1
    use_attention: bool = True  # False for MLP-Mixer style (better for small fixed context)

    # Vocabulary
    vocab_size: int = 32000
    max_seq_len: int = 2048
    pad_token_id: int = 0

    # TRM specific hyperparameters
    n_latent_recursion: int = 6  # n: latent reasoning iterations
    T_deep_recursion: int = 3  # T: deep recursion iterations
    N_supervision: int = 16  # N_sup: max supervision steps

    # Output dimensions
    num_tools: int = 10  # Number of available tools

    # Content slots (always valid, regardless of decision)
    # NOTE: Only include fields that can be EXTRACTED from conversation text.
    # Fields like intent_of_user (semantic summary) and contract_id (looked up)
    # are GENERATED, not extracted, so they should not be trained as slots.
    slot_fields: list[str] = field(
        default_factory=lambda: [
            "address",
            "phone",
            "device_number",
            # "intent_of_user",  # NOT TRAINED: semantic summary, not extractable
            "name",
            # "contract_id",     # NOT TRAINED: looked up/generated, not in text
        ]
    )

    # Tool params (auto-collected, valid only for tool_call + tool-specific)
    tool_param_fields: Optional[list[str]] = None

    # Computed unified fields (set via set_tool_param_fields)
    num_slots: int = 0  # len(slot_fields)
    num_tool_params: int = 0  # len(tool_param_fields)
    num_unified_fields: int = 0  # num_slots + num_tool_params

    # Role tokens
    num_roles: int = 4  # user, assistant, tool_call, tool_response

    # Training
    ema_decay: float = 0.999

    # Loss weights
    decision_loss_weight: float = 1.0
    tool_loss_weight: float = 1.0
    q_loss_weight: float = 0.5
    unified_span_loss_weight: float = 0.5  # Unified span extraction
    unified_presence_loss_weight: float = 0.5  # Unified presence

    # Focal Loss parameters (for imbalanced decision classification)
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
        )
        assert self.num_layers >= 1, "num_layers must be at least 1"
        assert self.n_latent_recursion >= 1, "n_latent_recursion must be at least 1"
        assert self.T_deep_recursion >= 1, "T_deep_recursion must be at least 1"

        # Update num_slots based on slot_fields
        self.num_slots = len(self.slot_fields)

        # Update num_tool_params and num_unified_fields
        if self.tool_param_fields is not None:
            self.num_tool_params = len(self.tool_param_fields)
        self.num_unified_fields = self.num_slots + self.num_tool_params

    def set_tool_param_fields(self, fields: list[str]) -> None:
        """Set tool param fields, removing duplicates with slot_fields.

        Fields that exist in slot_fields are promoted to slots (always valid).
        Only unique tool-only params are stored in tool_param_fields.

        Args:
            fields: List of parameter names collected from dataset tools
        """
        slot_set = set(self.slot_fields)
        # Filter out duplicates - if a param is also a slot, it's already covered
        unique_params = [f for f in fields if f not in slot_set]
        # Sort for deterministic ordering
        self.tool_param_fields = sorted(set(unique_params))
        self.num_tool_params = len(self.tool_param_fields)
        self.num_unified_fields = self.num_slots + self.num_tool_params

    def get_unified_fields(self) -> list[str]:
        """Get combined field list: slots + unique tool_params."""
        return self.slot_fields + (self.tool_param_fields or [])

    def get_slot_mask(self) -> list[int]:
        """Get mask where slots=1, params=0 (for direct_answer)."""
        return [1] * self.num_slots + [0] * self.num_tool_params

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_heads

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TRMConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "dropout": self.dropout,
            "use_attention": self.use_attention,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "pad_token_id": self.pad_token_id,
            "n_latent_recursion": self.n_latent_recursion,
            "T_deep_recursion": self.T_deep_recursion,
            "N_supervision": self.N_supervision,
            "num_tools": self.num_tools,
            "num_roles": self.num_roles,
            "ema_decay": self.ema_decay,
            # Loss weights
            "decision_loss_weight": self.decision_loss_weight,
            "tool_loss_weight": self.tool_loss_weight,
            "q_loss_weight": self.q_loss_weight,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
            # Unified fields (slots + tool params)
            "slot_fields": self.slot_fields,
            "tool_param_fields": self.tool_param_fields,
            "num_slots": self.num_slots,
            "num_tool_params": self.num_tool_params,
            "num_unified_fields": self.num_unified_fields,
            "unified_span_loss_weight": self.unified_span_loss_weight,
            "unified_presence_loss_weight": self.unified_presence_loss_weight,
        }
