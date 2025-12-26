"""TRM Model Configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TRMConfig:
    """Configuration for TRM (Tiny Recursive Model) for tool-calling.

    Architecture based on the paper "Less is More: Recursive Reasoning with Tiny Networks"
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
    num_slots: int = 6  # Number of slot fields to extract
    max_tool_args: int = 10  # Maximum number of tool arguments
    max_arg_len: int = 128  # Maximum length of each argument value

    # Role tokens
    num_roles: int = 4  # user, assistant, tool_call, tool_response

    # Training
    ema_decay: float = 0.999

    # Loss weights
    decision_loss_weight: float = 1.0
    tool_loss_weight: float = 1.0
    slots_loss_weight: float = 0.5
    q_loss_weight: float = 0.5

    # Focal Loss parameters (for imbalanced decision classification)
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Slot field names (should match dataset)
    slot_fields: list[str] = field(
        default_factory=lambda: [
            "address",
            "phone",
            "device_number",
            "intent_of_user",
            "name",
            "contract_id",
        ]
    )

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
            "num_slots": self.num_slots,
            "max_tool_args": self.max_tool_args,
            "max_arg_len": self.max_arg_len,
            "num_roles": self.num_roles,
            "ema_decay": self.ema_decay,
            "decision_loss_weight": self.decision_loss_weight,
            "tool_loss_weight": self.tool_loss_weight,
            "slots_loss_weight": self.slots_loss_weight,
            "q_loss_weight": self.q_loss_weight,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
            "slot_fields": self.slot_fields,
        }
