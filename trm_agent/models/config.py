"""TRM Model Configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class TRMConfig:
    """Configuration for TRM (Tiny Recursive Model) for tool-calling.

    Architecture based on the paper "Less is More: Recursive Reasoning with Tiny Networks"

    Note: Span extraction (slots/params) is handled by GLiNER2, not TRM.
    TRM only handles decision classification and tool selection.
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
    use_trm_loop: bool = True  # Use TRM recursive loop (False = single pass, let Mamba handle recurrence)
    n_latent_recursion: int = 6  # n: latent reasoning iterations
    T_deep_recursion: int = 3  # T: deep recursion iterations
    N_supervision: int = 16  # N_sup: max supervision steps

    # URM innovations (from Universal Reasoning Model paper)
    use_conv_swiglu: bool = True  # Use ConvSwiGLU instead of SwiGLU
    conv_kernel_size: int = 2  # Short conv kernel size for local feature mixing
    tbptl_no_grad_steps: int = 2  # Skip loss on first N supervision steps (TBPTL)

    # Hybrid architecture (Mamba + MoE + Attention)
    use_hybrid_block: bool = False  # Use hybrid blocks instead of Transformer

    # Mamba configuration (requires mamba-ssm package)
    mamba_version: int = 2  # Mamba version: 1 or 2 (default: 2 for better performance)
    mamba_d_state: int = 16  # SSM state dimension (Mamba1 only)
    mamba_d_conv: int = 4  # Local convolution width
    mamba_expand: int = 2  # Block expansion factor
    mamba_headdim: int = 64  # Head dimension (Mamba2 only, typically 64 or 128)
    mamba_chunk_size: int = 256  # Chunk size for Mamba2 (must be power of 2: 64, 128, 256)
    mamba_use_mem_eff_path: bool = True  # Use optimized CUDA kernels (set False if hanging)

    # MoE (Mixture of Experts) configuration - DeepSeek-V3 style
    moe_num_shared_experts: int = 1  # Shared experts (always active)
    moe_num_routed_experts: int = 8  # Routed experts (top-k selection)
    moe_top_k: int = 2  # Top-k experts per token
    moe_intermediate_size: int = 1024  # Expert MLP hidden size
    moe_use_sigmoid_gating: bool = True  # Sigmoid vs softmax gating
    moe_bias_update_speed: float = 0.001  # Bias update rate for load balancing
    moe_seq_aux_loss_weight: float = 0.0  # Sequence-wise aux loss (0 = disabled)

    # Output dimensions
    num_tools: int = 10  # Number of available tools
    num_intents: int = 0  # Number of intents (0 = disabled, load from intent_map)

    # Role tokens
    num_roles: int = 4  # user, assistant, tool_call, tool_response

    # Training
    ema_decay: float = 0.999

    # Loss weights
    decision_loss_weight: float = 1.0
    tool_loss_weight: float = 1.0
    intent_loss_weight: float = 1.0  # Intent prediction loss weight
    q_loss_weight: float = 0.5

    # Focal Loss parameters (for imbalanced classification)
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
        )
        assert self.num_layers >= 1, "num_layers must be at least 1"
        if self.use_trm_loop:
            assert self.n_latent_recursion >= 1, "n_latent_recursion must be at least 1"
            assert self.T_deep_recursion >= 1, "T_deep_recursion must be at least 1"
        assert self.mamba_version in (1, 2), "mamba_version must be 1 or 2"

        # Mamba2 requires hidden_size to be divisible by headdim
        if self.use_hybrid_block and self.mamba_version == 2:
            assert self.hidden_size % self.mamba_headdim == 0, (
                f"For Mamba2, hidden_size ({self.hidden_size}) must be divisible by "
                f"mamba_headdim ({self.mamba_headdim})"
            )
            # chunk_size must be power of 2 for Triton kernels
            assert self.mamba_chunk_size > 0 and (self.mamba_chunk_size & (self.mamba_chunk_size - 1)) == 0, (
                f"mamba_chunk_size ({self.mamba_chunk_size}) must be a power of 2 (64, 128, 256, etc.)"
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_heads

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TRMConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TRMConfig":
        """Load configuration from YAML file.

        Handles both flat configs and nested configs with 'model' section.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # Handle nested config (with 'model' section)
        if "model" in config_dict:
            config_dict = config_dict["model"]

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
            "use_trm_loop": self.use_trm_loop,
            "n_latent_recursion": self.n_latent_recursion,
            "T_deep_recursion": self.T_deep_recursion,
            "N_supervision": self.N_supervision,
            # URM innovations
            "use_conv_swiglu": self.use_conv_swiglu,
            "conv_kernel_size": self.conv_kernel_size,
            "tbptl_no_grad_steps": self.tbptl_no_grad_steps,
            # Hybrid architecture
            "use_hybrid_block": self.use_hybrid_block,
            # Mamba
            "mamba_version": self.mamba_version,
            "mamba_d_state": self.mamba_d_state,
            "mamba_d_conv": self.mamba_d_conv,
            "mamba_expand": self.mamba_expand,
            "mamba_headdim": self.mamba_headdim,
            "mamba_chunk_size": self.mamba_chunk_size,
            "mamba_use_mem_eff_path": self.mamba_use_mem_eff_path,
            # MoE (DeepSeek-V3 style)
            "moe_num_shared_experts": self.moe_num_shared_experts,
            "moe_num_routed_experts": self.moe_num_routed_experts,
            "moe_top_k": self.moe_top_k,
            "moe_intermediate_size": self.moe_intermediate_size,
            "moe_use_sigmoid_gating": self.moe_use_sigmoid_gating,
            "moe_bias_update_speed": self.moe_bias_update_speed,
            "moe_seq_aux_loss_weight": self.moe_seq_aux_loss_weight,
            "num_tools": self.num_tools,
            "num_intents": self.num_intents,
            "num_roles": self.num_roles,
            "ema_decay": self.ema_decay,
            # Loss weights
            "decision_loss_weight": self.decision_loss_weight,
            "tool_loss_weight": self.tool_loss_weight,
            "intent_loss_weight": self.intent_loss_weight,
            "q_loss_weight": self.q_loss_weight,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
        }
