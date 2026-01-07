"""TRM for Sudoku Solving.

Implementation of TRM (Tiny Recursive Model) for solving Sudoku puzzles.
Reuses the core TRMBlock for recursive reasoning while using task-specific
embeddings and output heads.

Key concepts:
- x: embedded input (9x9 grid with values 0-9)
- y: current answer (predicted solution)
- z: latent reasoning state
- LM head: predicts value (0-9) at each of 81 positions
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import yaml

from .config import TRMConfig
from .layers import TRMBlock


@dataclass
class SudokuConfig:
    """Configuration for TRM Sudoku solver.

    The model treats Sudoku as a sequence of 81 tokens (9x9 grid flattened).
    Each token has a value (0-9) and position information (row, col, box).
    """

    # Grid
    grid_size: int = 9
    seq_len: int = 81  # 9x9
    vocab_size: int = 11  # PAD(0) + digits 1-10 (shifted by 1)

    # Architecture
    hidden_size: int = 512
    num_layers: int = 2  # Paper finding: 2 layers is optimal
    num_heads: int = 8
    intermediate_size: int = 1024
    dropout: float = 0.1
    use_attention: bool = True

    # TRM recursion
    use_trm_loop: bool = True
    n_latent_recursion: int = 6  # n: latent reasoning iterations
    T_deep_recursion: int = 3  # T: deep recursion iterations
    N_supervision: int = 8  # N_sup: supervision steps

    # URM innovations
    use_conv_swiglu: bool = True
    conv_kernel_size: int = 2
    tbptl_no_grad_steps: int = 0

    # Hybrid architecture (default: use Mamba + MoE + Attention)
    use_hybrid_block: bool = True
    mamba_version: int = 1
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 64
    mamba_chunk_size: int = 256
    mamba_use_mem_eff_path: bool = True

    # MoE (Mixture of Experts) configuration - DeepSeek-V3 style
    moe_num_shared_experts: int = 1  # Shared experts (always active)
    moe_num_routed_experts: int = 8  # Routed experts (top-k selection)
    moe_top_k: int = 2  # Top-k experts per token
    moe_intermediate_size: int = 512  # Expert MLP hidden size
    moe_use_sigmoid_gating: bool = True  # Sigmoid vs softmax gating
    moe_bias_update_speed: float = 0.001  # Bias update rate for load balancing
    moe_seq_aux_loss_weight: float = 0.0  # Sequence-wise aux loss (0 = disabled)

    # Training
    pad_token_id: int = 0
    ignore_label_id: int = 0

    def __post_init__(self):
        """Validate configuration."""
        assert (
            self.hidden_size % self.num_heads == 0
        ), f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
        assert self.grid_size == 9, "Only 9x9 Sudoku is supported"
        assert self.seq_len == 81, "seq_len must be 81 for 9x9 Sudoku"

    def to_trm_config(self) -> TRMConfig:
        """Convert to TRMConfig for reusing TRMBlock."""
        return TRMConfig(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            use_attention=self.use_attention,
            vocab_size=self.vocab_size,
            max_seq_len=self.seq_len,
            pad_token_id=self.pad_token_id,
            use_trm_loop=self.use_trm_loop,
            n_latent_recursion=self.n_latent_recursion,
            T_deep_recursion=self.T_deep_recursion,
            N_supervision=self.N_supervision,
            use_conv_swiglu=self.use_conv_swiglu,
            conv_kernel_size=self.conv_kernel_size,
            tbptl_no_grad_steps=self.tbptl_no_grad_steps,
            use_hybrid_block=self.use_hybrid_block,
            mamba_version=self.mamba_version,
            mamba_d_state=self.mamba_d_state,
            mamba_d_conv=self.mamba_d_conv,
            mamba_expand=self.mamba_expand,
            mamba_headdim=self.mamba_headdim,
            mamba_chunk_size=self.mamba_chunk_size,
            mamba_use_mem_eff_path=self.mamba_use_mem_eff_path,
            # DeepSeek-V3 MoE parameters
            moe_num_shared_experts=self.moe_num_shared_experts,
            moe_num_routed_experts=self.moe_num_routed_experts,
            moe_top_k=self.moe_top_k,
            moe_intermediate_size=self.moe_intermediate_size,
            moe_use_sigmoid_gating=self.moe_use_sigmoid_gating,
            moe_bias_update_speed=self.moe_bias_update_speed,
            moe_seq_aux_loss_weight=self.moe_seq_aux_loss_weight,
            num_tools=1,  # Not used for Sudoku
            num_roles=1,  # Not used for Sudoku
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SudokuConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # Handle nested config with 'model' section
        if "model" in config_dict:
            config_dict = config_dict["model"]

        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "grid_size": self.grid_size,
            "seq_len": self.seq_len,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "dropout": self.dropout,
            "use_attention": self.use_attention,
            "use_trm_loop": self.use_trm_loop,
            "n_latent_recursion": self.n_latent_recursion,
            "T_deep_recursion": self.T_deep_recursion,
            "N_supervision": self.N_supervision,
            "use_conv_swiglu": self.use_conv_swiglu,
            "conv_kernel_size": self.conv_kernel_size,
            "tbptl_no_grad_steps": self.tbptl_no_grad_steps,
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
            "pad_token_id": self.pad_token_id,
            "ignore_label_id": self.ignore_label_id,
        }

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


@dataclass
class SudokuOutput:
    """Output from TRMForSudoku forward pass."""

    logits: torch.Tensor  # [batch, 81, vocab_size] - value predictions
    q_logits: torch.Tensor  # [batch, 1] - halting probability
    y: torch.Tensor  # [batch, 81, hidden_size] - answer embedding
    z: torch.Tensor  # [batch, 81, hidden_size] - latent embedding
    aux_loss: torch.Tensor = None  # scalar - MoE auxiliary loss (optional)


class SudokuEmbedding(nn.Module):
    """Embed Sudoku grid with value + position information.

    Combines:
    - Value embedding: 0-9 (or 1-10 if shifted)
    - Row embedding: 0-8 (which row)
    - Column embedding: 0-8 (which column)
    - Box embedding: 0-8 (which 3x3 box)
    """

    def __init__(self, config: SudokuConfig):
        super().__init__()
        self.config = config

        # Value embedding (vocab_size values)
        self.value_embed = nn.Embedding(config.vocab_size, config.hidden_size)

        # Position embeddings
        self.row_embed = nn.Embedding(config.grid_size, config.hidden_size)
        self.col_embed = nn.Embedding(config.grid_size, config.hidden_size)
        self.box_embed = nn.Embedding(config.grid_size, config.hidden_size)

        # Learnable bias (like in original TRM)
        self.input_bias = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Pre-compute position indices for 9x9 grid
        row_ids = torch.arange(81) // 9  # 0,0,0,...,0,1,1,1,...,8,8,8
        col_ids = torch.arange(81) % 9  # 0,1,2,...,8,0,1,2,...,8
        box_ids = (row_ids // 3) * 3 + (col_ids // 3)  # 0,0,0,1,1,1,2,2,2,...

        self.register_buffer("row_ids", row_ids)
        self.register_buffer("col_ids", col_ids)
        self.register_buffer("box_ids", box_ids)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.value_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.row_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.col_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.box_embed.weight, mean=0.0, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Embed Sudoku grid.

        Args:
            values: [batch, 81] - cell values

        Returns:
            [batch, 81, hidden_size] - embedded grid
        """
        # Value embedding
        x = self.value_embed(values)

        # Add position embeddings
        x = x + self.row_embed(self.row_ids)
        x = x + self.col_embed(self.col_ids)
        x = x + self.box_embed(self.box_ids)

        # Add learnable bias
        x = x + self.input_bias

        # Normalize and dropout
        x = self.norm(x)
        x = self.dropout(x)

        return x


class TRMForSudoku(nn.Module):
    """TRM for Sudoku solving using recursive reasoning.

    The model uses the TRM algorithm to iteratively refine predictions:
    1. Embed input grid with position information
    2. Initialize y (answer) and z (latent) embeddings
    3. Recursively update z and y through the network
    4. Predict value at each position using LM head
    """

    def __init__(self, config: SudokuConfig):
        super().__init__()
        self.config = config

        # Input embedding
        self.embedding = SudokuEmbedding(config)

        # Latent embeddings (learnable initial states)
        self.y_init = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.z_init = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        # Core network (reuse TRMBlock)
        self.net = TRMBlock(config.to_trm_config())

        # Output heads
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Initialize y_init and z_init with small values
        nn.init.normal_(self.y_init, mean=0.0, std=0.02)
        nn.init.normal_(self.z_init, mean=0.0, std=0.02)

        # Initialize LM head
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        # Initialize Q head
        for module in self.q_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def latent_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        n: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform n iterations of latent reasoning, then refine y.

        Algorithm:
            for i in range(n):
                z = net(x + y + z)  # Update z with x present
            y = net(y + z)  # Refine y without x

        Args:
            x: Input embedding [batch, 81, hidden_size]
            y: Current answer [batch, 81, hidden_size]
            z: Current latent [batch, 81, hidden_size]
            n: Number of latent recursions

        Returns:
            Tuple of (y, z, aux_loss)
        """
        if n is None:
            n = self.config.n_latent_recursion

        total_aux_loss = torch.tensor(0.0, device=x.device)

        # n iterations of latent reasoning
        for _ in range(n):
            z, aux_loss = self.net(x + y + z)
            total_aux_loss = total_aux_loss + aux_loss

        # Refine y (without x)
        y, aux_loss = self.net(y + z)
        total_aux_loss = total_aux_loss + aux_loss

        return y, z, total_aux_loss

    def deep_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        n: Optional[int] = None,
        T: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform deep recursion: T-1 times without grad, 1 time with grad.

        Args:
            x: Input embedding [batch, 81, hidden_size]
            y: Current answer [batch, 81, hidden_size]
            z: Current latent [batch, 81, hidden_size]
            n: Number of latent recursions
            T: Number of deep recursions

        Returns:
            Tuple of (y_detached, z_detached, logits, q_logits, aux_loss)
        """
        if n is None:
            n = self.config.n_latent_recursion
        if T is None:
            T = self.config.T_deep_recursion

        total_aux_loss = torch.tensor(0.0, device=x.device)

        # T-1 recursions without gradients (no aux_loss contribution)
        with torch.no_grad():
            for _ in range(T - 1):
                y, z, _ = self.latent_recursion(x, y, z, n)

        # Final recursion with gradients
        y, z, aux_loss = self.latent_recursion(x, y, z, n)
        total_aux_loss = total_aux_loss + aux_loss

        # Get predictions from output heads
        logits = self.lm_head(y)  # [batch, 81, vocab_size]
        q = self.q_head(y.mean(dim=1))  # [batch, 1]

        # Detach y and z for next supervision step
        return y.detach(), z.detach(), logits, q, total_aux_loss

    def _forward_single_pass(
        self,
        values: torch.Tensor,
        return_all_steps: bool = False,
    ) -> SudokuOutput | list[SudokuOutput]:
        """Single-pass forward without TRM recursion."""
        # Embed input
        x = self.embedding(values)

        # Single pass through network
        h, aux_loss = self.net(x)

        # Get predictions
        logits = self.lm_head(h)
        q = self.q_head(h.mean(dim=1))

        output = SudokuOutput(logits=logits, q_logits=q, y=h, z=h, aux_loss=aux_loss)

        if return_all_steps:
            return [output]
        return output

    def _forward_with_trm_loop(
        self,
        values: torch.Tensor,
        return_all_steps: bool = False,
    ) -> SudokuOutput | list[SudokuOutput]:
        """Forward pass with TRM recursive loop and deep supervision."""
        batch_size = values.shape[0]

        # Embed input
        x = self.embedding(values)

        # Initialize y and z
        y = self.y_init.expand(batch_size, self.config.seq_len, -1).clone()
        z = self.z_init.expand(batch_size, self.config.seq_len, -1).clone()

        if return_all_steps:
            all_outputs = []
            # Ensure at least one step has gradients
            no_grad_steps = min(
                self.config.tbptl_no_grad_steps if self.training else 0,
                self.config.N_supervision - 1,
            )

            for step in range(self.config.N_supervision):
                if step < no_grad_steps:
                    with torch.no_grad():
                        y, z, logits, q, _ = self.deep_recursion(x, y, z)
                    y = y.detach()
                    z = z.detach()
                else:
                    y, z, logits, q, aux_loss = self.deep_recursion(x, y, z)
                    all_outputs.append(
                        SudokuOutput(logits=logits, q_logits=q, y=y, z=z, aux_loss=aux_loss)
                    )

            return all_outputs

        # Single pass (for inference)
        y, z, logits, q, aux_loss = self.deep_recursion(x, y, z)

        return SudokuOutput(logits=logits, q_logits=q, y=y, z=z, aux_loss=aux_loss)

    def forward(
        self,
        values: torch.Tensor,
        return_all_steps: bool = False,
    ) -> SudokuOutput | list[SudokuOutput]:
        """Forward pass.

        Args:
            values: [batch, 81] - input grid values
            return_all_steps: Return outputs from all supervision steps

        Returns:
            SudokuOutput or list of SudokuOutput (if return_all_steps=True)
        """
        if self.config.use_trm_loop:
            return self._forward_with_trm_loop(values, return_all_steps)
        else:
            return self._forward_single_pass(values, return_all_steps)

    def train_step(self, values: torch.Tensor) -> list[SudokuOutput]:
        """Training forward pass with deep supervision."""
        return self.forward(values, return_all_steps=True)

    @torch.no_grad()
    def inference(
        self,
        values: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> SudokuOutput:
        """Inference with full N_sup steps."""
        batch_size = values.shape[0]
        num_steps = num_steps or self.config.N_supervision

        # Embed input
        x = self.embedding(values)

        # Initialize y and z
        y = self.y_init.expand(batch_size, self.config.seq_len, -1).clone()
        z = self.z_init.expand(batch_size, self.config.seq_len, -1).clone()

        # Run all supervision steps
        for _ in range(num_steps):
            y, z, logits, q, _ = self.deep_recursion(x, y, z)

        return SudokuOutput(
            logits=logits,
            q_logits=q,
            y=y,
            z=z,
            aux_loss=torch.tensor(0.0, device=x.device),  # Not needed for inference
        )

    def solve(self, puzzle: torch.Tensor) -> torch.Tensor:
        """Solve a Sudoku puzzle.

        Args:
            puzzle: [batch, 81] or [81] - input puzzle (0=empty)

        Returns:
            [batch, 81] or [81] - solved puzzle
        """
        squeeze = puzzle.dim() == 1
        if squeeze:
            puzzle = puzzle.unsqueeze(0)

        output = self.inference(puzzle)
        solution = output.logits.argmax(dim=-1)

        if squeeze:
            solution = solution.squeeze(0)

        return solution
