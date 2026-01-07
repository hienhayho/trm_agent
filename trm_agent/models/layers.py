"""TRM Model Layers.

Based on the paper "Less is More: Recursive Reasoning with Tiny Networks":
- RMSNorm (no bias)
- SwiGLU activation
- Rotary Position Embeddings
- 2-layer Transformer architecture
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TRMConfig

# Optional Mamba support (Mamba1 and Mamba2)
MAMBA_AVAILABLE = False
MAMBA2_AVAILABLE = False

try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError:
    Mamba = None

try:
    from mamba_ssm import Mamba2

    MAMBA2_AVAILABLE = True
except ImportError:
    Mamba2 = None


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Following the paper, we use RMSNorm without bias.
    Reference: Zhang & Sennrich, 2019
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        # Compute RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation function.

    SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊗ (xV + c)
    Following the paper, we don't use bias.
    Reference: Shazeer, 2020
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # Gate and up projections (no bias following the paper)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation."""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class ConvSwiGLU(nn.Module):
    """SwiGLU with depthwise short convolution for local feature mixing.

    From URM paper: Adding conv after SwiGLU gate improves reasoning
    by enhancing local token interactions (+8% on ARC-AGI).

    Reference: Universal Reasoning Model (URM), 2024
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        kernel_size: int = 2,
    ):
        super().__init__()
        # Gate and up projections (no bias following the paper)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # Depthwise short convolution for local feature mixing
        self.dwconv = nn.Conv1d(
            intermediate_size,
            intermediate_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=intermediate_size,  # depthwise
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ConvSwiGLU activation.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        seq_len = x.size(1)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up

        # Apply depthwise conv: (B, T, C) -> (B, C, T) -> conv -> (B, T, C)
        hidden = self.dwconv(hidden.transpose(1, 2)).transpose(1, 2)
        # Ensure output has the same seq_len as input (padding may add extra)
        hidden = hidden[:, :seq_len, :]
        hidden = F.silu(hidden)

        return self.down_proj(hidden)


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Reference: Su et al., 2024 (RoFormer)
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for given sequence length."""
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        # Repeat for each dim pair
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors."""
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        q_embed = self._apply_rotary(q, cos, sin)
        k_embed = self._apply_rotary(k, cos, sin)

        return q_embed, k_embed

    def _apply_rotary(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary embedding to a single tensor."""
        # Split into first and second half
        x1, x2 = x.chunk(2, dim=-1)
        # Rotate
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with Rotary Position Embeddings."""

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        # Q, K, V projections (no bias)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Rotary position embedding
        self.rotary = RotaryPositionEmbedding(
            dim=config.head_dim, max_seq_len=config.max_seq_len
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply multi-head self-attention."""
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [batch, heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        q, k = self.rotary(q, k, seq_len)

        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape back to [batch, seq_len, hidden_size]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.o_proj(output)


class MLPMixer(nn.Module):
    """MLP-Mixer style layer for fixed context length.

    Alternative to self-attention for small fixed context lengths.
    Reference: Tolstikhin et al., 2021
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        # Token mixing (across sequence dimension)
        self.token_mixer = nn.Sequential(
            nn.Linear(config.max_seq_len, config.max_seq_len, bias=False),
            nn.GELU(),
            nn.Linear(config.max_seq_len, config.max_seq_len, bias=False),
        )
        # Channel mixing (across hidden dimension)
        self.channel_mixer = SwiGLU(config.hidden_size, config.intermediate_size)

        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply MLP-Mixer layer."""
        # Token mixing: [batch, seq_len, hidden] -> transpose -> mix -> transpose back
        residual = x
        x = self.norm1(x)
        x = x.transpose(1, 2)  # [batch, hidden, seq_len]
        x = self.token_mixer(x)
        x = x.transpose(1, 2)  # [batch, seq_len, hidden]
        x = self.dropout(x) + residual

        # Channel mixing
        residual = x
        x = self.norm2(x)
        x = self.channel_mixer(x)
        x = self.dropout(x) + residual

        return x


class TransformerLayer(nn.Module):
    """Single Transformer Layer.

    Structure: RMSNorm -> Attention -> RMSNorm -> SwiGLU MLP
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        # Choose attention or MLP-Mixer
        if config.use_attention:
            self.attn = MultiHeadAttention(config)
        else:
            self.attn = MLPMixer(config)

        # Choose SwiGLU or ConvSwiGLU (URM innovation)
        if config.use_conv_swiglu:
            self.mlp = ConvSwiGLU(
                config.hidden_size,
                config.intermediate_size,
                config.conv_kernel_size,
            )
        else:
            self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply transformer layer."""
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, attention_mask)
        x = self.dropout(x) + residual

        # MLP with residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x) + residual

        return x


class MambaLayer(nn.Module):
    """Mamba selective state space layer.

    Supports both Mamba1 and Mamba2 architectures.
    - Mamba1: Original SSM with configurable d_state
    - Mamba2: Improved SSD formulation with headdim parameter

    Provides O(n) sequence processing as alternative to attention.
    Requires mamba-ssm package: pip install mamba-ssm[causal-conv1d]
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.mamba_version = config.mamba_version

        if config.mamba_version == 2:
            if not MAMBA2_AVAILABLE:
                raise ImportError(
                    "Mamba2 is required but not available. "
                    "Install with: pip install 'mamba-ssm[causal-conv1d]>=2.0' "
                    "Or set mamba_version=1 to use Mamba1."
                )
            # use_mem_eff_path=False disables CUDA kernels (use if training hangs)
            # chunk_size must be power of 2 for Triton kernels
            self.mamba = Mamba2(
                d_model=config.hidden_size,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                headdim=config.mamba_headdim,
                chunk_size=config.mamba_chunk_size,
                use_mem_eff_path=config.mamba_use_mem_eff_path,
            )
        else:  # Mamba1
            if not MAMBA_AVAILABLE:
                raise ImportError(
                    "mamba-ssm is required for MambaLayer. "
                    "Install with: pip install mamba-ssm[causal-conv1d]"
                )
            self.mamba = Mamba(
                d_model=config.hidden_size,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
            )

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply Mamba layer. Note: attention_mask is ignored (Mamba is causal)."""
        return self.mamba(x)


class Expert(nn.Module):
    """Single expert MLP using SwiGLU activation.

    Each expert is a small MLP that specializes in different input patterns.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply expert MLP with SwiGLU activation."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    """DeepSeek-V3 style Mixture of Experts with auxiliary-loss-free load balancing.

    Key innovations:
    - Shared experts (always active) + Routed experts (top-k selection)
    - Sigmoid gating instead of softmax
    - Dynamic bias for load balancing (no auxiliary loss needed)
    - Optional sequence-wise auxiliary loss for extreme imbalance prevention
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Shared experts (always active, process ALL tokens)
        self.num_shared_experts = config.moe_num_shared_experts
        self.shared_experts = nn.ModuleList([
            Expert(config.hidden_size, config.moe_intermediate_size)
            for _ in range(self.num_shared_experts)
        ])

        # Routed experts (top-k selection)
        self.num_routed_experts = config.moe_num_routed_experts
        self.routed_experts = nn.ModuleList([
            Expert(config.hidden_size, config.moe_intermediate_size)
            for _ in range(self.num_routed_experts)
        ])

        # Router: projects hidden_size -> num_routed_experts
        self.router = nn.Linear(config.hidden_size, self.num_routed_experts, bias=False)

        # Dynamic bias for load balancing (not trained by gradient)
        self.register_buffer("expert_bias", torch.zeros(self.num_routed_experts))

        # Config
        self.top_k = config.moe_top_k
        self.bias_update_speed = config.moe_bias_update_speed
        self.use_sigmoid_gating = config.moe_use_sigmoid_gating
        self.seq_aux_loss_weight = config.moe_seq_aux_loss_weight

    def _compute_routed_output(
        self,
        x_flat: torch.Tensor,
        topk_gates: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted sum of routed expert outputs."""
        output = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.routed_experts):
            # Find tokens routed to this expert
            expert_mask = (topk_indices == i).any(dim=-1)  # [B*L]
            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = expert(expert_input)

                # Get weights for this expert
                weight_idx = (topk_indices == i).float()  # [B*L, k]
                weights = (topk_gates * weight_idx).sum(dim=-1)  # [B*L]

                output[expert_mask] += weights[expert_mask].unsqueeze(-1) * expert_output

        return output

    @torch.no_grad()
    def _update_expert_bias(self, selected_experts: torch.Tensor):
        """Update bias to balance expert load without auxiliary loss (P-controller)."""
        # Count tokens per expert
        expert_counts = torch.zeros(
            self.num_routed_experts, device=selected_experts.device
        )
        for i in range(self.num_routed_experts):
            expert_counts[i] = (selected_experts == i).sum()

        # Target: uniform distribution
        total_selections = selected_experts.numel()
        target_count = total_selections / self.num_routed_experts

        # P-controller: overloaded → decrease bias, underutilized → increase bias
        bias_update = self.bias_update_speed * (target_count - expert_counts)
        self.expert_bias.add_(bias_update)

    def _compute_seq_aux_loss(
        self,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Compute per-sequence auxiliary loss for extreme imbalance prevention."""
        batch_size = router_logits.shape[0] // seq_len
        router_logits = router_logits.view(batch_size, seq_len, -1)
        selected_experts = selected_experts.view(batch_size, seq_len, -1)

        total_loss = torch.tensor(0.0, device=router_logits.device)

        for b in range(batch_size):
            seq_logits = router_logits[b]  # [L, num_routed_experts]
            seq_selected = selected_experts[b]  # [L, top_k]

            # Probability distribution
            probs = F.softmax(seq_logits, dim=-1).mean(dim=0)  # [num_routed_experts]

            # Frequency of selection
            freq = torch.zeros(self.num_routed_experts, device=seq_logits.device)
            for i in range(self.num_routed_experts):
                freq[i] = (seq_selected == i).float().mean()

            # Loss: encourage uniform distribution
            total_loss = total_loss + (probs * freq).sum() * self.num_routed_experts

        return total_loss / batch_size * self.seq_aux_loss_weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with shared + routed experts.

        Returns:
            output: [B, L, D] - MoE output
            aux_loss: scalar - auxiliary loss (0 if disabled)
        """
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)

        # 1. Shared experts (process ALL tokens)
        shared_output = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x_flat)

        # 2. Router scores with dynamic bias
        router_logits = self.router(x_flat)  # [B*L, num_routed_experts]
        scores_with_bias = router_logits + self.expert_bias

        # 3. Gating (sigmoid vs softmax)
        if self.use_sigmoid_gating:
            gates = torch.sigmoid(scores_with_bias)
        else:
            gates = F.softmax(scores_with_bias, dim=-1)

        # 4. Top-k selection
        topk_gates, topk_indices = torch.topk(gates, self.top_k, dim=-1)

        # 5. Normalize gates
        topk_gates = topk_gates / (topk_gates.sum(dim=-1, keepdim=True) + 1e-9)

        # 6. Compute routed expert outputs
        routed_output = self._compute_routed_output(x_flat, topk_gates, topk_indices)

        # 7. Update bias (training only, no gradients)
        if self.training:
            self._update_expert_bias(topk_indices)

        # 8. Combine outputs: shared + routed
        output = shared_output + routed_output

        # 9. Optional sequence-wise auxiliary loss
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.seq_aux_loss_weight > 0 and self.training:
            aux_loss = self._compute_seq_aux_loss(router_logits, topk_indices, seq_len)

        return output.view(batch_size, seq_len, hidden_size), aux_loss


class HybridBlock(nn.Module):
    """Hybrid block: Mamba + MoE + Attention.

    Combines three architectures for optimal performance:
    1. Mamba: O(n) efficient sequential processing
    2. MoE: Sparse capacity scaling (DeepSeek-V3 style with shared + routed experts)
    3. Attention: O(n²) global context

    Each sublayer uses pre-norm with residual connection.
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        # Mamba layer (O(n) sequential)
        self.mamba_norm = RMSNorm(config.hidden_size)
        self.mamba = MambaLayer(config)

        # MoE layer (DeepSeek-V3 style: shared + routed experts)
        self.moe_norm = RMSNorm(config.hidden_size)
        self.moe = MoELayer(config)

        # Attention layer (O(n²) global)
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = MultiHeadAttention(config)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Mamba -> MoE -> Attention with residuals.

        Returns:
            output: [B, L, D] - block output
            aux_loss: scalar - MoE auxiliary loss (0 if disabled)
        """
        # Mamba: O(n) sequential processing
        residual = x
        x = self.mamba_norm(x)
        x = self.mamba(x)
        x = self.dropout(x) + residual

        # MoE: sparse expert routing (returns aux_loss)
        residual = x
        x = self.moe_norm(x)
        x, aux_loss = self.moe(x)
        x = self.dropout(x) + residual

        # Attention: O(n²) global context
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, attention_mask)
        x = self.dropout(x) + residual

        return x, aux_loss


class TRMBlock(nn.Module):
    """TRM Block: Single tiny network with 2 layers.

    This is the core network used for both z and y updates in TRM.
    The paper shows that 2 layers is optimal to prevent overfitting.

    Supports two modes:
    - Standard: Transformer layers (Attention + MLP)
    - Hybrid: Mamba + MoE + Attention (requires mamba-ssm)
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.use_hybrid_block = config.use_hybrid_block

        # Choose block type based on config
        if config.use_hybrid_block:
            # Hybrid: Mamba + MoE + Attention
            self.layers = nn.ModuleList(
                [HybridBlock(config) for _ in range(config.num_layers)]
            )
        else:
            # Standard: Transformer layers
            self.layers = nn.ModuleList(
                [TransformerLayer(config) for _ in range(config.num_layers)]
            )

        # Final normalization
        self.final_norm = RMSNorm(config.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply TRM block.

        Returns:
            output: [B, L, D] - block output
            aux_loss: scalar - aggregated MoE auxiliary loss (0 if not using hybrid)
        """
        total_aux_loss = torch.tensor(0.0, device=x.device)

        for layer in self.layers:
            if self.use_hybrid_block:
                x, aux_loss = layer(x, attention_mask)
                total_aux_loss = total_aux_loss + aux_loss
            else:
                x = layer(x, attention_mask)

        return self.final_norm(x), total_aux_loss
