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


class TRMBlock(nn.Module):
    """TRM Block: Single tiny network with 2 layers.

    This is the core network used for both z and y updates in TRM.
    The paper shows that 2 layers is optimal to prevent overfitting.
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        # Stack of transformer layers (default: 2 layers)
        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.num_layers)]
        )

        # Final normalization
        self.final_norm = RMSNorm(config.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply TRM block."""
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.final_norm(x)
