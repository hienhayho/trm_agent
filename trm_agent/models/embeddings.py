"""TRM Model Embeddings.

Input embeddings for the TRM model including:
- Token embeddings
- Role embeddings (user, assistant, tool_call, tool_response)
- Tool embeddings (learnable per tool)
"""

from typing import Optional

import torch
import torch.nn as nn

from .config import TRMConfig


class TokenEmbedding(nn.Module):
    """Token embedding layer."""

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.hidden_size = config.hidden_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input tokens."""
        return self.embedding(input_ids)


class RoleEmbedding(nn.Module):
    """Role embedding for different conversation roles.

    Roles:
        0: user
        1: assistant
        2: tool_call
        3: tool_response
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.num_roles,
            embedding_dim=config.hidden_size,
        )

    def forward(self, role_ids: torch.Tensor) -> torch.Tensor:
        """Embed role tokens."""
        return self.embedding(role_ids)


class ToolEmbedding(nn.Module):
    """Learnable embedding for each available tool.

    Each tool gets a unique embedding that helps the model
    distinguish between different tools.
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.num_tools,
            embedding_dim=config.hidden_size,
        )
        self.config = config

    def forward(self, tool_ids: torch.Tensor) -> torch.Tensor:
        """Embed tool identifiers."""
        return self.embedding(tool_ids)


class InputEmbedding(nn.Module):
    """Combined input embedding for TRM.

    Combines token, role, and position embeddings.
    The paper adds a learnable embedding of shape [0, 1, D] to the input.
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        self.token_embedding = TokenEmbedding(config)
        self.role_embedding = RoleEmbedding(config)
        self.tool_embedding = ToolEmbedding(config)

        # Learnable input bias embedding (from paper)
        self.input_bias = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Layer norm for embedding
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        tool_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Create combined input embeddings.

        Args:
            input_ids: Token IDs [batch, seq_len]
            role_ids: Role IDs [batch, seq_len] (optional)
            tool_ids: Tool IDs for available tools [batch, num_tools] (optional)
            attention_mask: Attention mask [batch, seq_len] (optional)

        Returns:
            Combined embeddings [batch, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        embeddings = self.token_embedding(input_ids)

        # Add role embeddings if provided
        if role_ids is not None:
            embeddings = embeddings + self.role_embedding(role_ids)

        # Add input bias (from paper)
        embeddings = embeddings + self.input_bias

        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class LatentEmbedding(nn.Module):
    """Initial embeddings for y (answer) and z (latent reasoning).

    The paper initializes y and z with learnable embeddings.
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        # Initial y embedding (answer/solution)
        self.y_init = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)

        # Initial z embedding (latent reasoning)
        self.z_init = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)

    def get_initial_y(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """Get initial y embedding expanded to batch and sequence."""
        return self.y_init.expand(batch_size, seq_len, -1)

    def get_initial_z(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """Get initial z embedding expanded to batch and sequence."""
        return self.z_init.expand(batch_size, seq_len, -1)
