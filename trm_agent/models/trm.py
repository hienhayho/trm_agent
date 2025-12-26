"""TRM (Tiny Recursive Model) for Tool-Calling.

Implementation based on the paper "Less is More: Recursive Reasoning with Tiny Networks".

Key concepts:
- x: embedded input (history + tools)
- y: current answer (decision + tool/content output)
- z: latent reasoning feature
- Single 2-layer network for both z and y updates
- Deep recursion with T-1 no-grad + 1 with-grad iterations
- Deep supervision with N_sup supervision steps
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .config import TRMConfig
from .embeddings import InputEmbedding, LatentEmbedding
from .heads import OutputHead, QHead
from .layers import TRMBlock


@dataclass
class TRMOutput:
    """Output from TRM forward pass."""

    decision_logits: torch.Tensor  # [batch, 1]
    tool_logits: torch.Tensor  # [batch, num_tools]
    arg_start_logits: torch.Tensor  # [batch, seq_len, max_tool_args]
    arg_end_logits: torch.Tensor  # [batch, seq_len, max_tool_args]
    slot_start_logits: torch.Tensor  # [batch, seq_len, num_slots]
    slot_end_logits: torch.Tensor  # [batch, seq_len, num_slots]
    slot_presence_logits: torch.Tensor  # [batch, num_slots]
    q_logits: torch.Tensor  # [batch, 1]
    y: torch.Tensor  # [batch, seq_len, hidden_size]
    z: torch.Tensor  # [batch, seq_len, hidden_size]


class TRMForToolCalling(nn.Module):
    """Tiny Recursive Model for Tool-Calling.

    The model uses a single tiny network (2 layers) that recursively:
    1. Updates latent reasoning z given (x, y, z)
    2. Refines answer y given (y, z)

    Key paper insights:
    - The task is determined by input presence: x present → reasoning, x absent → refinement
    - Full backprop through recursion (no 1-step gradient approximation)
    - 2 layers is optimal to prevent overfitting
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        # Input embedding
        self.input_embedding = InputEmbedding(config)

        # Latent embeddings for y and z initialization
        self.latent_embedding = LatentEmbedding(config)

        # Single network for both z and y updates (paper finding)
        self.net = TRMBlock(config)

        # Output heads
        self.output_head = OutputHead(config)
        self.q_head = QHead(config)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights following the paper."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _combine_for_z_update(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Combine inputs for z update: z = net(x + y + z).

        The presence of x indicates this is a reasoning step.
        """
        return x + y + z

    def _combine_for_y_update(
        self, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Combine inputs for y update: y = net(y + z).

        The absence of x indicates this is an answer refinement step.
        """
        return y + z

    def latent_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform n iterations of latent reasoning, then refine y.

        Algorithm:
            for i in range(n):
                z = net(x + y + z)  # Update z with x present
            y = net(y + z)  # Refine y without x

        Args:
            x: Input embedding [batch, seq_len, hidden_size]
            y: Current answer [batch, seq_len, hidden_size]
            z: Current latent [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch, seq_len]
            n: Number of latent recursions (default: config.n_latent_recursion)

        Returns:
            Updated (y, z) tuple
        """
        if n is None:
            n = self.config.n_latent_recursion

        # n iterations of latent reasoning
        for _ in range(n):
            combined = self._combine_for_z_update(x, y, z)
            z = self.net(combined, attention_mask)

        # Refine y (without x)
        combined = self._combine_for_y_update(y, z)
        y = self.net(combined, attention_mask)

        return y, z

    def deep_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n: Optional[int] = None,
        T: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Perform deep recursion: T-1 times without grad, 1 time with grad.

        This is the core TRM algorithm that enables learning without
        requiring the 1-step gradient approximation or fixed-point theorem.

        Args:
            x: Input embedding [batch, seq_len, hidden_size]
            y: Current answer [batch, seq_len, hidden_size]
            z: Current latent [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch, seq_len]
            n: Number of latent recursions (default: config.n_latent_recursion)
            T: Number of deep recursions (default: config.T_deep_recursion)

        Returns:
            Tuple of:
            - Updated y (detached)
            - Updated z (detached)
            - Output head predictions (dict)
            - Q head predictions
        """
        if n is None:
            n = self.config.n_latent_recursion
        if T is None:
            T = self.config.T_deep_recursion

        # T-1 recursions without gradients
        with torch.no_grad():
            for _ in range(T - 1):
                y, z = self.latent_recursion(x, y, z, attention_mask, n)

        # Final recursion with gradients
        y, z = self.latent_recursion(x, y, z, attention_mask, n)

        # Get predictions from output heads
        outputs = self.output_head(y)
        q = self.q_head(y)

        # Detach y and z for next supervision step
        return y.detach(), z.detach(), outputs, q

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        role_ids: Optional[torch.Tensor] = None,
        tool_ids: Optional[torch.Tensor] = None,
        return_all_steps: bool = False,
    ) -> TRMOutput | list[TRMOutput]:
        """Forward pass with optional deep supervision.

        During training, the model iterates through N_sup supervision steps,
        each time improving y and z.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            role_ids: Role IDs [batch, seq_len]
            tool_ids: Tool IDs [batch, num_tools]
            return_all_steps: If True, return outputs from all supervision steps

        Returns:
            TRMOutput or list of TRMOutput (if return_all_steps=True)
        """
        batch_size, seq_len = input_ids.shape

        # Embed input
        x = self.input_embedding(input_ids, role_ids, tool_ids, attention_mask)

        # Initialize y and z
        y = self.latent_embedding.get_initial_y(batch_size, seq_len)
        z = self.latent_embedding.get_initial_z(batch_size, seq_len)

        # Move to same device as x
        y = y.to(x.device)
        z = z.to(x.device)

        if return_all_steps:
            all_outputs = []
            for _ in range(self.config.N_supervision):
                y, z, outputs, q = self.deep_recursion(x, y, z, attention_mask)
                all_outputs.append(
                    TRMOutput(
                        decision_logits=outputs["decision_logits"],
                        tool_logits=outputs["tool_logits"],
                        arg_start_logits=outputs["arg_start_logits"],
                        arg_end_logits=outputs["arg_end_logits"],
                        slot_start_logits=outputs["slot_start_logits"],
                        slot_end_logits=outputs["slot_end_logits"],
                        slot_presence_logits=outputs["slot_presence_logits"],
                        q_logits=q,
                        y=y,
                        z=z,
                    )
                )
            return all_outputs

        # Single pass (for inference or last step)
        y, z, outputs, q = self.deep_recursion(x, y, z, attention_mask)

        return TRMOutput(
            decision_logits=outputs["decision_logits"],
            tool_logits=outputs["tool_logits"],
            arg_start_logits=outputs["arg_start_logits"],
            arg_end_logits=outputs["arg_end_logits"],
            slot_start_logits=outputs["slot_start_logits"],
            slot_end_logits=outputs["slot_end_logits"],
            slot_presence_logits=outputs["slot_presence_logits"],
            q_logits=q,
            y=y,
            z=z,
        )

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        role_ids: Optional[torch.Tensor] = None,
        tool_ids: Optional[torch.Tensor] = None,
    ) -> list[TRMOutput]:
        """Training forward pass with deep supervision.

        Returns outputs from all N_sup supervision steps.
        """
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            role_ids=role_ids,
            tool_ids=tool_ids,
            return_all_steps=True,
        )

    @torch.no_grad()
    def inference(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        role_ids: Optional[torch.Tensor] = None,
        tool_ids: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> TRMOutput:
        """Inference with full N_sup steps.

        At test time, we use all supervision steps for best performance.
        """
        batch_size, seq_len = input_ids.shape
        num_steps = num_steps or self.config.N_supervision

        # Embed input
        x = self.input_embedding(input_ids, role_ids, tool_ids, attention_mask)

        # Initialize y and z
        y = self.latent_embedding.get_initial_y(batch_size, seq_len).to(x.device)
        z = self.latent_embedding.get_initial_z(batch_size, seq_len).to(x.device)

        # Run all supervision steps
        for _ in range(num_steps):
            y, z, outputs, q = self.deep_recursion(x, y, z, attention_mask)

        return TRMOutput(
            decision_logits=outputs["decision_logits"],
            tool_logits=outputs["tool_logits"],
            arg_start_logits=outputs["arg_start_logits"],
            arg_end_logits=outputs["arg_end_logits"],
            slot_start_logits=outputs["slot_start_logits"],
            slot_end_logits=outputs["slot_end_logits"],
            slot_presence_logits=outputs["slot_presence_logits"],
            q_logits=q,
            y=y,
            z=z,
        )
