"""Span Decoder for TRM inference.

Decodes span predictions (start/end logits) into text values.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from trm_agent.models.trm import TRMOutput


@dataclass
class DecodedSpan:
    """A decoded span with text and positions."""

    text: str
    start: int
    end: int
    confidence: float

    @property
    def is_valid(self) -> bool:
        """Check if span is valid (non-empty with valid positions)."""
        return self.start >= 0 and self.end >= self.start and len(self.text) > 0


@dataclass
class SlotValues:
    """Decoded slot values from span extraction."""

    values: dict[str, DecodedSpan]

    def to_dict(self) -> dict[str, str]:
        """Convert to simple string dictionary."""
        return {k: v.text for k, v in self.values.items() if v.is_valid}


@dataclass
class ToolArguments:
    """Decoded tool arguments from span extraction."""

    name: str
    arguments: dict[str, DecodedSpan]

    def to_dict(self) -> dict[str, str]:
        """Convert to simple string dictionary."""
        return {k: v.text for k, v in self.arguments.items() if v.is_valid}


def decode_spans(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    token_offsets: list[tuple[int, int]],
    full_text: str,
    threshold: float = 0.0,
) -> list[DecodedSpan]:
    """Decode span logits into text values.

    Args:
        start_logits: Start position logits [seq_len, num_spans]
        end_logits: End position logits [seq_len, num_spans]
        token_offsets: List of (char_start, char_end) for each token
        full_text: The full text for extracting spans
        threshold: Minimum confidence threshold (log prob)

    Returns:
        List of DecodedSpan for each span dimension
    """
    seq_len, num_spans = start_logits.shape
    decoded = []

    for span_idx in range(num_spans):
        # Get best start and end positions
        start_probs = torch.softmax(start_logits[:, span_idx], dim=0)
        end_probs = torch.softmax(end_logits[:, span_idx], dim=0)

        start_pos = start_probs.argmax().item()
        end_pos = end_probs.argmax().item()

        # Compute confidence as geometric mean of start and end probs
        confidence = (start_probs[start_pos] * end_probs[end_pos]).sqrt().item()

        # Validate positions
        if end_pos < start_pos:
            # Invalid span - swap or skip
            decoded.append(DecodedSpan(text="", start=-1, end=-1, confidence=0.0))
            continue

        # Get character offsets
        if start_pos >= len(token_offsets) or end_pos >= len(token_offsets):
            decoded.append(DecodedSpan(text="", start=-1, end=-1, confidence=0.0))
            continue

        char_start, _ = token_offsets[start_pos]
        _, char_end = token_offsets[end_pos]

        # Handle special tokens (offset = -1)
        if char_start < 0 or char_end < 0:
            decoded.append(DecodedSpan(text="", start=-1, end=-1, confidence=0.0))
            continue

        # Extract text
        if char_start <= char_end <= len(full_text):
            text = full_text[char_start:char_end]
        else:
            text = ""

        decoded.append(
            DecodedSpan(
                text=text,
                start=start_pos,
                end=end_pos,
                confidence=confidence,
            )
        )

    return decoded


def extract_slot_values(
    outputs: TRMOutput,
    token_offsets: list[tuple[int, int]],
    full_text: str,
    slot_fields: list[str],
    presence_threshold: float = 0.5,
) -> SlotValues:
    """Extract slot values from model outputs.

    Args:
        outputs: TRM model outputs
        token_offsets: Token-to-character offset mapping
        full_text: Full input text
        slot_fields: List of slot field names
        presence_threshold: Threshold for slot presence prediction

    Returns:
        SlotValues with decoded slot spans
    """
    # Get slot presence predictions
    slot_presence = torch.sigmoid(outputs.slot_presence_logits).squeeze(0)  # [num_slots]

    # Decode spans (squeeze batch dimension)
    start_logits = outputs.slot_start_logits.squeeze(0)  # [seq_len, num_slots]
    end_logits = outputs.slot_end_logits.squeeze(0)  # [seq_len, num_slots]

    decoded_spans = decode_spans(start_logits, end_logits, token_offsets, full_text)

    # Build result
    values = {}
    for i, field in enumerate(slot_fields):
        if i < len(decoded_spans):
            span = decoded_spans[i]
            # Only include if slot is predicted as present
            if slot_presence[i].item() >= presence_threshold:
                values[field] = span
            else:
                values[field] = DecodedSpan(text="", start=-1, end=-1, confidence=0.0)
        else:
            values[field] = DecodedSpan(text="", start=-1, end=-1, confidence=0.0)

    return SlotValues(values=values)


def extract_tool_arguments(
    outputs: TRMOutput,
    token_offsets: list[tuple[int, int]],
    full_text: str,
    tool_names: list[str],
    arg_name_to_idx: dict[str, dict[str, int]],
) -> Optional[ToolArguments]:
    """Extract tool name and arguments from model outputs.

    Args:
        outputs: TRM model outputs
        token_offsets: Token-to-character offset mapping
        full_text: Full input text
        tool_names: List of tool names (indexed by tool_logits)
        arg_name_to_idx: Mapping tool_name -> arg_name -> index

    Returns:
        ToolArguments if tool_call is predicted, None otherwise
    """
    # Check decision
    decision = torch.sigmoid(outputs.decision_logits).squeeze().item()
    if decision < 0.5:
        return None  # Direct answer, no tool call

    # Get predicted tool
    tool_idx = outputs.tool_logits.squeeze(0).argmax().item()
    if tool_idx >= len(tool_names):
        return None

    tool_name = tool_names[tool_idx]

    # Decode argument spans
    start_logits = outputs.arg_start_logits.squeeze(0)  # [seq_len, max_args]
    end_logits = outputs.arg_end_logits.squeeze(0)  # [seq_len, max_args]

    decoded_spans = decode_spans(start_logits, end_logits, token_offsets, full_text)

    # Map decoded spans to argument names
    arguments = {}
    if tool_name in arg_name_to_idx:
        idx_to_arg = {v: k for k, v in arg_name_to_idx[tool_name].items()}
        for idx, span in enumerate(decoded_spans):
            if idx in idx_to_arg and span.is_valid:
                arg_name = idx_to_arg[idx]
                arguments[arg_name] = span

    return ToolArguments(name=tool_name, arguments=arguments)


class SpanDecoder:
    """High-level span decoder for TRM inference.

    Wraps the span decoding utilities with configuration.
    """

    def __init__(
        self,
        slot_fields: list[str],
        tool_names: list[str],
        arg_name_to_idx: dict[str, dict[str, int]],
        presence_threshold: float = 0.5,
    ):
        """Initialize span decoder.

        Args:
            slot_fields: List of slot field names
            tool_names: List of tool names
            arg_name_to_idx: Mapping tool_name -> arg_name -> index
            presence_threshold: Threshold for slot presence prediction
        """
        self.slot_fields = slot_fields
        self.tool_names = tool_names
        self.arg_name_to_idx = arg_name_to_idx
        self.presence_threshold = presence_threshold

    def decode(
        self,
        outputs: TRMOutput,
        token_offsets: list[tuple[int, int]],
        full_text: str,
    ) -> dict:
        """Decode model outputs into structured prediction.

        Args:
            outputs: TRM model outputs
            token_offsets: Token-to-character offset mapping
            full_text: Full input text

        Returns:
            Dictionary with:
            - decision: "tool_call" or "direct_answer"
            - slots: Dict of slot name -> value
            - tool: Dict with "name" and "arguments" (if tool_call)
        """
        # Decision
        decision_prob = torch.sigmoid(outputs.decision_logits).squeeze().item()
        decision = "tool_call" if decision_prob >= 0.5 else "direct_answer"

        # Slots
        slot_values = extract_slot_values(
            outputs,
            token_offsets,
            full_text,
            self.slot_fields,
            self.presence_threshold,
        )

        result = {
            "decision": decision,
            "decision_confidence": decision_prob if decision == "tool_call" else 1 - decision_prob,
            "slots": slot_values.to_dict(),
        }

        # Tool arguments (if tool_call)
        if decision == "tool_call":
            tool_args = extract_tool_arguments(
                outputs,
                token_offsets,
                full_text,
                self.tool_names,
                self.arg_name_to_idx,
            )
            if tool_args:
                result["tool"] = {
                    "name": tool_args.name,
                    "arguments": tool_args.to_dict(),
                }
            else:
                result["tool"] = {"name": "", "arguments": {}}

        return result

    def decode_batch(
        self,
        outputs: TRMOutput,
        batch_token_offsets: list[list[tuple[int, int]]],
        batch_full_texts: list[str],
    ) -> list[dict]:
        """Decode batch of model outputs.

        Args:
            outputs: TRM model outputs (batched)
            batch_token_offsets: List of token offsets per sample
            batch_full_texts: List of full texts per sample

        Returns:
            List of decoded predictions
        """
        batch_size = outputs.decision_logits.size(0)
        results = []

        for i in range(batch_size):
            # Create single-sample output
            single_output = TRMOutput(
                decision_logits=outputs.decision_logits[i:i+1],
                tool_logits=outputs.tool_logits[i:i+1],
                arg_start_logits=outputs.arg_start_logits[i:i+1],
                arg_end_logits=outputs.arg_end_logits[i:i+1],
                slot_start_logits=outputs.slot_start_logits[i:i+1],
                slot_end_logits=outputs.slot_end_logits[i:i+1],
                slot_presence_logits=outputs.slot_presence_logits[i:i+1],
                q_logits=outputs.q_logits[i:i+1],
                y=outputs.y[i:i+1],
                z=outputs.z[i:i+1],
            )

            result = self.decode(
                single_output,
                batch_token_offsets[i],
                batch_full_texts[i],
            )
            results.append(result)

        return results
