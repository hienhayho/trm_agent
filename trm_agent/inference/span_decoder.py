"""Span Decoder for TRM inference.

Decodes span predictions (start/end logits) into text values.
Uses unified field extraction with decision+tool masking.
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
        start_probs = torch.softmax(start_logits[:, span_idx], dim=0)
        end_probs = torch.softmax(end_logits[:, span_idx], dim=0)

        start_pos = start_probs.argmax().item()
        end_pos = end_probs.argmax().item()

        confidence = (start_probs[start_pos] * end_probs[end_pos]).sqrt().item()

        if end_pos < start_pos:
            decoded.append(DecodedSpan(text="", start=-1, end=-1, confidence=0.0))
            continue

        if start_pos >= len(token_offsets) or end_pos >= len(token_offsets):
            decoded.append(DecodedSpan(text="", start=-1, end=-1, confidence=0.0))
            continue

        char_start, _ = token_offsets[start_pos]
        _, char_end = token_offsets[end_pos]

        if char_start < 0 or char_end < 0:
            decoded.append(DecodedSpan(text="", start=-1, end=-1, confidence=0.0))
            continue

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


def extract_unified_fields(
    outputs: TRMOutput,
    token_offsets: list[tuple[int, int]],
    full_text: str,
    unified_fields: list[str],
    num_slots: int,
    tool_names: list[str],
    tool_param_mask: Optional[torch.Tensor] = None,
    presence_threshold: float = 0.5,
) -> tuple[dict[str, DecodedSpan], Optional[ToolArguments]]:
    """Extract unified fields from model outputs.

    Uses decision+tool masking:
    - Slots (first num_slots): always extracted
    - Tool params: only if tool_call and tool uses that param

    Args:
        outputs: TRM model outputs
        token_offsets: Token-to-character offset mapping
        full_text: Full input text
        unified_fields: List of all field names (slots + tool_params)
        num_slots: Number of slot fields
        tool_names: List of tool names
        tool_param_mask: [num_tools, num_tool_params] mask tensor
        presence_threshold: Threshold for presence prediction

    Returns:
        Tuple of (slots dict, ToolArguments or None)
    """
    # Decision
    decision_prob = torch.sigmoid(outputs.decision_logits).squeeze().item()
    is_tool_call = decision_prob >= 0.5

    # Decode all spans
    start_logits = outputs.param_start_logits.squeeze(0)
    end_logits = outputs.param_end_logits.squeeze(0)
    presence = torch.sigmoid(outputs.param_presence_logits).squeeze(0)

    decoded_spans = decode_spans(start_logits, end_logits, token_offsets, full_text)

    # Extract slots (always valid)
    slots = {}
    for i in range(num_slots):
        if i < len(decoded_spans) and presence[i].item() >= presence_threshold:
            span = decoded_spans[i]
            if span.is_valid:
                slots[unified_fields[i]] = span

    # Extract tool arguments (only for tool_call)
    tool_args = None
    if is_tool_call:
        tool_idx = outputs.tool_logits.squeeze(0).argmax().item()
        if tool_idx < len(tool_names):
            tool_name = tool_names[tool_idx]
            arguments = {}

            for i in range(num_slots, len(unified_fields)):
                param_idx = i - num_slots

                # Check if this param is valid for this tool
                is_valid = True
                if tool_param_mask is not None and tool_idx < tool_param_mask.size(0):
                    is_valid = tool_param_mask[tool_idx, param_idx].item() > 0

                if is_valid and i < len(decoded_spans):
                    if presence[i].item() >= presence_threshold:
                        span = decoded_spans[i]
                        if span.is_valid:
                            arguments[unified_fields[i]] = span

            tool_args = ToolArguments(name=tool_name, arguments=arguments)

    return slots, tool_args


class SpanDecoder:
    """High-level span decoder for TRM inference.

    Uses unified field extraction with decision+tool masking.
    """

    def __init__(
        self,
        unified_fields: list[str],
        num_slots: int,
        tool_names: list[str],
        tool_param_mask: Optional[torch.Tensor] = None,
        presence_threshold: float = 0.5,
    ):
        """Initialize span decoder.

        Args:
            unified_fields: List of all field names (slots + tool_params)
            num_slots: Number of slot fields
            tool_names: List of tool names
            tool_param_mask: [num_tools, num_tool_params] mask tensor
            presence_threshold: Threshold for presence prediction
        """
        self.unified_fields = unified_fields
        self.num_slots = num_slots
        self.tool_names = tool_names
        self.tool_param_mask = tool_param_mask
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
            - decision_confidence: Confidence score
            - slots: Dict with slot values (always present)
            - tool: Dict with "name" and "arguments" (if tool_call)
        """
        decision_prob = torch.sigmoid(outputs.decision_logits).squeeze().item()
        decision = "tool_call" if decision_prob >= 0.5 else "direct_answer"

        result = {
            "decision": decision,
            "decision_confidence": decision_prob if decision == "tool_call" else 1 - decision_prob,
        }

        # Extract unified fields
        slots, tool_args = extract_unified_fields(
            outputs,
            token_offsets,
            full_text,
            self.unified_fields,
            self.num_slots,
            self.tool_names,
            self.tool_param_mask,
            self.presence_threshold,
        )

        # Always include slots
        result["slots"] = {k: v.text for k, v in slots.items()}

        # Include tool arguments if tool_call
        if decision == "tool_call" and tool_args:
            result["tool"] = {
                "name": tool_args.name,
                "arguments": tool_args.to_dict(),
            }
        elif decision == "tool_call":
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
            single_output = TRMOutput(
                decision_logits=outputs.decision_logits[i:i+1],
                tool_logits=outputs.tool_logits[i:i+1],
                param_start_logits=outputs.param_start_logits[i:i+1],
                param_end_logits=outputs.param_end_logits[i:i+1],
                param_presence_logits=outputs.param_presence_logits[i:i+1],
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
