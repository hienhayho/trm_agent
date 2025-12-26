"""PyTorch Dataset for TRM Tool-Calling.

Loads JSONL dataset and prepares samples for training.
"""

import json
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from trm_agent.utils import find_value_token_span
from .tokenizer import TRMTokenizer, ROLE_IDS


class TRMToolCallingDataset(Dataset):
    """Dataset for TRM tool-calling training.

    Each sample contains:
    - Tokenized conversation history
    - Decision label (tool_call=1, direct_answer=0)
    - Tool information (name, arguments) if tool_call
    - Slot values extracted from conversation
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: TRMTokenizer,
        max_seq_len: int = 2048,
        tool_name_to_id: Optional[dict[str, int]] = None,
        slot_fields: Optional[list[str]] = None,
        max_tool_args: int = 10,
    ):
        """Initialize dataset.

        Args:
            data_path: Path to JSONL dataset file
            tokenizer: TRM tokenizer instance
            max_seq_len: Maximum sequence length
            tool_name_to_id: Mapping from tool name to tool ID
            slot_fields: List of slot field names
            max_tool_args: Maximum number of tool arguments
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tool_name_to_id = tool_name_to_id or {}
        self.max_tool_args = max_tool_args
        self.slot_fields = slot_fields or [
            "address",
            "phone",
            "device_number",
            "intent_of_user",
            "name",
            "contract_id",
        ]

        # Load samples
        self.samples = self._load_samples()

        # Build tool name mapping if not provided
        if not self.tool_name_to_id:
            self._build_tool_name_mapping()

        # Build argument name to index mapping per tool
        self.arg_name_to_idx: dict[str, dict[str, int]] = {}
        self._build_arg_name_mapping()

    def _load_samples(self) -> list[dict]:
        """Load samples from JSONL file."""
        samples = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def _build_tool_name_mapping(self):
        """Build mapping from tool names to IDs."""
        tool_names = set()
        for sample in self.samples:
            # From tools list
            for tool in sample.get("tools", []):
                if "function" in tool:
                    tool_names.add(tool["function"]["name"])
            # From tool call
            tool_info = sample.get("tool", {})
            if tool_info and "name" in tool_info:
                tool_names.add(tool_info["name"])

        self.tool_name_to_id = {name: idx for idx, name in enumerate(sorted(tool_names))}

    def _build_arg_name_mapping(self):
        """Build mapping from tool_name -> arg_name -> index."""
        for sample in self.samples:
            for tool in sample.get("tools", []):
                func = tool.get("function", {})
                tool_name = func.get("name", "")
                params = func.get("parameters", {}).get("properties", {})

                if tool_name and tool_name not in self.arg_name_to_idx:
                    self.arg_name_to_idx[tool_name] = {}

                if tool_name:
                    for idx, arg_name in enumerate(sorted(params.keys())):
                        if arg_name not in self.arg_name_to_idx[tool_name]:
                            self.arg_name_to_idx[tool_name][arg_name] = len(
                                self.arg_name_to_idx[tool_name]
                            )

    def __len__(self) -> int:
        """Get number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a training sample.

        Returns dictionary with:
        - input_ids: Token IDs [seq_len]
        - attention_mask: Attention mask [seq_len]
        - role_ids: Role IDs [seq_len]
        - decision_label: 0 or 1
        - tool_name_label: Tool ID (or -1 if not tool_call)
        - slot_presence_labels: Binary labels for slot presence [num_slots]
        - slot_start_labels: Start positions for slots [num_slots]
        - slot_end_labels: End positions for slots [num_slots]
        - arg_start_labels: Start positions for arguments [max_tool_args]
        - arg_end_labels: End positions for arguments [max_tool_args]
        """
        sample = self.samples[idx]

        # Encode conversation history with offsets for span extraction
        history = sample.get("history", [])
        encoded = self.tokenizer.encode_conversation_with_offsets(
            history, max_length=self.max_seq_len
        )

        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)
        role_ids = torch.tensor(encoded["role_ids"], dtype=torch.long)

        # Get offsets and full_text for span extraction
        token_offsets = encoded["offsets"]
        full_text = encoded["full_text"]

        # Decision label
        decision = sample.get("decision", "direct_answer")
        decision_label = torch.tensor(1 if decision == "tool_call" else 0, dtype=torch.float)

        # Tool name label
        tool_info = sample.get("tool", {})
        tool_name = tool_info.get("name", "") if tool_info else ""
        tool_name_label = torch.tensor(
            self.tool_name_to_id.get(tool_name, -1), dtype=torch.long
        )

        # Slot presence labels and span labels
        slots = sample.get("slots", {})
        num_slots = len(self.slot_fields)
        slot_presence = []
        slot_start_labels = torch.full((num_slots,), -1, dtype=torch.long)
        slot_end_labels = torch.full((num_slots,), -1, dtype=torch.long)

        for slot_idx, field in enumerate(self.slot_fields):
            value = slots.get(field, "")
            slot_presence.append(1.0 if value else 0.0)

            # Find span for non-empty values
            if value:
                span = find_value_token_span(value, full_text, token_offsets)
                if span.is_valid:
                    slot_start_labels[slot_idx] = span.start
                    slot_end_labels[slot_idx] = span.end

        slot_presence_labels = torch.tensor(slot_presence, dtype=torch.float)

        # Tool argument span labels
        arg_start_labels = torch.full((self.max_tool_args,), -1, dtype=torch.long)
        arg_end_labels = torch.full((self.max_tool_args,), -1, dtype=torch.long)

        if tool_info and decision == "tool_call":
            arguments = tool_info.get("arguments", {})
            if tool_name in self.arg_name_to_idx:
                arg_mapping = self.arg_name_to_idx[tool_name]
                for arg_name, arg_value in arguments.items():
                    if arg_name in arg_mapping:
                        arg_idx = arg_mapping[arg_name]
                        if arg_idx < self.max_tool_args and arg_value:
                            # Convert to string if not already
                            arg_value_str = str(arg_value) if not isinstance(arg_value, str) else arg_value
                            span = find_value_token_span(
                                arg_value_str, full_text, token_offsets
                            )
                            if span.is_valid:
                                arg_start_labels[arg_idx] = span.start
                                arg_end_labels[arg_idx] = span.end

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "role_ids": role_ids,
            "decision_label": decision_label,
            "tool_name_label": tool_name_label,
            "slot_presence_labels": slot_presence_labels,
            "slot_start_labels": slot_start_labels,
            "slot_end_labels": slot_end_labels,
            "arg_start_labels": arg_start_labels,
            "arg_end_labels": arg_end_labels,
        }

    def get_tool_names(self) -> list[str]:
        """Get list of tool names in order of their IDs."""
        return sorted(self.tool_name_to_id.keys(), key=lambda x: self.tool_name_to_id[x])

    def get_label_statistics(self) -> dict[str, Any]:
        """Get statistics about label distribution."""
        num_tool_call = sum(
            1 for s in self.samples if s.get("decision") == "tool_call"
        )
        num_direct = len(self.samples) - num_tool_call

        return {
            "total_samples": len(self.samples),
            "tool_call": num_tool_call,
            "direct_answer": num_direct,
            "tool_call_ratio": num_tool_call / len(self.samples) if self.samples else 0,
            "num_tools": len(self.tool_name_to_id),
            "tool_names": list(self.tool_name_to_id.keys()),
        }
