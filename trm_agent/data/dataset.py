"""PyTorch Dataset for TRM Tool-Calling.

Loads JSONL dataset and prepares samples for training.
"""

import json
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

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
    ):
        """Initialize dataset.

        Args:
            data_path: Path to JSONL dataset file
            tokenizer: TRM tokenizer instance
            max_seq_len: Maximum sequence length
            tool_name_to_id: Mapping from tool name to tool ID
            slot_fields: List of slot field names
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tool_name_to_id = tool_name_to_id or {}
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
        """
        sample = self.samples[idx]

        # Encode conversation history
        history = sample.get("history", [])
        encoded = self.tokenizer.encode_conversation(history, max_length=self.max_seq_len)

        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)
        role_ids = torch.tensor(encoded["role_ids"], dtype=torch.long)

        # Decision label
        decision = sample.get("decision", "direct_answer")
        decision_label = torch.tensor(1 if decision == "tool_call" else 0, dtype=torch.float)

        # Tool name label
        tool_info = sample.get("tool", {})
        tool_name = tool_info.get("name", "") if tool_info else ""
        tool_name_label = torch.tensor(
            self.tool_name_to_id.get(tool_name, -1), dtype=torch.long
        )

        # Slot presence labels
        slots = sample.get("slots", {})
        slot_presence = []
        for field in self.slot_fields:
            value = slots.get(field, "")
            slot_presence.append(1.0 if value else 0.0)
        slot_presence_labels = torch.tensor(slot_presence, dtype=torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "role_ids": role_ids,
            "decision_label": decision_label,
            "tool_name_label": tool_name_label,
            "slot_presence_labels": slot_presence_labels,
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
