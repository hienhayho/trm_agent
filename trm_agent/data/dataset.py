"""PyTorch Dataset for TRM Tool-Calling.

Loads JSONL dataset and prepares samples for training.

Note: Span extraction (slots/params) is handled by GLiNER2, not TRM.
TRM only handles decision classification and tool selection.
"""

import json
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from .tokenizer import TRMTokenizer


def load_intent_mapping(intent_file: str | Path) -> dict[str, int]:
    """Load intent mapping from JSON file.

    The JSON file should have format:
    {
        "intent_name_1": "description",
        "intent_name_2": "description",
        ...
    }

    Intent IDs are assigned alphabetically.

    Args:
        intent_file: Path to intent mapping JSON file

    Returns:
        Dictionary mapping intent name to ID
    """
    intent_file = Path(intent_file)
    if not intent_file.exists():
        raise FileNotFoundError(f"Intent mapping file not found: {intent_file}")

    with open(intent_file, "r", encoding="utf-8") as f:
        intent_data = json.load(f)

    # Sort keys for consistent ID assignment
    intent_names = sorted(intent_data.keys())
    return {name: idx for idx, name in enumerate(intent_names)}


class TRMToolCallingDataset(Dataset):
    """Dataset for TRM tool-calling training.

    Each sample contains:
    - Tokenized conversation history
    - Decision label (tool_call=1, direct_answer=0)
    - Tool name label (tool index or -1 for direct_answer)
    - Intent label (intent index or -1 for unknown)

    Note: Span extraction (slots/params) is handled by GLiNER2.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: TRMTokenizer,
        max_seq_len: int = 2048,
        tool_name_to_id: Optional[dict[str, int]] = None,
        intent_to_id: Optional[dict[str, int]] = None,
    ):
        """Initialize dataset.

        Args:
            data_path: Path to JSONL dataset file
            tokenizer: TRM tokenizer instance
            max_seq_len: Maximum sequence length
            tool_name_to_id: Mapping from tool name to tool ID
            intent_to_id: Mapping from intent name to intent ID
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tool_name_to_id = tool_name_to_id or {}
        self.intent_to_id = intent_to_id or {}

        # Load samples
        self.samples = self._load_samples()

        # Build tool name mapping if not provided
        if not self.tool_name_to_id:
            self._build_tool_name_mapping()

        # Build intent mapping if not provided but intents exist in data
        if not self.intent_to_id:
            self._build_intent_mapping()

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
            for tool in sample.get("tools", []):
                if "function" in tool:
                    tool_names.add(tool["function"]["name"])
            tool_info = sample.get("tool", {})
            if tool_info and "name" in tool_info:
                tool_names.add(tool_info["name"])

        self.tool_name_to_id = {name: idx for idx, name in enumerate(sorted(tool_names))}

    def _build_intent_mapping(self):
        """Build mapping from intent names to IDs from dataset."""
        intent_names = set()
        for sample in self.samples:
            intent = sample.get("intent")
            if intent and isinstance(intent, str) and intent.strip():
                intent_names.add(intent.strip())

        if intent_names:
            self.intent_to_id = {name: idx for idx, name in enumerate(sorted(intent_names))}

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
        - intent_label: Intent ID (or -1 if no intent)
        """
        sample = self.samples[idx]

        # Encode conversation history
        history = sample.get("history", [])
        encoded = self.tokenizer.encode_conversation(
            history, max_length=self.max_seq_len
        )

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

        # Intent label
        intent = sample.get("intent", "")
        if intent and isinstance(intent, str):
            intent = intent.strip()
        intent_label = torch.tensor(
            self.intent_to_id.get(intent, -1) if intent else -1, dtype=torch.long
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "role_ids": role_ids,
            "decision_label": decision_label,
            "tool_name_label": tool_name_label,
            "intent_label": intent_label,
        }

    def get_tool_names(self) -> list[str]:
        """Get list of tool names in order of their IDs."""
        return sorted(self.tool_name_to_id.keys(), key=lambda x: self.tool_name_to_id[x])

    def get_intent_names(self) -> list[str]:
        """Get list of intent names in order of their IDs."""
        return sorted(self.intent_to_id.keys(), key=lambda x: self.intent_to_id[x])

    def get_label_statistics(self) -> dict[str, Any]:
        """Get statistics about label distribution including intent."""
        num_tool_call = sum(
            1 for s in self.samples if s.get("decision") == "tool_call"
        )
        num_direct = len(self.samples) - num_tool_call

        # Intent distribution
        intent_counts: dict[str, int] = {}
        num_with_intent = 0
        for sample in self.samples:
            intent = sample.get("intent", "")
            if intent and isinstance(intent, str) and intent.strip():
                intent = intent.strip()
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
                num_with_intent += 1

        # Sort by count (descending) for display
        sorted_intents = sorted(intent_counts.items(), key=lambda x: -x[1])

        return {
            "total_samples": len(self.samples),
            "tool_call": num_tool_call,
            "direct_answer": num_direct,
            "tool_call_ratio": num_tool_call / len(self.samples) if self.samples else 0,
            "num_tools": len(self.tool_name_to_id),
            "tool_names": list(self.tool_name_to_id.keys()),
            # Intent statistics
            "num_intents": len(self.intent_to_id),
            "samples_with_intent": num_with_intent,
            "intent_coverage": num_with_intent / len(self.samples) if self.samples else 0,
            "intent_distribution": sorted_intents,
            "intent_names": list(self.intent_to_id.keys()),
        }
