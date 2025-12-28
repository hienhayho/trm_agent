"""PyTorch Dataset for TRM Tool-Calling.

Loads JSONL dataset and prepares samples for training.
Uses unified field extraction (slots + tool params) with decision+tool masking.
"""

import json
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from trm_agent.utils import find_all_value_token_spans_fuzzy
from .tokenizer import TRMTokenizer


class TRMToolCallingDataset(Dataset):
    """Dataset for TRM tool-calling training.

    Uses unified field extraction where:
    - slot_fields: Always extracted (for both direct_answer and tool_call)
    - tool_param_fields: Only for tool_call, masked by tool-specific mask
    - unified_fields = slot_fields + tool_param_fields (deduplicated)

    Each sample contains:
    - Tokenized conversation history
    - Decision label (tool_call=1, direct_answer=0)
    - Tool information (name, arguments) if tool_call
    - Unified field labels (spans + presence) with masking info
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: TRMTokenizer,
        max_seq_len: int = 2048,
        tool_name_to_id: Optional[dict[str, int]] = None,
        slot_fields: Optional[list[str]] = None,
        tool_param_fields: Optional[list[str]] = None,
        max_spans_per_field: int = 4,
    ):
        """Initialize dataset.

        Args:
            data_path: Path to JSONL dataset file
            tokenizer: TRM tokenizer instance
            max_seq_len: Maximum sequence length
            tool_name_to_id: Mapping from tool name to tool ID
            slot_fields: Content slot fields (always extracted)
            tool_param_fields: Tool param fields (auto-collected if None, deduplicated)
            max_spans_per_field: Maximum spans per field for multi-span support
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tool_name_to_id = tool_name_to_id or {}
        self.max_spans_per_field = max_spans_per_field
        # Default slot fields - only extractable fields (not generated/summarized)
        self.slot_fields = slot_fields or [
            "address",
            "phone",
            "device_number",
            # "intent_of_user",  # NOT TRAINED: semantic summary, not extractable
            "name",
            # "contract_id",     # NOT TRAINED: looked up/generated, not in text
        ]

        # Load samples
        self.samples = self._load_samples()

        # Build tool name mapping if not provided
        if not self.tool_name_to_id:
            self._build_tool_name_mapping()

        # Build tool param fields and masks
        self.tool_param_fields: list[str] = []
        self.param_name_to_idx: dict[str, int] = {}
        self.tool_param_mask: dict[str, list[int]] = {}  # tool_name -> param mask
        self._build_tool_param_mapping(tool_param_fields)

        # Build unified fields (slots + unique tool_params)
        self.unified_fields = self.slot_fields + self.tool_param_fields
        self.num_slots = len(self.slot_fields)
        self.num_tool_params = len(self.tool_param_fields)
        self.num_unified_fields = len(self.unified_fields)

        # Build unified tool mask (decision + tool masking)
        self.unified_tool_mask: dict[str, list[int]] = {}
        self._build_unified_tool_mask()

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

    def _build_tool_param_mapping(self, external_fields: Optional[list[str]] = None):
        """Build tool param fields and per-tool masks.

        Removes duplicates with slot_fields (slot behavior subsumes tool param).
        """
        slot_set = set(self.slot_fields)

        if external_fields:
            # Use provided fields, remove duplicates with slots
            unique_params = [f for f in external_fields if f not in slot_set]
            self.tool_param_fields = sorted(set(unique_params))
        else:
            # Auto-collect from all tools
            all_params: set[str] = set()
            for sample in self.samples:
                for tool in sample.get("tools", []):
                    func = tool.get("function", {})
                    params = func.get("parameters", {}).get("properties", {})
                    all_params.update(params.keys())

            # Remove duplicates with slots
            unique_params = [p for p in all_params if p not in slot_set]
            self.tool_param_fields = sorted(unique_params)

        self.param_name_to_idx = {
            name: idx for idx, name in enumerate(self.tool_param_fields)
        }

        # Build per-tool param mask (which params each tool uses)
        tool_params: dict[str, set[str]] = {}
        for sample in self.samples:
            for tool in sample.get("tools", []):
                func = tool.get("function", {})
                tool_name = func.get("name", "")
                params = func.get("parameters", {}).get("properties", {})
                if tool_name:
                    if tool_name not in tool_params:
                        tool_params[tool_name] = set()
                    tool_params[tool_name].update(params.keys())

        for tool_name in self.tool_name_to_id.keys():
            params = tool_params.get(tool_name, set())
            mask = [1 if p in params else 0 for p in self.tool_param_fields]
            self.tool_param_mask[tool_name] = mask

    def _build_unified_tool_mask(self):
        """Build unified mask for each tool (slots always 1, params tool-specific).

        For direct_answer: only slots valid
        For tool_call: slots + tool-specific params valid
        """
        for tool_name in self.tool_name_to_id.keys():
            param_mask = self.tool_param_mask.get(tool_name, [0] * self.num_tool_params)
            # slots=1, params=tool-specific
            unified_mask = [1] * self.num_slots + param_mask
            self.unified_tool_mask[tool_name] = unified_mask

    def get_slot_only_mask(self) -> list[int]:
        """Get mask for direct_answer: slots=1, params=0."""
        return [1] * self.num_slots + [0] * self.num_tool_params

    def get_tool_param_mask_tensor(self) -> torch.Tensor:
        """Get tool-param mask as tensor [num_tools, num_tool_params].

        Returns:
            Tensor where mask[tool_idx, param_idx] = 1 if tool uses that param
        """
        num_tools = len(self.tool_name_to_id)
        mask = torch.zeros((num_tools, self.num_tool_params), dtype=torch.float)

        for tool_name, tool_idx in self.tool_name_to_id.items():
            if tool_name in self.tool_param_mask:
                mask[tool_idx] = torch.tensor(
                    self.tool_param_mask[tool_name], dtype=torch.float
                )

        return mask

    def get_unified_tool_mask_tensor(self) -> torch.Tensor:
        """Get unified mask as tensor [num_tools, num_unified_fields].

        For use with tool_call samples (slots always 1, params tool-specific).
        """
        num_tools = len(self.tool_name_to_id)
        mask = torch.zeros((num_tools, self.num_unified_fields), dtype=torch.float)

        for tool_name, tool_idx in self.tool_name_to_id.items():
            if tool_name in self.unified_tool_mask:
                mask[tool_idx] = torch.tensor(
                    self.unified_tool_mask[tool_name], dtype=torch.float
                )

        return mask

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

        Unified field labels:
        - unified_start_labels: Start positions [num_unified_fields]
        - unified_end_labels: End positions [num_unified_fields]
        - unified_presence_labels: Presence flags [num_unified_fields]
        - unified_all_start_labels: All valid starts [num_unified_fields, max_spans]
        - unified_all_end_labels: All valid ends [num_unified_fields, max_spans]
        - unified_span_weights: Weights [num_unified_fields, max_spans]
        - unified_num_spans: Number of valid spans [num_unified_fields]
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
        role_ids_list = role_ids.tolist()

        # Decision label
        decision = sample.get("decision", "direct_answer")
        decision_label = torch.tensor(1 if decision == "tool_call" else 0, dtype=torch.float)

        # Tool name label
        tool_info = sample.get("tool", {})
        tool_name = tool_info.get("name", "") if tool_info else ""
        tool_name_label = torch.tensor(
            self.tool_name_to_id.get(tool_name, -1), dtype=torch.long
        )

        # Initialize unified field tensors
        unified_start_labels = torch.full((self.num_unified_fields,), -1, dtype=torch.long)
        unified_end_labels = torch.full((self.num_unified_fields,), -1, dtype=torch.long)
        unified_presence_labels = torch.zeros((self.num_unified_fields,), dtype=torch.float)

        unified_all_start_labels = torch.full(
            (self.num_unified_fields, self.max_spans_per_field), -1, dtype=torch.long
        )
        unified_all_end_labels = torch.full(
            (self.num_unified_fields, self.max_spans_per_field), -1, dtype=torch.long
        )
        unified_span_weights = torch.zeros(
            (self.num_unified_fields, self.max_spans_per_field), dtype=torch.float
        )
        unified_num_spans = torch.zeros((self.num_unified_fields,), dtype=torch.long)

        # === Extract slot values (always, from sample["slots"]) ===
        slots = sample.get("slots", {})
        for slot_idx, field in enumerate(self.slot_fields):
            value = slots.get(field, "")
            if value:
                unified_presence_labels[slot_idx] = 1.0

                # Use fuzzy matching with field type hint
                weighted_spans = find_all_value_token_spans_fuzzy(
                    value, full_text, token_offsets,
                    role_ids=role_ids_list,
                    field_type=field,  # Pass field name for specialized normalization
                )
                if weighted_spans:
                    best_span = weighted_spans[0]
                    unified_start_labels[slot_idx] = best_span.start
                    unified_end_labels[slot_idx] = best_span.end

                    num_spans = min(len(weighted_spans), self.max_spans_per_field)
                    unified_num_spans[slot_idx] = num_spans
                    for span_idx, ws in enumerate(weighted_spans[:num_spans]):
                        unified_all_start_labels[slot_idx, span_idx] = ws.start
                        unified_all_end_labels[slot_idx, span_idx] = ws.end
                        unified_span_weights[slot_idx, span_idx] = ws.weight

        # === Extract tool param values (only for tool_call) ===
        if tool_info and decision == "tool_call" and self.num_tool_params > 0:
            arguments = tool_info.get("arguments", {})
            for param_name, param_value in arguments.items():
                if param_name in self.param_name_to_idx:
                    param_idx = self.param_name_to_idx[param_name]
                    unified_idx = self.num_slots + param_idx  # Offset by slots

                    if param_value:
                        unified_presence_labels[unified_idx] = 1.0

                        param_value_str = (
                            str(param_value)
                            if not isinstance(param_value, str)
                            else param_value
                        )

                        # Use fuzzy matching with field type hint
                        weighted_spans = find_all_value_token_spans_fuzzy(
                            param_value_str, full_text, token_offsets,
                            role_ids=role_ids_list,
                            field_type=param_name,  # Pass param name for specialized normalization
                        )
                        if weighted_spans:
                            best_span = weighted_spans[0]
                            unified_start_labels[unified_idx] = best_span.start
                            unified_end_labels[unified_idx] = best_span.end

                            num_spans = min(len(weighted_spans), self.max_spans_per_field)
                            unified_num_spans[unified_idx] = num_spans
                            for span_idx, ws in enumerate(weighted_spans[:num_spans]):
                                unified_all_start_labels[unified_idx, span_idx] = ws.start
                                unified_all_end_labels[unified_idx, span_idx] = ws.end
                                unified_span_weights[unified_idx, span_idx] = ws.weight

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "role_ids": role_ids,
            "decision_label": decision_label,
            "tool_name_label": tool_name_label,
            # Unified field labels
            "unified_start_labels": unified_start_labels,
            "unified_end_labels": unified_end_labels,
            "unified_presence_labels": unified_presence_labels,
            "unified_all_start_labels": unified_all_start_labels,
            "unified_all_end_labels": unified_all_end_labels,
            "unified_span_weights": unified_span_weights,
            "unified_num_spans": unified_num_spans,
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
            "num_slots": self.num_slots,
            "num_tool_params": self.num_tool_params,
            "num_unified_fields": self.num_unified_fields,
            "slot_fields": self.slot_fields,
            "tool_param_fields": self.tool_param_fields,
        }

    def get_extraction_stats(self, show_progress: bool = True) -> dict[str, Any]:
        """Compute extraction statistics for debugging.

        Args:
            show_progress: Whether to show tqdm progress bar

        Returns stats on how many values couldn't be found in the conversation.
        """
        stats = {
            "total_samples": len(self.samples),
            "slot_extraction": {field: {"total": 0, "found": 0, "missing": 0} for field in self.slot_fields},
            "param_extraction": {field: {"total": 0, "found": 0, "missing": 0} for field in self.tool_param_fields},
            "samples_with_missing_slots": 0,
            "samples_with_missing_params": 0,
        }

        iterator = range(len(self.samples))
        if show_progress:
            iterator = tqdm(iterator, desc="Computing extraction stats", leave=False)

        for idx in iterator:
            sample = self.samples[idx]
            item = self[idx]  # Get processed item with span labels

            # Check slots
            slots = sample.get("slots", {})
            sample_missing_slot = False
            for slot_idx, field in enumerate(self.slot_fields):
                value = slots.get(field, "")
                if value:
                    stats["slot_extraction"][field]["total"] += 1
                    if item["unified_start_labels"][slot_idx] >= 0:
                        stats["slot_extraction"][field]["found"] += 1
                    else:
                        stats["slot_extraction"][field]["missing"] += 1
                        sample_missing_slot = True

            if sample_missing_slot:
                stats["samples_with_missing_slots"] += 1

            # Check params (only for tool_call)
            decision = sample.get("decision", "direct_answer")
            tool_info = sample.get("tool", {})
            if decision == "tool_call" and tool_info:
                arguments = tool_info.get("arguments", {})
                sample_missing_param = False
                for param_name, param_value in arguments.items():
                    if param_name in self.param_name_to_idx and param_value:
                        param_idx = self.param_name_to_idx[param_name]
                        unified_idx = self.num_slots + param_idx

                        stats["param_extraction"][param_name]["total"] += 1
                        if item["unified_start_labels"][unified_idx] >= 0:
                            stats["param_extraction"][param_name]["found"] += 1
                        else:
                            stats["param_extraction"][param_name]["missing"] += 1
                            sample_missing_param = True

                if sample_missing_param:
                    stats["samples_with_missing_params"] += 1

        return stats

    def get_missing_examples(
        self,
        field_name: str,
        max_examples: int = 5,
        is_slot: bool = True,
        show_progress: bool = False,
    ) -> list[dict[str, str]]:
        """Get examples of values that couldn't be extracted.

        Args:
            field_name: Name of the field to check
            max_examples: Maximum number of examples to return
            is_slot: True if checking slot, False if checking param
            show_progress: Whether to show tqdm progress bar

        Returns:
            List of dicts with 'value', 'text_snippet', 'sample_idx'
        """
        examples = []

        iterator = range(len(self.samples))
        if show_progress:
            iterator = tqdm(iterator, desc=f"Finding examples for {field_name}", leave=False)

        for idx in iterator:
            if len(examples) >= max_examples:
                break

            sample = self.samples[idx]
            item = self[idx]

            if is_slot:
                if field_name not in self.slot_fields:
                    continue
                field_idx = self.slot_fields.index(field_name)
                value = sample.get("slots", {}).get(field_name, "")
                unified_idx = field_idx
            else:
                if field_name not in self.param_name_to_idx:
                    continue
                param_idx = self.param_name_to_idx[field_name]
                unified_idx = self.num_slots + param_idx

                tool_info = sample.get("tool", {})
                if not tool_info:
                    continue
                value = tool_info.get("arguments", {}).get(field_name, "")

            if not value:
                continue

            # Check if extraction failed
            if item["unified_start_labels"][unified_idx] < 0:
                # Get text snippet from conversation
                history = sample.get("history", [])
                text_parts = []
                for msg in history[-3:]:  # Last 3 messages
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        text_parts.append(content[:200])

                examples.append({
                    "sample_idx": idx,
                    "value": str(value)[:100],
                    "text_snippet": " | ".join(text_parts)[:300],
                })

        return examples

    def analyze_extraction_failures(
        self,
        max_examples_per_field: int = 3,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Analyze why extractions are failing.

        Args:
            max_examples_per_field: Maximum examples per problematic field
            show_progress: Whether to show tqdm progress bar

        Returns detailed analysis with examples for each problematic field.
        """
        stats = self.get_extraction_stats(show_progress=show_progress)
        analysis = {
            "summary": {},
            "problematic_fields": [],
            "examples": {},
        }

        # Analyze slots
        for field, field_stats in stats["slot_extraction"].items():
            if field_stats["total"] == 0:
                continue

            rate = field_stats["found"] / field_stats["total"]
            analysis["summary"][field] = {
                "type": "slot",
                "extraction_rate": rate,
                "missing": field_stats["missing"],
                "total": field_stats["total"],
            }

            if rate < 0.5:  # Problematic if less than 50%
                analysis["problematic_fields"].append(field)
                analysis["examples"][field] = self.get_missing_examples(
                    field, max_examples_per_field, is_slot=True
                )

        # Analyze params
        for field, field_stats in stats["param_extraction"].items():
            if field_stats["total"] == 0:
                continue

            rate = field_stats["found"] / field_stats["total"]
            analysis["summary"][field] = {
                "type": "param",
                "extraction_rate": rate,
                "missing": field_stats["missing"],
                "total": field_stats["total"],
            }

            if rate < 0.5:
                analysis["problematic_fields"].append(field)
                analysis["examples"][field] = self.get_missing_examples(
                    field, max_examples_per_field, is_slot=False
                )

        return analysis
