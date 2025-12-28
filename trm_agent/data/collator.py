"""Batch collation for TRM training.

Handles padding and batching of variable-length sequences.
Uses unified field labels (slots + tool params).
"""

from typing import Any

import torch


class TRMCollator:
    """Collator for TRM tool-calling batches.

    Pads sequences to the maximum length in the batch and
    creates proper attention masks. Uses unified field labels.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        max_seq_len: int = 2048,
        num_unified_fields: int = 0,
        max_spans_per_field: int = 4,
    ):
        """Initialize collator.

        Args:
            pad_token_id: Token ID for padding
            max_seq_len: Maximum sequence length
            num_unified_fields: Number of unified fields (slots + tool_params)
            max_spans_per_field: Maximum number of valid spans per field
        """
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.num_unified_fields = num_unified_fields
        self.max_spans_per_field = max_spans_per_field

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Collate a batch of samples.

        Args:
            batch: List of samples from dataset

        Returns:
            Batched tensors with padding
        """
        if not batch:
            return {}

        # Get max sequence length in this batch
        max_len = min(
            max(len(sample["input_ids"]) for sample in batch),
            self.max_seq_len,
        )

        # Prepare batch tensors
        batch_size = len(batch)

        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        role_ids = torch.zeros((batch_size, max_len), dtype=torch.long)

        decision_labels = torch.zeros(batch_size, dtype=torch.float)
        tool_name_labels = torch.full((batch_size,), -1, dtype=torch.long)

        # Unified field tensors
        num_fields = len(batch[0]["unified_start_labels"])
        unified_start_labels = torch.full((batch_size, num_fields), -1, dtype=torch.long)
        unified_end_labels = torch.full((batch_size, num_fields), -1, dtype=torch.long)
        unified_presence_labels = torch.zeros((batch_size, num_fields), dtype=torch.float)

        # Multi-span labels
        max_spans = batch[0]["unified_all_start_labels"].shape[1]
        unified_all_start_labels = torch.full(
            (batch_size, num_fields, max_spans), -1, dtype=torch.long
        )
        unified_all_end_labels = torch.full(
            (batch_size, num_fields, max_spans), -1, dtype=torch.long
        )
        unified_span_weights = torch.zeros(
            (batch_size, num_fields, max_spans), dtype=torch.float
        )
        unified_num_spans = torch.zeros((batch_size, num_fields), dtype=torch.long)

        # Fill in batch tensors
        for i, sample in enumerate(batch):
            seq_len = min(len(sample["input_ids"]), max_len)

            input_ids[i, :seq_len] = sample["input_ids"][:seq_len]
            attention_mask[i, :seq_len] = sample["attention_mask"][:seq_len]
            role_ids[i, :seq_len] = sample["role_ids"][:seq_len]

            decision_labels[i] = sample["decision_label"]
            tool_name_labels[i] = sample["tool_name_label"]

            # Unified field labels
            unified_start_labels[i] = sample["unified_start_labels"]
            unified_end_labels[i] = sample["unified_end_labels"]
            unified_presence_labels[i] = sample["unified_presence_labels"]

            # Multi-span labels
            unified_all_start_labels[i] = sample["unified_all_start_labels"]
            unified_all_end_labels[i] = sample["unified_all_end_labels"]
            unified_span_weights[i] = sample["unified_span_weights"]
            unified_num_spans[i] = sample["unified_num_spans"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "role_ids": role_ids,
            "decision_labels": decision_labels,
            "tool_name_labels": tool_name_labels,
            # Unified field labels
            "unified_start_labels": unified_start_labels,
            "unified_end_labels": unified_end_labels,
            "unified_presence_labels": unified_presence_labels,
            "unified_all_start_labels": unified_all_start_labels,
            "unified_all_end_labels": unified_all_end_labels,
            "unified_span_weights": unified_span_weights,
            "unified_num_spans": unified_num_spans,
        }


def create_dataloader(
    dataset: Any,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    collator: TRMCollator | None = None,
    pad_token_id: int = 0,
    num_unified_fields: int = 0,
    max_spans_per_field: int = 4,
    distributed: bool = False,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for TRM training.

    Args:
        dataset: TRM dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        collator: Custom collator (optional)
        pad_token_id: Padding token ID
        num_unified_fields: Number of unified fields (slots + tool_params)
        max_spans_per_field: Maximum spans per field
        distributed: Whether to use DistributedSampler for DDP

    Returns:
        DataLoader instance
    """
    if collator is None:
        collator = TRMCollator(
            pad_token_id=pad_token_id,
            num_unified_fields=num_unified_fields,
            max_spans_per_field=max_spans_per_field,
        )

    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # Sampler handles shuffling

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        sampler=sampler,
    )
