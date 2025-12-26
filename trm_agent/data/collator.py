"""Batch collation for TRM training.

Handles padding and batching of variable-length sequences.
"""

from typing import Any

import torch


class TRMCollator:
    """Collator for TRM tool-calling batches.

    Pads sequences to the maximum length in the batch and
    creates proper attention masks.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        max_seq_len: int = 2048,
    ):
        """Initialize collator.

        Args:
            pad_token_id: Token ID for padding
            max_seq_len: Maximum sequence length
        """
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

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

        # Stack slot presence labels
        num_slots = len(batch[0]["slot_presence_labels"])
        slot_presence_labels = torch.zeros((batch_size, num_slots), dtype=torch.float)

        # Fill in batch tensors
        for i, sample in enumerate(batch):
            seq_len = min(len(sample["input_ids"]), max_len)

            input_ids[i, :seq_len] = sample["input_ids"][:seq_len]
            attention_mask[i, :seq_len] = sample["attention_mask"][:seq_len]
            role_ids[i, :seq_len] = sample["role_ids"][:seq_len]

            decision_labels[i] = sample["decision_label"]
            tool_name_labels[i] = sample["tool_name_label"]
            slot_presence_labels[i] = sample["slot_presence_labels"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "role_ids": role_ids,
            "decision_labels": decision_labels,
            "tool_name_labels": tool_name_labels,
            "slot_presence_labels": slot_presence_labels,
        }


def create_dataloader(
    dataset: Any,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    collator: TRMCollator | None = None,
    pad_token_id: int = 0,
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
        distributed: Whether to use DistributedSampler for DDP

    Returns:
        DataLoader instance
    """
    if collator is None:
        collator = TRMCollator(pad_token_id=pad_token_id)

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
