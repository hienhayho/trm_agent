"""Distributed Data Parallel (DDP) utilities."""

import os

import torch


def is_main_process() -> bool:
    """Check if current process is the main process in distributed environment.

    Returns:
        True if this is the main process (rank 0), False otherwise.
    """
    # Check common distributed training environment variables
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        return int(rank) == 0
    except ValueError:
        return True


def get_rank() -> int:
    """Get the rank of current process.

    Returns:
        Process rank (0 for main process or non-distributed).
    """
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        return int(rank)
    except ValueError:
        return 0


def get_world_size() -> int:
    """Get the total number of processes.

    Returns:
        World size (1 for non-distributed).
    """
    world_size = os.environ.get("WORLD_SIZE", "1")
    try:
        return int(world_size)
    except ValueError:
        return 1


def is_distributed() -> bool:
    """Check if running in distributed mode.

    Returns:
        True if running with multiple processes.
    """
    return get_world_size() > 1


def barrier():
    """Synchronize all processes.

    Only effective when running in distributed mode with torch.distributed initialized.
    """
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Sum tensor across all processes.

    Args:
        tensor: Tensor to reduce (will be modified in-place)

    Returns:
        Reduced tensor (same as input, modified in-place)
    """
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor


def gather_metrics(metrics_dict: dict, device: torch.device) -> dict:
    """Gather and sum metrics from all processes.

    Args:
        metrics_dict: Dictionary of metric names to values (int or float)
        device: Device to create tensors on

    Returns:
        Dictionary with aggregated metrics
    """
    if not torch.distributed.is_initialized():
        return metrics_dict

    # Convert to tensor, reduce, convert back
    result = {}
    for key, value in metrics_dict.items():
        tensor = torch.tensor(value, dtype=torch.float64, device=device)
        all_reduce_sum(tensor)
        result[key] = tensor.item()

    return result
