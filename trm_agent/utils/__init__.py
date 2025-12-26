"""Utility functions for TRM Agent data processing."""

from .data_processor import (
    RawDataProcessor,
    process_raw_conversation,
    extract_slots,
    build_history,
    DEFAULT_SLOT_FIELDS,
)
from .ddp import (
    is_main_process,
    get_rank,
    get_world_size,
    is_distributed,
    barrier,
    all_reduce_sum,
    gather_metrics,
)
from .logger import get_logger, logger

__all__ = [
    # Data processing
    "RawDataProcessor",
    "process_raw_conversation",
    "extract_slots",
    "build_history",
    "DEFAULT_SLOT_FIELDS",
    # Logging
    "get_logger",
    "logger",
    # DDP utilities
    "is_main_process",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "barrier",
    "all_reduce_sum",
    "gather_metrics",
]
