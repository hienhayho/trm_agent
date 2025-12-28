"""Utility functions for TRM Agent data processing."""

from .data_processor import (
    RawDataProcessor,
    process_raw_conversation,
    extract_slots,
    build_history,
    DEFAULT_SLOT_FIELDS,
)
from .ddp import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    is_distributed,
    barrier,
    all_reduce_sum,
    gather_metrics,
)
from .logger import get_logger, logger
from .span_utils import (
    SpanPosition,
    WeightedSpan,
    NO_SPAN,
    ROLE_WEIGHTS,
    find_span_in_text,
    find_all_spans_in_text,
    char_span_to_token_span,
    find_value_token_span,
    find_all_value_token_spans,
    find_best_value_token_span,
    # Fuzzy matching
    normalize_text,
    normalize_phone,
    normalize_for_matching,
    find_fuzzy_spans_in_text,
    find_all_value_token_spans_fuzzy,
)

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
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "barrier",
    "all_reduce_sum",
    "gather_metrics",
    # Span utilities
    "SpanPosition",
    "WeightedSpan",
    "NO_SPAN",
    "ROLE_WEIGHTS",
    "find_span_in_text",
    "find_all_spans_in_text",
    "char_span_to_token_span",
    "find_value_token_span",
    "find_all_value_token_spans",
    "find_best_value_token_span",
    # Fuzzy matching
    "normalize_text",
    "normalize_phone",
    "normalize_for_matching",
    "find_fuzzy_spans_in_text",
    "find_all_value_token_spans_fuzzy",
]
