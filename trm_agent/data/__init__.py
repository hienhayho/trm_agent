"""Data utilities for TRM training.

Note: Span extraction (slots/params) is handled by GLiNER2, not TRM.
TRM only handles decision classification and tool selection.
"""

from .tokenizer import TRMTokenizer, SPECIAL_TOKENS
from .dataset import TRMToolCallingDataset
from .collator import TRMCollator

__all__ = [
    "TRMTokenizer",
    "SPECIAL_TOKENS",
    "TRMToolCallingDataset",
    "TRMCollator",
]
