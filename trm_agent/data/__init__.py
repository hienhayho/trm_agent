"""Data utilities for TRM training.

Note: Span extraction (slots/params) is handled by GLiNER2, not TRM.
TRM only handles decision classification and tool selection.
"""

from .tokenizer import TRMTokenizer, SPECIAL_TOKENS
from .dataset import TRMToolCallingDataset, load_intent_mapping
from .collator import TRMCollator
from .sudoku_dataset import SudokuDataset, SudokuDatasetFromCSV, create_sudoku_dataloader

__all__ = [
    "TRMTokenizer",
    "SPECIAL_TOKENS",
    "TRMToolCallingDataset",
    "TRMCollator",
    "load_intent_mapping",
    # Sudoku
    "SudokuDataset",
    "SudokuDatasetFromCSV",
    "create_sudoku_dataloader",
]
