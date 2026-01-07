"""Sudoku Dataset for TRM training.

Loads Sudoku puzzles from numpy files (preprocessed from sapientinc/sudoku-extreme).
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SudokuDataset(Dataset):
    """Dataset for Sudoku puzzles.

    Expects preprocessed numpy files in the format:
    - all__inputs.npy: [N, 81] input puzzles
    - all__labels.npy: [N, 81] solutions
    - dataset.json: metadata

    Values are 1-indexed (1-10 for digits 1-9, with 1 being empty/0 in original).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
    ):
        """Initialize dataset.

        Args:
            data_dir: Path to preprocessed data directory
            split: "train" or "test"
        """
        self.data_dir = Path(data_dir) / split
        self.split = split

        # Load data
        self.inputs = np.load(self.data_dir / "all__inputs.npy")
        self.labels = np.load(self.data_dir / "all__labels.npy")

        # Load metadata
        with open(self.data_dir / "dataset.json", "r") as f:
            self.metadata = json.load(f)

        assert len(self.inputs) == len(self.labels), "Input/label size mismatch"

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a sample.

        Returns:
            dict with:
            - input: [81] input puzzle values (1-10, where 1=empty)
            - target: [81] solution values (1-10)
            - given_mask: [81] True where cell was given (not empty)
        """
        inp = torch.from_numpy(self.inputs[idx].astype(np.int64))
        target = torch.from_numpy(self.labels[idx].astype(np.int64))

        # Given mask: cells that were provided in input (value > 1, since 1 = empty/0)
        given_mask = inp > 1

        return {
            "input": inp,
            "target": target,
            "given_mask": given_mask,
        }

    @property
    def vocab_size(self) -> int:
        """Vocabulary size (PAD + 10 values)."""
        return self.metadata.get("vocab_size", 11)

    @property
    def seq_len(self) -> int:
        """Sequence length (81 for 9x9 Sudoku)."""
        return self.metadata.get("seq_len", 81)


class SudokuDatasetFromCSV(Dataset):
    """Dataset for Sudoku puzzles loaded directly from CSV.

    For use when you have raw CSV files from sapientinc/sudoku-extreme.
    """

    def __init__(
        self,
        csv_path: str | Path,
        subsample_size: Optional[int] = None,
        min_difficulty: Optional[int] = None,
        num_aug: int = 0,
        seed: int = 42,
    ):
        """Initialize dataset.

        Args:
            csv_path: Path to CSV file
            subsample_size: Number of samples to use (None = all)
            min_difficulty: Minimum difficulty rating
            num_aug: Number of augmentations per puzzle
            seed: Random seed
        """
        import csv

        self.num_aug = num_aug
        self.rng = np.random.default_rng(seed)

        # Load puzzles
        puzzles = []
        solutions = []

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                source, puzzle, solution, rating = row
                if min_difficulty is None or int(rating) >= min_difficulty:
                    puzzles.append(puzzle.replace(".", "0"))
                    solutions.append(solution)

        # Subsample if requested
        if subsample_size is not None and subsample_size < len(puzzles):
            indices = self.rng.choice(len(puzzles), size=subsample_size, replace=False)
            puzzles = [puzzles[i] for i in indices]
            solutions = [solutions[i] for i in indices]

        # Convert to numpy arrays
        self.puzzles = np.array(
            [[int(c) for c in p] for p in puzzles], dtype=np.int64
        ).reshape(-1, 9, 9)
        self.solutions = np.array(
            [[int(c) for c in s] for s in solutions], dtype=np.int64
        ).reshape(-1, 9, 9)

        # Total samples = original + augmented
        self.base_size = len(self.puzzles)

    def __len__(self) -> int:
        return self.base_size * (1 + self.num_aug)

    def _shuffle_sudoku(
        self, board: np.ndarray, solution: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply random Sudoku-preserving transformations.

        Transformations:
        - Digit permutation (1-9 → random permutation)
        - Transpose (optional)
        - Row band shuffle + row shuffle within bands
        - Column stack shuffle + column shuffle within stacks
        """
        # Digit mapping: permute 1-9, keep 0 unchanged
        digit_map = np.zeros(10, dtype=np.int64)
        digit_map[1:] = self.rng.permutation(np.arange(1, 10))

        # Transpose flag
        transpose = self.rng.random() < 0.5

        # Row permutation: shuffle bands, then rows within bands
        bands = self.rng.permutation(3)
        row_perm = np.concatenate([b * 3 + self.rng.permutation(3) for b in bands])

        # Column permutation: shuffle stacks, then columns within stacks
        stacks = self.rng.permutation(3)
        col_perm = np.concatenate([s * 3 + self.rng.permutation(3) for s in stacks])

        def transform(x: np.ndarray) -> np.ndarray:
            if transpose:
                x = x.T
            # Apply row/col permutation
            x = x[row_perm][:, col_perm]
            # Apply digit mapping
            return digit_map[x]

        return transform(board.copy()), transform(solution.copy())

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a sample with optional augmentation."""
        base_idx = idx % self.base_size
        aug_idx = idx // self.base_size

        puzzle = self.puzzles[base_idx]
        solution = self.solutions[base_idx]

        # Apply augmentation if not the original
        if aug_idx > 0:
            puzzle, solution = self._shuffle_sudoku(puzzle, solution)

        # Flatten to [81] and shift values by 1 (0→1, 1→2, ..., 9→10)
        # This makes 1 = empty and 2-10 = digits 1-9
        inp = torch.from_numpy(puzzle.flatten() + 1)
        target = torch.from_numpy(solution.flatten() + 1)

        # Given mask: cells that were provided (not empty in original)
        given_mask = inp > 1

        return {
            "input": inp,
            "target": target,
            "given_mask": given_mask,
        }


def create_sudoku_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for Sudoku dataset."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
