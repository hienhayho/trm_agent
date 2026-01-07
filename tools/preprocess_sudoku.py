"""Preprocess Sudoku dataset from sapientinc/sudoku-extreme.

Downloads the dataset from HuggingFace and converts to numpy format
with optional subsampling and augmentation.

Usage:
    uv run python tools/preprocess_sudoku.py --output-dir data/sudoku

    # With subsampling and augmentation:
    uv run python tools/preprocess_sudoku.py \
        --output-dir data/sudoku \
        --subsample-size 10000 \
        --num-aug 8 \
        --min-difficulty 1500
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from trm_agent.utils import get_logger

logger = get_logger(__name__)


def shuffle_sudoku(
    board: np.ndarray,
    solution: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply random Sudoku-preserving transformations.

    Transformations that preserve Sudoku validity:
    - Digit permutation (1-9 → random permutation)
    - Transpose
    - Row band shuffle + row shuffle within bands
    - Column stack shuffle + column shuffle within stacks

    Args:
        board: [9, 9] input puzzle
        solution: [9, 9] solution
        rng: Random number generator

    Returns:
        Tuple of (transformed_board, transformed_solution)
    """
    # Digit mapping: permute 1-9, keep 0 (empty) unchanged
    digit_map = np.zeros(10, dtype=np.int64)
    digit_map[1:] = rng.permutation(np.arange(1, 10))

    # Transpose flag
    transpose = rng.random() < 0.5

    # Row permutation: shuffle the 3 bands, then rows within each band
    bands = rng.permutation(3)
    row_perm = np.concatenate([b * 3 + rng.permutation(3) for b in bands])

    # Column permutation: shuffle the 3 stacks, then columns within each stack
    stacks = rng.permutation(3)
    col_perm = np.concatenate([s * 3 + rng.permutation(3) for s in stacks])

    def transform(x: np.ndarray) -> np.ndarray:
        if transpose:
            x = x.T.copy()
        # Apply row/col permutation
        x = x[row_perm][:, col_perm]
        # Apply digit mapping
        return digit_map[x]

    return transform(board.copy()), transform(solution.copy())


def convert_subset(
    set_name: str,
    source_repo: str,
    output_dir: Path,
    subsample_size: Optional[int] = None,
    min_difficulty: Optional[int] = None,
    num_aug: int = 0,
    seed: int = 42,
):
    """Convert a subset (train/test) of the dataset.

    Args:
        set_name: "train" or "test"
        source_repo: HuggingFace repo ID
        output_dir: Output directory
        subsample_size: Number of samples to use (None = all)
        min_difficulty: Minimum difficulty rating
        num_aug: Number of augmentations per puzzle (train only)
        seed: Random seed
    """
    rng = np.random.default_rng(seed)

    # Download CSV from HuggingFace
    logger.info(f"Downloading {set_name}.csv from {source_repo}...")
    csv_path = hf_hub_download(source_repo, f"{set_name}.csv", repo_type="dataset")

    # Read CSV
    inputs = []
    labels = []

    logger.info(f"Loading {set_name} data...")
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            source, puzzle, solution, rating = row
            if min_difficulty is None or int(rating) >= min_difficulty:
                assert len(puzzle) == 81 and len(solution) == 81
                # Convert string to numpy array
                inp = np.frombuffer(
                    puzzle.replace(".", "0").encode(), dtype=np.uint8
                ).reshape(9, 9) - ord("0")
                lbl = np.frombuffer(solution.encode(), dtype=np.uint8).reshape(
                    9, 9
                ) - ord("0")
                inputs.append(inp)
                labels.append(lbl)

    logger.info(f"Loaded {len(inputs)} samples (after difficulty filter)")

    # Subsample if requested (for training set)
    if set_name == "train" and subsample_size is not None:
        if subsample_size < len(inputs):
            logger.info(f"Subsampling to {subsample_size} samples...")
            indices = rng.choice(len(inputs), size=subsample_size, replace=False)
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # Generate dataset with augmentation
    num_augments = num_aug if set_name == "train" else 0

    results = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [0],
        "group_indices": [0],
    }

    puzzle_id = 0
    example_id = 0

    logger.info(
        f"Processing {len(inputs)} puzzles with {num_augments} augmentations each..."
    )
    for orig_inp, orig_lbl in zip(tqdm(inputs, desc=f"Processing {set_name}"), labels):
        for aug_idx in range(1 + num_augments):
            if aug_idx == 0:
                inp, lbl = orig_inp, orig_lbl
            else:
                inp, lbl = shuffle_sudoku(orig_inp, orig_lbl, rng)

            results["inputs"].append(inp)
            results["labels"].append(lbl)
            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)

        results["group_indices"].append(puzzle_id)

    # Convert to numpy arrays
    def seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        assert np.all((arr >= 0) & (arr <= 9))
        # Shift values by 1: 0→1, 1→2, ..., 9→10
        # This makes PAD=0, empty=1, and digits 1-9 become 2-10
        return arr + 1

    results_np = {
        "inputs": seq_to_numpy(results["inputs"]),
        "labels": seq_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = {
        "seq_len": 81,
        "vocab_size": 11,  # PAD + "0"(empty) ... "9"
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": 1,
        "total_groups": len(results["group_indices"]) - 1,
        "mean_puzzle_examples": 1 + num_augments if set_name == "train" else 1,
        "total_puzzles": len(results["group_indices"]) - 1,
        "total_samples": len(results["inputs"]),
        "original_samples": len(inputs),
        "num_augmentations": num_augments,
        "sets": ["all"],
    }

    # Save
    save_dir = output_dir / set_name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving to {save_dir}...")

    # Save metadata
    with open(save_dir / "dataset.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save data arrays
    for k, v in results_np.items():
        np.save(save_dir / f"all__{k}.npy", v)

    # Save identifiers mapping (for visualization)
    with open(output_dir / "identifiers.json", "w") as f:
        json.dump(["<blank>"], f)

    logger.info(
        f"Saved {metadata['total_samples']} samples "
        f"({metadata['original_samples']} original + {num_augments}x augmentation)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Sudoku dataset from HuggingFace"
    )
    parser.add_argument(
        "--source-repo",
        type=str,
        default="sapientinc/sudoku-extreme",
        help="HuggingFace dataset repository",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sudoku",
        help="Output directory for preprocessed data",
    )
    parser.add_argument(
        "--subsample-size",
        type=int,
        default=None,
        help="Number of training samples to use (None = all)",
    )
    parser.add_argument(
        "--min-difficulty",
        type=int,
        default=None,
        help="Minimum difficulty rating to include",
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=0,
        help="Number of augmentations per training puzzle",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling and augmentation",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Sudoku Dataset Preprocessing")
    logger.info(f"  Source: {args.source_repo}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  Subsample size: {args.subsample_size or 'all'}")
    logger.info(f"  Min difficulty: {args.min_difficulty or 'none'}")
    logger.info(f"  Augmentations: {args.num_aug}")
    logger.info(f"  Seed: {args.seed}")
    logger.info("=" * 60)

    # Process train set
    convert_subset(
        "train",
        args.source_repo,
        output_dir,
        subsample_size=args.subsample_size,
        min_difficulty=args.min_difficulty,
        num_aug=args.num_aug,
        seed=args.seed,
    )

    # Process test set (no subsampling or augmentation)
    convert_subset(
        "test",
        args.source_repo,
        output_dir,
        subsample_size=None,
        min_difficulty=args.min_difficulty,
        num_aug=0,
        seed=args.seed,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
