"""Merge multiple TRM training data files into a single JSONL file.

Supports:
- JSONL files (.jsonl): One JSON object per line
- JSON files (.json): A list of valid samples

Usage:
    # Merge all files from a folder
    uv run python tools/merge_trm_data.py data/raw_files -o data/merged.jsonl

    # Recursive search in subfolders
    uv run python tools/merge_trm_data.py data/raw_files -o data/merged.jsonl --recursive

    # Only merge specific file types
    uv run python tools/merge_trm_data.py data/raw_files -o data/merged.jsonl --ext .jsonl

    # Shuffle output
    uv run python tools/merge_trm_data.py data/raw_files -o data/merged.jsonl --shuffle

    # Validate samples have required fields
    uv run python tools/merge_trm_data.py data/raw_files -o data/merged.jsonl --validate
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional

from trm_agent.utils import get_logger

logger = get_logger(__name__)

# Required fields for a valid TRM sample
TRM_REQUIRED_FIELDS = {"history", "decision"}


def load_jsonl_file(file_path: Path) -> list[dict]:
    """Load samples from a JSONL file (one JSON object per line)."""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"[{file_path}:{line_num}] JSON decode error: {e}")
    return samples


def load_json_file(file_path: Path) -> list[dict]:
    """Load samples from a JSON file (expects a list of samples)."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # Single sample, wrap in list
        return [data]
    else:
        logger.warning(f"[{file_path}] Unexpected JSON structure, skipping")
        return []


def load_file(file_path: Path) -> list[dict]:
    """Load samples from a file (auto-detect format)."""
    suffix = file_path.suffix.lower()

    if suffix == ".jsonl":
        return load_jsonl_file(file_path)
    elif suffix == ".json":
        return load_json_file(file_path)
    else:
        logger.warning(f"[{file_path}] Unsupported file format: {suffix}")
        return []


def validate_sample(sample: dict, required_fields: set[str]) -> bool:
    """Check if sample has all required fields."""
    if not isinstance(sample, dict):
        return False
    return all(field in sample for field in required_fields)


def find_data_files(
    input_dir: Path,
    recursive: bool = False,
    extensions: Optional[list[str]] = None,
) -> list[Path]:
    """Find all data files in directory."""
    if extensions is None:
        extensions = [".jsonl", ".json"]

    files = []
    pattern = "**/*" if recursive else "*"

    for ext in extensions:
        if recursive:
            files.extend(input_dir.glob(f"**/*{ext}"))
        else:
            files.extend(input_dir.glob(f"*{ext}"))

    # Sort for consistent ordering
    return sorted(files)


def merge_files(
    input_dir: Path,
    output_path: Path,
    recursive: bool = False,
    extensions: Optional[list[str]] = None,
    shuffle: bool = False,
    validate: bool = False,
    seed: int = 42,
    tools: Optional[Path] = None,
) -> dict:
    """Merge all data files from input directory into a single JSONL file.

    Args:
        input_dir: Directory containing data files
        output_path: Output JSONL file path
        recursive: Search subdirectories
        extensions: File extensions to include (default: [".jsonl", ".json"])
        shuffle: Shuffle samples before writing
        validate: Validate samples have required TRM fields
        seed: Random seed for shuffling

    Returns:
        Statistics dict with counts
    """
    stats = {
        "files_found": 0,
        "files_processed": 0,
        "files_empty": 0,
        "samples_total": 0,
        "samples_valid": 0,
        "samples_invalid": 0,
        "samples_written": 0,
    }
    if tools:
        logger.info(f"Using tools from: {tools}")
        with open(tools, "r", encoding="utf-8") as f:
            tools_data = json.load(f)
        logger.info(f"Loaded {len(tools_data)} tools")

    # Find all data files
    files = find_data_files(input_dir, recursive=recursive, extensions=extensions)
    stats["files_found"] = len(files)

    if not files:
        logger.error(f"No data files found in {input_dir}")
        return stats

    logger.info(f"Found {len(files)} data files")

    # Load all samples
    all_samples = []
    for i, file_path in enumerate(files):
        logger.info(f"[{i + 1}/{len(files)}] Loading: {file_path}")

        samples = load_file(file_path)
        stats["samples_total"] += len(samples)

        if not samples:
            stats["files_empty"] += 1
            continue

        stats["files_processed"] += 1

        # Validate if requested
        if validate:
            valid_samples = []
            for sample in samples:
                if validate_sample(sample, TRM_REQUIRED_FIELDS):
                    valid_samples.append(sample)
                    stats["samples_valid"] += 1
                else:
                    stats["samples_invalid"] += 1
            all_samples.extend(valid_samples)
            logger.info(f"  Loaded {len(valid_samples)}/{len(samples)} valid samples")
        else:
            all_samples.extend(samples)
            stats["samples_valid"] += len(samples)
            logger.info(f"  Loaded {len(samples)} samples")

    logger.info(f"Total samples collected: {len(all_samples)}")

    # Shuffle if requested
    if shuffle:
        logger.info(f"Shuffling samples (seed={seed})")
        random.seed(seed)
        random.shuffle(all_samples)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            if tools:
                sample["tools"] = tools_data
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            stats["samples_written"] += 1

    logger.info(f"Written {stats['samples_written']} samples to {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple TRM training data files into a single JSONL file"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing data files (.jsonl, .json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively",
    )
    parser.add_argument(
        "--ext",
        type=str,
        nargs="+",
        default=None,
        help="File extensions to include (default: .jsonl .json)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples before writing",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate samples have required TRM fields (history, decision)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--tools",
        type=Path,
        default=None,
        help="(Internal use) Path to tools json files.",
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return

    if not args.input_dir.is_dir():
        logger.error(f"Input path is not a directory: {args.input_dir}")
        return

    # Process extensions
    extensions = args.ext
    if extensions:
        # Ensure extensions start with dot
        extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]

    # Run merge
    logger.info("=" * 60)
    logger.info("TRM Data Merge")
    logger.info("=" * 60)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Recursive: {args.recursive}")
    logger.info(f"Extensions: {extensions or ['.jsonl', '.json']}")
    logger.info(f"Shuffle: {args.shuffle}")
    logger.info(f"Validate: {args.validate}")
    logger.info("=" * 60)

    stats = merge_files(
        input_dir=args.input_dir,
        output_path=args.output,
        recursive=args.recursive,
        extensions=extensions,
        shuffle=args.shuffle,
        validate=args.validate,
        seed=args.seed,
        tools=args.tools,
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Files found:       {stats['files_found']}")
    logger.info(f"Files processed:   {stats['files_processed']}")
    logger.info(f"Files empty:       {stats['files_empty']}")
    logger.info(f"Samples total:     {stats['samples_total']}")
    if args.validate:
        logger.info(f"Samples valid:     {stats['samples_valid']}")
        logger.info(f"Samples invalid:   {stats['samples_invalid']}")
    logger.info(f"Samples written:   {stats['samples_written']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
