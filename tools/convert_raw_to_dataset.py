"""
Convert raw conversation JSON files to TRM training dataset JSONL format.

Usage:
    uv run python tools/convert_raw_to_dataset.py --input <folder> --output <file> [options]

Example:
    uv run python tools/convert_raw_to_dataset.py --input data/raw --output data/dataset.jsonl --tools data/tools.json

    # With system prompt:
    uv run python tools/convert_raw_to_dataset.py --input data/raw --output data/dataset.jsonl --system prompts/system.txt

    # For testing with small dataset:
    uv run python tools/convert_raw_to_dataset.py --input data/raw --output data/test.jsonl --num-input-files 10
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from trm_agent.utils import process_raw_conversation, DEFAULT_SLOT_FIELDS, get_logger

logger = get_logger(__name__)


def load_json_file(file_path: Path) -> list | dict:
    """Load a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(samples: list[dict], output_path: Path) -> None:
    """Save samples to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def find_json_files(folder: Path) -> list[Path]:
    """Find all JSON files in a folder (non-recursive)."""
    return sorted(folder.glob("*.json"))


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw conversation JSON files to TRM training dataset JSONL format."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to folder containing raw JSON conversation files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--tools",
        type=str,
        default=None,
        help="Path to JSON file containing tool definitions",
    )
    parser.add_argument(
        "--slot-fields",
        type=str,
        nargs="+",
        default=None,
        help="List of slot field names to extract (default: address, phone, device_number, intent_of_user, name, contract_id)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for JSON files recursively in subdirectories",
    )
    parser.add_argument(
        "--num-input-files",
        type=int,
        default=None,
        help="Limit the number of input files to process (for testing with small dataset)",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Path to system prompt text file (prepended to history)",
    )

    args = parser.parse_args()

    input_folder = Path(args.input)
    output_file = Path(args.output)

    # Validate input folder
    if not input_folder.exists():
        logger.error(f"Input folder '{input_folder}' does not exist.")
        raise SystemExit(1)

    if not input_folder.is_dir():
        logger.error(f"'{input_folder}' is not a directory.")
        raise SystemExit(1)

    # Load tools if provided
    tools = []
    if args.tools:
        tools_path = Path(args.tools)
        if not tools_path.exists():
            logger.error(f"Tools file '{tools_path}' does not exist.")
            raise SystemExit(1)
        tools = load_json_file(tools_path)
        if not isinstance(tools, list):
            tools = [tools]
        logger.info(f"Loaded {len(tools)} tool definitions from '{tools_path}'")

    # Load system prompt if provided
    system_prompt = None
    if args.system:
        system_path = Path(args.system)
        if not system_path.exists():
            logger.error(f"System prompt file '{system_path}' does not exist.")
            raise SystemExit(1)
        with open(system_path, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
        logger.info(f"Loaded system prompt from '{system_path}' ({len(system_prompt)} chars)")

    # Use provided slot fields or default
    slot_fields = args.slot_fields or DEFAULT_SLOT_FIELDS

    # Find JSON files
    if args.recursive:
        json_files = sorted(input_folder.rglob("*.json"))
    else:
        json_files = find_json_files(input_folder)

    if not json_files:
        logger.error(f"No JSON files found in '{input_folder}'.")
        raise SystemExit(1)

    # Limit number of files if requested
    total_files = len(json_files)
    if args.num_input_files is not None and args.num_input_files < total_files:
        json_files = json_files[:args.num_input_files]
        logger.info(f"Limited to {args.num_input_files} files (from {total_files} total)")
    else:
        logger.info(f"Found {total_files} JSON files in '{input_folder}'")

    # Process files
    all_samples = []
    processed_count = 0
    error_count = 0

    pbar = tqdm(json_files, desc="Processing files", unit="file")
    for json_file in pbar:
        try:
            raw_data = load_json_file(json_file)

            if not isinstance(raw_data, list):
                logger.warning(f"'{json_file}' does not contain a list. Skipping.")
                error_count += 1
                continue

            samples = process_raw_conversation(
                raw_data=raw_data,
                tools=tools,
                slot_fields=slot_fields,
            )

            all_samples.extend(samples)
            processed_count += 1
            pbar.set_postfix(samples=len(all_samples), errors=error_count)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse '{json_file}': {e}")
            error_count += 1
        except Exception as e:
            logger.error(f"Failed to process '{json_file}': {e}")
            error_count += 1

    pbar.close()

    # Prepend system prompt to each sample's history
    if system_prompt:
        system_message = {"role": "system", "content": system_prompt}
        for sample in all_samples:
            sample["history"].insert(0, system_message)
        logger.info(f"Prepended system prompt to {len(all_samples)} samples")

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    save_jsonl(all_samples, output_file)

    logger.info(
        f"Conversion complete: {processed_count} files, "
        f"{error_count} errors, {len(all_samples)} samples -> {output_file}"
    )


if __name__ == "__main__":
    main()
