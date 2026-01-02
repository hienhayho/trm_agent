"""
Convert raw conversation JSON files to TRM training dataset JSONL format.

Outputs:
    1. JSONL training file: Contains sub-conversations for TRM training
    2. Text file: Contains full conversations for tokenizer training (no duplicates)

Usage:
    uv run python tools/convert_raw_to_dataset.py --input <folder> --output <file> [options]

Example:
    uv run python tools/convert_raw_to_dataset.py --input data/raw --output data/dataset.jsonl --tools data/tools.json
    # Outputs: data/dataset.jsonl (training) + data/dataset_tokenizer.txt (tokenizer)

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


def extract_text_from_conversation(raw_data: list, tools: list) -> list[str]:
    """Extract text from a raw conversation for tokenizer training.

    Unlike training samples (which split into sub-conversations),
    this extracts each conversation once to avoid duplicates.

    Args:
        raw_data: Raw conversation data (list of turns)
        tools: Tool definitions

    Returns:
        List of text lines from this conversation
    """
    lines = []

    # Extract tool information (once per conversation)
    for tool in tools:
        if "function" in tool:
            func = tool["function"]
            if "name" in func:
                lines.append(func["name"])
            if "description" in func:
                lines.append(func["description"])
            # Extract parameter names and descriptions
            params = func.get("parameters", {}).get("properties", {})
            for param_name, param_info in params.items():
                lines.append(param_name)
                if "description" in param_info:
                    lines.append(param_info["description"])

    # Extract text from each turn
    for turn in raw_data:
        role = turn.get("role", "")
        content = turn.get("content", "")

        if isinstance(content, dict):
            # Extract think trace if present
            think = content.get("think", "")
            if think:
                lines.append(think)

            # Extract response
            response = content.get("response", "")
            if isinstance(response, str) and response:
                lines.append(response)
            elif isinstance(response, dict):
                # Tool call response
                lines.append(json.dumps(response, ensure_ascii=False))

            # Extract slot values
            for key in ["address", "phone", "device_number", "intent_of_user", "name", "contract_id"]:
                value = content.get(key, "")
                if value:
                    lines.append(str(value))
        elif isinstance(content, str) and content:
            lines.append(content)

    return lines


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
    all_tokenizer_lines = []  # Text for tokenizer training (no duplicates)
    processed_count = 0
    error_count = 0

    # Add tool info once for tokenizer (not per conversation)
    tools_added = False

    pbar = tqdm(json_files, desc="Processing files", unit="file")
    for json_file in pbar:
        try:
            raw_data = load_json_file(json_file)

            if not isinstance(raw_data, list):
                logger.warning(f"'{json_file}' does not contain a list. Skipping.")
                error_count += 1
                continue

            # Extract training samples (sub-conversations)
            samples = process_raw_conversation(
                raw_data=raw_data,
                tools=tools,
                slot_fields=slot_fields,
            )
            all_samples.extend(samples)

            # Extract text for tokenizer (full conversation, no duplicates)
            tokenizer_lines = extract_text_from_conversation(
                raw_data=raw_data,
                tools=tools if not tools_added else [],  # Only add tools once
            )
            all_tokenizer_lines.extend(tokenizer_lines)
            tools_added = True

            processed_count += 1
            pbar.set_postfix(samples=len(all_samples), errors=error_count)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse '{json_file}': {e}")
            error_count += 1
        except Exception as e:
            logger.error(f"Failed to process '{json_file}': {e}")
            error_count += 1

    pbar.close()

    # Prepend system prompt to each sample's history and tokenizer text
    if system_prompt:
        system_message = {"role": "system", "content": system_prompt}
        for sample in all_samples:
            sample["history"].insert(0, system_message)
        # Add system prompt to tokenizer text (once)
        all_tokenizer_lines.insert(0, system_prompt)
        logger.info(f"Prepended system prompt to {len(all_samples)} samples")

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save training JSONL
    save_jsonl(all_samples, output_file)

    # Save tokenizer text file (same name with _tokenizer.txt suffix)
    tokenizer_file = output_file.parent / (output_file.stem + "_tokenizer.txt")
    with open(tokenizer_file, "w", encoding="utf-8") as f:
        for line in all_tokenizer_lines:
            # Clean line (remove extra whitespace, ensure single line)
            clean_line = " ".join(line.split())
            if clean_line:
                f.write(clean_line + "\n")

    logger.info(
        f"Conversion complete: {processed_count} files, "
        f"{error_count} errors, {len(all_samples)} samples"
    )
    logger.info(f"  Training file: {output_file}")
    logger.info(f"  Tokenizer file: {tokenizer_file} ({len(all_tokenizer_lines)} lines)")


if __name__ == "__main__":
    main()
