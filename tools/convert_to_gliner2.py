"""Convert TRM dataset to GLiNER2 training format.

GLiNER2 expects InputExample format:
    InputExample(
        text="...",
        entities={"label": ["entity1", "entity2"], ...}
    )

This script converts TRM JSONL dataset to a pickle file with InputExamples.

Usage:
    uv run python tools/convert_to_gliner2.py data/train.jsonl data/gliner2_train.pkl
    uv run python tools/convert_to_gliner2.py data/test.jsonl data/gliner2_test.pkl --split test
"""

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Optional


def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def conversation_to_text(history: list[dict]) -> str:
    """Convert conversation history to plain text."""
    parts = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, str):
            parts.append(f"{role}: {content}")
        elif isinstance(content, dict):
            if "name" in content:
                parts.append(f"{role}: [tool: {content.get('name')}]")
            else:
                parts.append(f"{role}: {json.dumps(content, ensure_ascii=False)}")

    return "\n".join(parts)


def find_entity_in_text(text: str, value: str) -> bool:
    """Check if entity value exists in text."""
    if not value or not text:
        return False

    value_normalized = normalize_text(value.lower())
    text_lower = text.lower()

    return value_normalized in text_lower


def convert_sample_to_gliner2(sample: dict) -> Optional[dict]:
    """Convert a TRM sample to GLiNER2 InputExample format.

    Returns:
        Dict with 'text' and 'entities' keys, or None if no entities.
    """
    history = sample.get("history", [])
    slots = sample.get("slots", {})
    tool = sample.get("tool", {})

    # Get full conversation text
    text = conversation_to_text(history)
    if not text:
        return None

    # Collect entities by label
    entities: dict[str, list[str]] = {}

    # Extract slots
    for slot_name, slot_value in slots.items():
        if slot_value and isinstance(slot_value, str) and slot_value.strip():
            if find_entity_in_text(text, slot_value):
                if slot_name not in entities:
                    entities[slot_name] = []
                if slot_value not in entities[slot_name]:
                    entities[slot_name].append(slot_value)

    # Extract tool arguments
    if tool and isinstance(tool, dict):
        tool_args = tool.get("arguments", {})
        for arg_name, arg_value in tool_args.items():
            if arg_value and isinstance(arg_value, str) and arg_value.strip():
                if find_entity_in_text(text, arg_value):
                    if arg_name not in entities:
                        entities[arg_name] = []
                    if arg_value not in entities[arg_name]:
                        entities[arg_name].append(arg_value)

    if not entities:
        return None

    return {
        "text": text,
        "entities": entities,
    }


def convert_dataset(
    input_path: Path,
    output_path: Path,
    max_samples: Optional[int] = None,
) -> dict:
    """Convert TRM dataset to GLiNER2 format.

    Returns:
        Stats dict with conversion info.
    """
    samples = []
    total = 0
    converted = 0
    entity_counts: dict[str, int] = {}

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total += 1
            if max_samples and total > max_samples:
                break

            try:
                sample = json.loads(line)
                gliner2_sample = convert_sample_to_gliner2(sample)
                if gliner2_sample:
                    samples.append(gliner2_sample)
                    converted += 1

                    # Count entities
                    for label, values in gliner2_sample["entities"].items():
                        entity_counts[label] = entity_counts.get(label, 0) + len(values)
            except json.JSONDecodeError:
                continue

    # Save output as pickle (for InputExample compatibility)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(samples, f)

    # Also save as JSON for inspection
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    return {
        "total_samples": total,
        "converted_samples": converted,
        "entity_counts": entity_counts,
        "output_path": str(output_path),
        "json_path": str(json_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert TRM dataset to GLiNER2 training format"
    )
    parser.add_argument("input", type=Path, help="Input JSONL file")
    parser.add_argument("output", type=Path, help="Output pickle file for GLiNER2")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to convert",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "val"],
        help="Dataset split name (for logging)",
    )

    args = parser.parse_args()

    print(f"Converting {args.split} split: {args.input}")
    stats = convert_dataset(args.input, args.output, args.max_samples)

    print(f"\nConversion complete:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Converted samples: {stats['converted_samples']}")
    print(
        f"  Conversion rate: {stats['converted_samples'] / max(stats['total_samples'], 1) * 100:.1f}%"
    )
    print(f"\nEntity counts:")
    for label, count in sorted(stats["entity_counts"].items()):
        print(f"  {label}: {count}")
    print(f"\nOutput saved to:")
    print(f"  Pickle: {stats['output_path']}")
    print(f"  JSON:   {stats['json_path']}")


if __name__ == "__main__":
    main()
