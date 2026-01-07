"""Validate GLiNER2 training data - check if labels exist in input text.

Since GLiNER2 is a span extraction model, all entity values MUST exist in the
input text. This script analyzes TRM JSONL files to find samples where
slot/argument values don't appear in the conversation text.

Usage:
    # Analyze single file
    uv run python tools/validate_gliner2_data.py data/train.jsonl

    # Analyze multiple files
    uv run python tools/validate_gliner2_data.py data/train1.jsonl data/train2.jsonl

    # Save detailed report
    uv run python tools/validate_gliner2_data.py data/train.jsonl --output report.json

    # Show samples with missing values
    uv run python tools/validate_gliner2_data.py data/train.jsonl --show-missing 10

    # Only check tool arguments (skip slots)
    uv run python tools/validate_gliner2_data.py data/train.jsonl --args-only
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from trm_agent.utils import get_logger

logger = get_logger(__name__)


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
    """Check if entity value exists in text (case-insensitive)."""
    if not value or not text:
        return False

    value_normalized = normalize_text(value.lower())
    text_lower = text.lower()

    return value_normalized in text_lower


def analyze_sample(sample: dict, sample_idx: int, args_only: bool = False) -> dict:
    """Analyze a single sample for missing values.

    Args:
        sample: Sample dict from JSONL
        sample_idx: Line number in file
        args_only: If True, only check tool arguments (skip slots)

    Returns:
        Dict with analysis results:
        - sample_idx: sample index
        - decision: tool_call or direct_answer
        - total_values: total number of values (slots + args)
        - found_values: values found in text
        - missing_values: values NOT found in text
        - text_preview: first 200 chars of conversation
    """
    history = sample.get("history", [])
    slots = sample.get("slots", {})
    tool = sample.get("tool", {})
    decision = sample.get("decision", "unknown")

    # Get full conversation text
    text = conversation_to_text(history)

    # Collect all values and check existence
    found_values = {}
    missing_values = {}

    # Check slots (skip if args_only)
    if not args_only:
        for slot_name, slot_value in slots.items():
            if slot_value and isinstance(slot_value, str) and slot_value.strip():
                if find_entity_in_text(text, slot_value):
                    found_values[f"slot:{slot_name}"] = slot_value
                else:
                    missing_values[f"slot:{slot_name}"] = slot_value

    # Check tool arguments
    if tool and isinstance(tool, dict):
        tool_args = tool.get("arguments", {})
        for arg_name, arg_value in tool_args.items():
            if arg_value and isinstance(arg_value, str) and arg_value.strip():
                if find_entity_in_text(text, arg_value):
                    found_values[f"arg:{arg_name}"] = arg_value
                else:
                    missing_values[f"arg:{arg_name}"] = arg_value

    return {
        "sample_idx": sample_idx,
        "decision": decision,
        "tool_name": tool.get("name") if tool else None,
        "total_values": len(found_values) + len(missing_values),
        "found_count": len(found_values),
        "missing_count": len(missing_values),
        "found_values": found_values,
        "missing_values": missing_values,
        "text_preview": text[:300] + "..." if len(text) > 300 else text,
        "text_length": len(text),
    }


def analyze_file(
    jsonl_path: Path,
    max_samples: Optional[int] = None,
    args_only: bool = False,
) -> dict:
    """Analyze a JSONL file for missing values.

    Args:
        jsonl_path: Path to JSONL file
        max_samples: Maximum samples to analyze (None = all)
        args_only: If True, only check tool arguments (skip slots)

    Returns:
        Dict with aggregated statistics and per-sample details.
    """
    results = {
        "file": str(jsonl_path),
        "total_samples": 0,
        "samples_with_values": 0,
        "samples_all_found": 0,
        "samples_some_missing": 0,
        "samples_all_missing": 0,
        "samples_no_values": 0,
        "total_values": 0,
        "total_found": 0,
        "total_missing": 0,
        "missing_by_field": defaultdict(int),
        "missing_value_examples": defaultdict(list),
        "samples_with_missing": [],
    }

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            if max_samples and results["total_samples"] >= max_samples:
                break

            results["total_samples"] += 1

            try:
                sample = json.loads(line)
                analysis = analyze_sample(sample, line_num, args_only=args_only)

                results["total_values"] += analysis["total_values"]
                results["total_found"] += analysis["found_count"]
                results["total_missing"] += analysis["missing_count"]

                if analysis["total_values"] == 0:
                    results["samples_no_values"] += 1
                else:
                    results["samples_with_values"] += 1

                    if analysis["missing_count"] == 0:
                        results["samples_all_found"] += 1
                    elif analysis["found_count"] == 0:
                        results["samples_all_missing"] += 1
                        results["samples_with_missing"].append(analysis)
                    else:
                        results["samples_some_missing"] += 1
                        results["samples_with_missing"].append(analysis)

                    # Track missing fields
                    for field, value in analysis["missing_values"].items():
                        results["missing_by_field"][field] += 1
                        # Keep up to 3 examples per field
                        if len(results["missing_value_examples"][field]) < 3:
                            results["missing_value_examples"][field].append({
                                "sample_idx": analysis["sample_idx"],
                                "value": value,
                            })

            except json.JSONDecodeError as e:
                logger.warning(f"[Line {line_num}] JSON decode error: {e}")
                continue

    # Convert defaultdicts to regular dicts for JSON serialization
    results["missing_by_field"] = dict(results["missing_by_field"])
    results["missing_value_examples"] = dict(results["missing_value_examples"])

    return results


def print_report(results: dict, show_missing: int = 0) -> None:
    """Print analysis report to console."""
    print("\n" + "=" * 70)
    print(f"GLiNER2 Data Validation Report: {results['file']}")
    print("=" * 70)

    print(f"\n{'SUMMARY':^70}")
    print("-" * 70)
    print(f"Total samples:              {results['total_samples']:>10}")
    print(f"Samples with values:        {results['samples_with_values']:>10}")
    print(f"Samples without values:     {results['samples_no_values']:>10}")

    print(f"\n{'VALUE ANALYSIS':^70}")
    print("-" * 70)
    print(f"Total values (slots+args):  {results['total_values']:>10}")
    print(f"Values found in text:       {results['total_found']:>10}")
    print(f"Values NOT in text:         {results['total_missing']:>10}")

    if results['total_values'] > 0:
        found_pct = results['total_found'] / results['total_values'] * 100
        missing_pct = results['total_missing'] / results['total_values'] * 100
        print(f"Found percentage:           {found_pct:>9.1f}%")
        print(f"Missing percentage:         {missing_pct:>9.1f}%")

    print(f"\n{'SAMPLE BREAKDOWN':^70}")
    print("-" * 70)
    print(f"All values found:           {results['samples_all_found']:>10} (usable for training)")
    print(f"Some values missing:        {results['samples_some_missing']:>10} (partial extraction)")
    print(f"All values missing:         {results['samples_all_missing']:>10} (will be skipped)")

    if results['missing_by_field']:
        print(f"\n{'MISSING VALUES BY FIELD':^70}")
        print("-" * 70)
        sorted_fields = sorted(
            results['missing_by_field'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for field, count in sorted_fields:
            examples = results['missing_value_examples'].get(field, [])
            example_values = [e['value'][:30] for e in examples[:2]]
            example_str = ", ".join(f'"{v}"' for v in example_values)
            print(f"  {field:<30} {count:>5}x  (e.g., {example_str})")

    # Show detailed missing samples
    if show_missing > 0 and results['samples_with_missing']:
        print(f"\n{'SAMPLES WITH MISSING VALUES':^70}")
        print("-" * 70)
        for i, sample in enumerate(results['samples_with_missing'][:show_missing]):
            print(f"\n[Sample {sample['sample_idx']}] decision={sample['decision']}, tool={sample['tool_name']}")
            print(f"  Found ({sample['found_count']}): {list(sample['found_values'].keys())}")
            print(f"  Missing ({sample['missing_count']}):")
            for field, value in sample['missing_values'].items():
                print(f"    - {field}: \"{value}\"")
            print(f"  Text preview: {sample['text_preview'][:150]}...")

    print("\n" + "=" * 70)

    # Summary verdict
    if results['samples_with_values'] > 0:
        usable_pct = results['samples_all_found'] / results['samples_with_values'] * 100
        print(f"\nVERDICT: {usable_pct:.1f}% of samples with values are fully usable for GLiNER2 training")
        if usable_pct < 80:
            print("WARNING: High percentage of samples have values not in text!")
            print("         These values cannot be extracted by GLiNER2 (span extraction).")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate GLiNER2 training data - check if labels exist in input text"
    )
    parser.add_argument(
        "files",
        type=Path,
        nargs="+",
        help="Path(s) to TRM JSONL file(s) to analyze",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save detailed report to JSON file",
    )
    parser.add_argument(
        "--show-missing",
        type=int,
        default=5,
        help="Number of samples with missing values to show (default: 5)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to analyze per file (default: all)",
    )
    parser.add_argument(
        "--args-only",
        action="store_true",
        help="Only check tool arguments (skip slots)",
    )

    args = parser.parse_args()

    all_results = []

    for file_path in args.files:
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            continue

        logger.info(f"Analyzing: {file_path}" + (" (args only)" if args.args_only else ""))
        results = analyze_file(file_path, max_samples=args.max_samples, args_only=args.args_only)
        all_results.append(results)
        print_report(results, show_missing=args.show_missing)

    # Aggregate results if multiple files
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("AGGREGATED RESULTS (ALL FILES)")
        print("=" * 70)

        totals = {
            "total_samples": sum(r["total_samples"] for r in all_results),
            "samples_with_values": sum(r["samples_with_values"] for r in all_results),
            "samples_all_found": sum(r["samples_all_found"] for r in all_results),
            "samples_some_missing": sum(r["samples_some_missing"] for r in all_results),
            "samples_all_missing": sum(r["samples_all_missing"] for r in all_results),
            "total_values": sum(r["total_values"] for r in all_results),
            "total_found": sum(r["total_found"] for r in all_results),
            "total_missing": sum(r["total_missing"] for r in all_results),
        }

        print(f"Total samples:              {totals['total_samples']:>10}")
        print(f"Samples with values:        {totals['samples_with_values']:>10}")
        print(f"All values found:           {totals['samples_all_found']:>10}")
        print(f"Some/all values missing:    {totals['samples_some_missing'] + totals['samples_all_missing']:>10}")

        if totals['samples_with_values'] > 0:
            usable_pct = totals['samples_all_found'] / totals['samples_with_values'] * 100
            print(f"\nOverall usable for training: {usable_pct:.1f}%")
        print("=" * 70 + "\n")

    # Save detailed report
    if args.output:
        report = {
            "files": [str(f) for f in args.files],
            "results": all_results,
        }
        # Remove samples_with_missing for cleaner output (can be large)
        for r in report["results"]:
            r["samples_with_missing_count"] = len(r["samples_with_missing"])
            r["samples_with_missing"] = r["samples_with_missing"][:20]  # Keep only first 20

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed report saved to: {args.output}")


if __name__ == "__main__":
    main()
