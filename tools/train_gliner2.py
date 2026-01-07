"""Train GLiNER2 model with LoRA adapters.

Fine-tunes a pre-trained GLiNER2 model using LoRA (Low-Rank Adaptation)
for efficient domain-specific entity extraction.

Supports:
- Config file (YAML) for easy configuration
- TRM JSONL format (auto-converts) and pre-converted GLiNER2 format
- Multiple training data files (concatenated)
- Automatic validation split from training data (per-file)
- DDP training via torchrun

Usage:
    # Train with config file:
    uv run python tools/train_gliner2.py --config configs/gliner2_train.yaml

    # Override config options via CLI:
    uv run python tools/train_gliner2.py --config configs/gliner2_train.yaml \
        --epochs 20 --batch-size 32

    # Train without config file:
    uv run python tools/train_gliner2.py \
        --train-data data/train.jsonl \
        --val-data data/test.jsonl \
        --output-dir outputs/gliner2

    # Split validation from training data (10% per file):
    uv run python tools/train_gliner2.py \
        --train-data data/train1.jsonl data/train2.jsonl \
        --val-split 0.1 \
        --output-dir outputs/gliner2

    # Multiple training files (concatenated):
    uv run python tools/train_gliner2.py \
        --train-data data/train1.jsonl data/train2.jsonl \
        --output-dir outputs/gliner2

    # Multi-GPU training with torchrun:
    torchrun --nproc_per_node=4 tools/train_gliner2.py \
        --config configs/gliner2_train.yaml

    # Use the trained adapter:
    GLINER2_ADAPTER=outputs/gliner2/best uv run chainlit run app.py
"""

import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import Optional

import yaml

from gliner2 import GLiNER2
from gliner2.training.data import InputExample
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

import torch.distributed as dist

from trm_agent.utils import get_logger, is_main_process, get_world_size

# Initialize logger (automatically handles DDP - only logs on main process)
logger = get_logger("gliner2_train")


# ============================================================================
# TRM to GLiNER2 Conversion Functions
# ============================================================================


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


def convert_trm_sample_to_gliner2(
    sample: dict,
    sample_idx: int = 0,
    args_only: bool = False,
    tool_call_only: bool = False,
) -> tuple[Optional[dict], Optional[str]]:
    """Convert a TRM sample to GLiNER2 InputExample format.

    Args:
        sample: TRM sample dict
        sample_idx: Sample index for logging
        args_only: If True, only extract tool arguments (skip slots)
        tool_call_only: If True, skip direct_answer samples

    Returns:
        Tuple of (converted_dict, skip_reason)
        - If successful: (dict with 'text' and 'entities', None)
        - If skipped: (None, reason_string)
    """
    history = sample.get("history", [])
    slots = sample.get("slots", {})
    tool = sample.get("tool", {})
    decision = sample.get("decision", "unknown")

    # Skip direct_answer if tool_call_only is enabled
    if tool_call_only and decision != "tool_call":
        return None, f"skip_direct_answer (decision={decision})"

    # Get full conversation text
    text = conversation_to_text(history)
    if not text:
        return None, "empty_history"

    # Collect all slot/arg values and track which are found
    all_values = {}  # name -> value
    found_values = {}  # name -> value (only those found in text)

    # Extract slots (skip if args_only)
    if not args_only:
        for slot_name, slot_value in slots.items():
            if slot_value and isinstance(slot_value, str) and slot_value.strip():
                all_values[f"slot:{slot_name}"] = slot_value
                if find_entity_in_text(text, slot_value):
                    found_values[slot_name] = slot_value

    # Extract tool arguments
    if tool and isinstance(tool, dict):
        tool_args = tool.get("arguments", {})
        for arg_name, arg_value in tool_args.items():
            if arg_value and isinstance(arg_value, str) and arg_value.strip():
                all_values[f"arg:{arg_name}"] = arg_value
                if find_entity_in_text(text, arg_value):
                    found_values[arg_name] = arg_value

    # Build entities dict (group by label)
    entities: dict[str, list[str]] = {}
    for name, value in found_values.items():
        if name not in entities:
            entities[name] = []
        if value not in entities[name]:
            entities[name].append(value)

    if not entities:
        if not all_values:
            return None, f"no_entities (decision={decision})"
        else:
            # Has values but none found in text
            not_found = {k: v for k, v in all_values.items() if k.split(":", 1)[1] not in found_values}
            return None, f"values_not_in_text: {not_found}"

    return {"text": text, "entities": entities}, None


def convert_trm_jsonl_to_gliner2(
    jsonl_path: Path,
    max_samples: Optional[int] = None,
    log_skipped: bool = True,
    args_only: bool = False,
    tool_call_only: bool = False,
) -> tuple[list[dict], dict]:
    """Convert TRM JSONL dataset to GLiNER2 format.

    Args:
        jsonl_path: Path to TRM JSONL file
        max_samples: Maximum samples to convert (None for all)
        log_skipped: Whether to log skipped samples with reasons
        args_only: If True, only extract tool arguments (skip slots)
        tool_call_only: If True, skip direct_answer samples

    Returns:
        Tuple of (samples_list, stats_dict)
    """
    samples = []
    total = 0
    converted = 0
    entity_counts: dict[str, int] = {}
    skip_reasons: dict[str, int] = {}  # reason -> count

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total += 1
            if max_samples and total > max_samples:
                break

            try:
                sample = json.loads(line)
                gliner2_sample, skip_reason = convert_trm_sample_to_gliner2(
                    sample, total, args_only=args_only, tool_call_only=tool_call_only
                )

                if gliner2_sample:
                    samples.append(gliner2_sample)
                    converted += 1

                    # Count entities
                    for label, values in gliner2_sample["entities"].items():
                        entity_counts[label] = entity_counts.get(label, 0) + len(values)
                else:
                    # Track skip reason
                    # Normalize reason for counting (remove specific values)
                    reason_key = skip_reason.split(":")[0] if skip_reason else "unknown"
                    skip_reasons[reason_key] = skip_reasons.get(reason_key, 0) + 1

                    if log_skipped:
                        logger.debug(f"[Sample {total}] Skipped: {skip_reason}")

            except json.JSONDecodeError as e:
                skip_reasons["json_error"] = skip_reasons.get("json_error", 0) + 1
                if log_skipped:
                    logger.debug(f"[Sample {total}] JSON decode error: {e}")
                continue

    # Log skip summary
    if skip_reasons:
        logger.info(f"Skipped samples by reason: {skip_reasons}")

    stats = {
        "total_samples": total,
        "converted_samples": converted,
        "entity_counts": entity_counts,
        "skip_reasons": skip_reasons,
    }

    return samples, stats


# ============================================================================
# Data Loading Functions
# ============================================================================


def load_gliner2_data(
    data_path: Path,
    args_only: bool = False,
    tool_call_only: bool = False,
) -> list[InputExample]:
    """Load GLiNER2 training data from pickle, JSON, or TRM JSONL file.

    Automatically detects format:
    - .pkl: Pre-converted GLiNER2 pickle format
    - .json: Pre-converted GLiNER2 JSON format
    - .jsonl: TRM dataset (auto-converts to GLiNER2)

    Args:
        data_path: Path to data file
        args_only: If True, only extract tool arguments (skip slots) - JSONL only
        tool_call_only: If True, skip direct_answer samples - JSONL only

    Returns:
        List of InputExample objects
    """
    suffix = data_path.suffix.lower()

    if suffix == ".jsonl":
        # TRM format - convert automatically
        mode_info = []
        if args_only:
            mode_info.append("args-only")
        if tool_call_only:
            mode_info.append("tool-call-only")
        mode_str = f" ({', '.join(mode_info)})" if mode_info else ""
        logger.info(f"Detected TRM JSONL format, converting to GLiNER2{mode_str}...")
        raw_data, stats = convert_trm_jsonl_to_gliner2(
            data_path, args_only=args_only, tool_call_only=tool_call_only
        )
        skipped = stats['total_samples'] - stats['converted_samples']
        logger.info(f"Converted {stats['converted_samples']}/{stats['total_samples']} samples (skipped: {skipped})")
        if stats.get("skip_reasons"):
            logger.info(f"Skip reasons: {stats['skip_reasons']}")
        if stats["entity_counts"]:
            logger.info(f"Entity types: {list(stats['entity_counts'].keys())}")
    elif suffix == ".pkl":
        with open(data_path, "rb") as f:
            raw_data = pickle.load(f)
    else:
        # Assume JSON format
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

    # Convert to InputExample
    examples = []
    for item in raw_data:
        examples.append(
            InputExample(
                text=item["text"],
                entities=item["entities"],
            )
        )

    return examples


def load_multiple_gliner2_data(
    data_paths: list[Path],
    args_only: bool = False,
    tool_call_only: bool = False,
) -> list[InputExample]:
    """Load and concatenate GLiNER2 training data from multiple files.

    Args:
        data_paths: List of paths to data files
        args_only: If True, only extract tool arguments (skip slots)
        tool_call_only: If True, skip direct_answer samples

    Returns:
        Concatenated list of InputExample objects
    """
    all_examples = []

    for i, data_path in enumerate(data_paths):
        logger.info(f"[{i + 1}/{len(data_paths)}] Loading: {data_path}")
        examples = load_gliner2_data(data_path, args_only=args_only, tool_call_only=tool_call_only)
        all_examples.extend(examples)
        logger.info(f"Loaded {len(examples)} examples (total: {len(all_examples)})")

    return all_examples


def load_and_split_gliner2_data(
    data_paths: list[Path],
    val_split: float = 0.1,
    seed: int = 42,
    args_only: bool = False,
    tool_call_only: bool = False,
) -> tuple[list[InputExample], list[InputExample]]:
    """Load GLiNER2 data and split each file into train/val.

    Splits each file individually to ensure proportional representation
    from each source in both train and val sets.

    Args:
        data_paths: List of paths to data files
        val_split: Fraction of each file to use for validation (0.0 to 1.0)
        seed: Random seed for reproducibility
        args_only: If True, only extract tool arguments (skip slots)
        tool_call_only: If True, skip direct_answer samples

    Returns:
        Tuple of (train_examples, val_examples)
    """
    import random

    all_train = []
    all_val = []
    rng = random.Random(seed)

    for i, data_path in enumerate(data_paths):
        logger.info(f"[{i + 1}/{len(data_paths)}] Loading: {data_path}")
        examples = load_gliner2_data(data_path, args_only=args_only, tool_call_only=tool_call_only)

        # Shuffle and split
        examples_copy = list(examples)
        rng.shuffle(examples_copy)

        val_size = int(len(examples_copy) * val_split)
        val_examples = examples_copy[:val_size]
        train_examples = examples_copy[val_size:]

        all_train.extend(train_examples)
        all_val.extend(val_examples)

        logger.info(
            f"Split: {len(train_examples)} train, {len(val_examples)} val "
            f"(total: {len(all_train)} train, {len(all_val)} val)"
        )

    return all_train, all_val


def get_entity_labels(examples: list[InputExample]) -> list[str]:
    """Extract unique entity labels from examples."""
    labels = set()
    for example in examples:
        labels.update(example.entities.keys())
    return sorted(labels)


def log_sample_examples(
    examples: list[InputExample],
    num_samples: int = 3,
    max_text_length: int = 200,
) -> None:
    """Log sample examples for verification.

    Args:
        examples: List of InputExample objects
        num_samples: Number of samples to log
        max_text_length: Maximum text length to display
    """
    logger.info("=" * 60)
    logger.info(f"Sample Examples ({min(num_samples, len(examples))} of {len(examples)})")
    logger.info("=" * 60)

    for i, example in enumerate(examples[:num_samples]):
        logger.info(f"--- Sample {i + 1} ---")

        # Truncate text if too long
        text = example.text
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."
        logger.info(f"Text: {text}")

        logger.info("Entities:")
        for label, values in example.entities.items():
            logger.info(f"  {label}: {values}")

    logger.info("=" * 60)


# ============================================================================
# Config Loading
# ============================================================================


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    """Merge config file with CLI arguments. CLI args override config."""
    # Map CLI arg names to config keys
    arg_to_config = {
        "train_data": "train_data",
        "val_data": "val_data",
        "val_split": "val_split",
        "output_dir": "output_dir",
        "base_model": "base_model",
        "epochs": "num_epochs",
        "batch_size": "batch_size",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "task_lr": "task_lr",
        "lora_r": "lora_r",
        "lora_alpha": "lora_alpha",
        "lora_dropout": "lora_dropout",
        "lora_target_modules": "lora_target_modules",
        "logging_steps": "logging_steps",
        "seed": "seed",
    }

    # Override config with CLI args if provided
    for arg_name, config_key in arg_to_config.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            config[config_key] = arg_value

    # Handle special cases
    if args.no_fp16:
        config["fp16"] = False

    # Handle boolean flags (always override if set)
    if args.args_only:
        config["args_only"] = True
    if args.tool_call_only:
        config["tool_call_only"] = True

    return config


# ============================================================================
# Training with GLiNER2Trainer (supports DDP via torchrun)
# ============================================================================


def train_gliner2_lora(
    train_data_paths: list[Path],
    val_data_paths: Optional[list[Path]],
    output_dir: Path,
    base_model: str = "fastino/gliner2-multi-v1",
    # Training
    num_epochs: int = 10,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 2,
    # Learning rates
    encoder_lr: float = 1e-5,  # Ignored when LoRA enabled
    task_lr: float = 5e-4,     # Used for LoRA + task heads when LoRA enabled
    # Optimization
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    scheduler_type: str = "linear",
    warmup_ratio: float = 0.1,
    # Mixed precision
    fp16: bool = True,
    # Checkpointing
    eval_strategy: str = "epoch",
    save_best: bool = True,
    save_total_limit: int = 3,
    # Early stopping
    early_stopping: bool = False,
    early_stopping_patience: int = 5,
    # Logging
    logging_steps: int = 50,
    # LoRA settings
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.1,
    lora_target_modules: Optional[list[str]] = None,
    save_adapter_only: bool = True,
    # Data split
    val_split: float = 0.0,  # Split ratio for validation (0.0 = no split, use val_data_paths)
    # Data filtering (for TRM JSONL format)
    args_only: bool = False,  # Only extract tool arguments (skip slots)
    tool_call_only: bool = False,  # Skip direct_answer samples
    # Other
    seed: int = 42,
):
    """Train GLiNER2 with LoRA adapter using GLiNER2Trainer.

    Supports DDP via torchrun (auto-detects LOCAL_RANK environment variable).
    """
    # Auto-detect DDP from environment (set by torchrun)
    # torchrun sets RANK, LOCAL_RANK, and WORLD_SIZE
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = get_world_size()

    # Initialize distributed if LOCAL_RANK is set but dist not initialized
    # (happens when running with torchrun but before trainer is created)
    if local_rank >= 0 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        import torch
        torch.cuda.set_device(local_rank)

    logger.info("=" * 60)
    logger.info("GLiNER2 LoRA Training")
    logger.info("=" * 60)
    if local_rank >= 0:
        logger.info(f"Distributed training: {world_size} GPUs, local_rank={local_rank}")
    else:
        logger.info("Single GPU training")

    # If tool_call_only is enabled, also enable args_only (tool calls have args, not slots)
    if tool_call_only and not args_only:
        args_only = True
        logger.info("Note: --tool-call-only implies --args-only (auto-enabled)")

    # Log data filtering options
    if args_only or tool_call_only:
        filter_opts = []
        if args_only:
            filter_opts.append("args-only")
        if tool_call_only:
            filter_opts.append("tool-call-only")
        logger.info(f"Data filtering: {', '.join(filter_opts)}")

    # Load training data
    val_examples = None

    if val_split > 0:
        # Split validation from training data (per-file) - takes priority over val_data_paths
        logger.info(f"Loading and splitting training data ({len(train_data_paths)} file(s), val_split={val_split}):")
        train_examples, val_examples = load_and_split_gliner2_data(
            train_data_paths, val_split=val_split, seed=seed,
            args_only=args_only, tool_call_only=tool_call_only
        )
        logger.info(f"Total: {len(train_examples)} training, {len(val_examples)} validation examples")
    else:
        # Load all training data (multiple files concatenated)
        logger.info(f"Loading training data ({len(train_data_paths)} file(s)):")
        train_examples = load_multiple_gliner2_data(
            train_data_paths, args_only=args_only, tool_call_only=tool_call_only
        )
        logger.info(f"Total: {len(train_examples)} training examples")

        # Load validation data if provided (multiple files concatenated)
        if val_data_paths:
            valid_val_paths = [p for p in val_data_paths if p.exists()]
            if valid_val_paths:
                logger.info(f"Loading validation data ({len(valid_val_paths)} file(s)):")
                val_examples = load_multiple_gliner2_data(
                    valid_val_paths, args_only=args_only, tool_call_only=tool_call_only
                )
                logger.info(f"Total: {len(val_examples)} validation examples")

    # Get entity labels
    entity_labels = get_entity_labels(train_examples)
    logger.info(f"Entity labels ({len(entity_labels)}): {entity_labels}")

    # Log sample examples for verification
    log_sample_examples(train_examples, num_samples=3)

    # Create output directory (only on main process)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save entity labels
        labels_path = output_dir / "entity_labels.json"
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(entity_labels, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved entity labels to {labels_path}")

    # Default target modules
    if lora_target_modules is None:
        lora_target_modules = ["encoder", "span_rep", "classifier"]

    logger.info("LoRA Configuration:")
    logger.info(f"  Rank (r):        {lora_r}")
    logger.info(f"  Alpha:           {lora_alpha}")
    logger.info(f"  Scaling (α/r):   {lora_alpha / lora_r:.4f}")
    logger.info(f"  Dropout:         {lora_dropout}")
    logger.info(f"  Target modules:  {lora_target_modules}")
    logger.info(f"  Save adapter:    {save_adapter_only}")

    logger.info("Training Configuration:")
    logger.info(f"  Epochs:          {num_epochs}")
    logger.info(f"  Batch size/GPU:  {batch_size}")
    logger.info(f"  Grad accum:      {gradient_accumulation_steps}")
    logger.info(f"  Effective batch: {batch_size * gradient_accumulation_steps * world_size}")
    logger.info(f"  Task LR:         {task_lr}")
    logger.info(f"  Scheduler:       {scheduler_type}")
    logger.info(f"  Warmup ratio:    {warmup_ratio}")
    logger.info(f"  FP16:            {fp16}")

    # Load base model
    logger.info(f"Loading base model: {base_model}")
    model = GLiNER2.from_pretrained(base_model)

    # Disable early_stopping if no validation data
    effective_early_stopping = early_stopping and val_examples is not None
    if early_stopping and not effective_early_stopping:
        logger.warning("early_stopping disabled (no validation data provided)")

    # Training configuration with DDP and LoRA support
    config = TrainingConfig(
        # Output
        output_dir=str(output_dir),
        experiment_name="gliner2_lora",
        # Training
        num_epochs=int(num_epochs),
        batch_size=int(batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        # Learning rates (ensure float type)
        encoder_lr=float(encoder_lr),  # Ignored when LoRA enabled
        task_lr=float(task_lr),        # Used for LoRA + task heads when LoRA enabled
        # Optimization
        weight_decay=float(weight_decay),
        max_grad_norm=float(max_grad_norm),
        scheduler_type=scheduler_type,
        warmup_ratio=float(warmup_ratio),
        # Mixed precision
        fp16=fp16,
        # Checkpointing & Evaluation
        eval_strategy=eval_strategy if val_examples else "no",
        save_best=save_best if val_examples else False,
        save_total_limit=int(save_total_limit),
        # Early stopping (requires validation data)
        early_stopping=effective_early_stopping,
        early_stopping_patience=int(early_stopping_patience),
        # Logging
        logging_steps=int(logging_steps),
        # LoRA settings
        use_lora=use_lora,
        lora_r=int(lora_r),
        lora_alpha=float(lora_alpha),
        lora_dropout=float(lora_dropout),
        lora_target_modules=lora_target_modules,
        save_adapter_only=save_adapter_only,
        # Other
        seed=int(seed),
        # DDP: -1 for single GPU, >= 0 for distributed (set by torchrun)
        local_rank=int(local_rank),
    )

    logger.info(f"local_rank={local_rank}, world_size={world_size}")
    logger.info(f"TrainingConfig: encoder_lr={encoder_lr} (type={type(encoder_lr).__name__}), task_lr={task_lr} (type={type(task_lr).__name__})")

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = GLiNER2Trainer(model=model, config=config)

    # Train
    logger.info("=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)

    results = trainer.train(
        train_data=train_examples,
        eval_data=val_examples,
    )

    # Print results
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Total time: {results.get('total_time_seconds', 0):.2f}s")
    logger.info(f"Final loss: {results.get('final_loss', 'N/A')}")

    best_path = output_dir / "best"
    logger.info(f"Best model saved to: {best_path}")
    logger.info("To use the adapter:")
    logger.info(f'  GLINER2_ADAPTER="{best_path}" uv run chainlit run app.py')

    # Cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train GLiNER2 with LoRA adapters (supports DDP via torchrun)"
    )

    # Config file
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file. CLI args override config file.",
    )

    # Data paths
    parser.add_argument(
        "--train-data",
        type=Path,
        nargs="+",
        default=None,
        help="Path(s) to training data (.jsonl for TRM, .pkl/.json for GLiNER2).",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        nargs="+",
        default=None,
        help="Path(s) to validation data.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=None,
        help="Split ratio for validation from training data (0.0-1.0). Applied per-file. Ignored if --val-data is provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for adapter",
    )

    # Model
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Pre-trained model to fine-tune",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per GPU")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--task-lr", type=float, default=None, help="Learning rate")

    # LoRA
    parser.add_argument("--lora-r", type=int, default=None, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=None, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=None, help="LoRA dropout")
    parser.add_argument("--lora-target-modules", type=str, nargs="+", default=None)

    # Other
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # Data filtering (for TRM JSONL format)
    parser.add_argument(
        "--args-only",
        action="store_true",
        help="Only extract tool arguments (skip slots). Useful for GLiNER2 training.",
    )
    parser.add_argument(
        "--tool-call-only",
        action="store_true",
        help="Skip direct_answer samples, only train on tool_call samples.",
    )

    args = parser.parse_args()

    # Load config from file or use defaults
    if args.config and args.config.exists():
        logger.info(f"Loading config from: {args.config}")
        config = load_config(args.config)
        config = merge_config_with_args(config, args)
    else:
        # Use CLI args or defaults
        config = {
            "train_data": args.train_data,
            "val_data": args.val_data,
            "val_split": args.val_split or 0.0,
            "output_dir": args.output_dir or Path("outputs/gliner2"),
            "base_model": args.base_model or "fastino/gliner2-multi-v1",
            "num_epochs": args.epochs or 10,
            "batch_size": args.batch_size or 16,
            "gradient_accumulation_steps": args.gradient_accumulation_steps or 2,
            "encoder_lr": 1e-5,  # Ignored when LoRA enabled
            "task_lr": args.task_lr or 5e-4,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "scheduler_type": "linear",
            "warmup_ratio": 0.1,
            "fp16": not args.no_fp16,
            "eval_strategy": "epoch",
            "save_best": True,
            "save_total_limit": 3,
            "early_stopping": False,
            "early_stopping_patience": 5,
            "logging_steps": args.logging_steps or 50,
            "use_lora": True,
            "lora_r": args.lora_r or 16,
            "lora_alpha": args.lora_alpha or 32.0,
            "lora_dropout": args.lora_dropout or 0.1,
            "lora_target_modules": args.lora_target_modules or ["encoder", "span_rep", "classifier"],
            "save_adapter_only": True,
            "seed": args.seed or 42,
            "args_only": args.args_only,
            "tool_call_only": args.tool_call_only,
        }

    # Validate required fields
    if not config.get("train_data"):
        parser.error("--train-data is required (or specify in config file)")

    # Convert paths
    train_data_paths = [Path(p) for p in config["train_data"]]
    val_data_paths = [Path(p) for p in config["val_data"]] if config.get("val_data") else None
    output_dir = Path(config["output_dir"])

    # Run training
    train_gliner2_lora(
        train_data_paths=train_data_paths,
        val_data_paths=val_data_paths,
        output_dir=output_dir,
        base_model=config.get("base_model", "fastino/gliner2-multi-v1"),
        num_epochs=config.get("num_epochs", 10),
        batch_size=config.get("batch_size", 16),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
        encoder_lr=config.get("encoder_lr", 1e-5),
        task_lr=config.get("task_lr", 5e-4),
        weight_decay=config.get("weight_decay", 0.01),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        scheduler_type=config.get("scheduler_type", "linear"),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        fp16=config.get("fp16", True),
        eval_strategy=config.get("eval_strategy", "epoch"),
        save_best=config.get("save_best", True),
        save_total_limit=config.get("save_total_limit", 3),
        early_stopping=config.get("early_stopping", False),
        early_stopping_patience=config.get("early_stopping_patience", 5),
        logging_steps=config.get("logging_steps", 50),
        use_lora=config.get("use_lora", True),
        lora_r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32.0),
        lora_dropout=config.get("lora_dropout", 0.1),
        lora_target_modules=config.get("lora_target_modules", ["encoder", "span_rep", "classifier"]),
        save_adapter_only=config.get("save_adapter_only", True),
        val_split=config.get("val_split", 0.0),
        args_only=config.get("args_only", False),
        tool_call_only=config.get("tool_call_only", False),
        seed=config.get("seed", 42),
    )


if __name__ == "__main__":
    main()
