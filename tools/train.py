"""
TRM Training Script.

Usage:
    uv run python tools/train.py --config configs/default.yaml --train-data data/train.jsonl

Example:
    uv run python tools/train.py \
        --config configs/default.yaml \
        --train-data data/train.jsonl \
        --val-data data/val.jsonl \
        --output-dir outputs/ \
        --tokenizer-path tokenizer/tokenizer.model

    # Multiple training files:
    uv run python tools/train.py \
        --config configs/default.yaml \
        --train-data data/train1.jsonl data/train2.jsonl data/train3.jsonl \
        --output-dir outputs/

    # With validation split (10% of EACH training file):
    uv run python tools/train.py \
        --config configs/default.yaml \
        --train-data data/train1.jsonl data/train2.jsonl \
        --val-split 0.1 \
        --output-dir outputs/

    # Auto-train tokenizer from dataset:
    uv run python tools/train.py \
        --config configs/default.yaml \
        --train-data data/train.jsonl \
        --output-dir outputs/

    # Distributed training with torchrun:
    torchrun --standalone --nproc_per_node=2 tools/train.py \
        --config configs/default.yaml \
        --train-data data/train.jsonl \
        --output-dir outputs/
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, random_split

from trm_agent.data import TRMTokenizer, TRMToolCallingDataset
from trm_agent.data.collator import create_dataloader
from trm_agent.models import TRMConfig, TRMForToolCalling
from trm_agent.training import TRMTrainer
from trm_agent.training.trainer import TrainingConfig
from trm_agent.utils import (
    barrier,
    cleanup_distributed,
    get_logger,
    is_distributed,
    is_main_process,
    setup_distributed,
)

logger = get_logger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_text_for_tokenizer(
    data_paths: list[str | Path],
    output_path: Path,
) -> Path:
    """Extract text from JSONL datasets for tokenizer training.

    Args:
        data_paths: List of paths to JSONL training files
        output_path: Path to save extracted text

    Returns:
        Path to the extracted text file
    """
    logger.info(f"Extracting text from {len(data_paths)} file(s) for tokenizer training...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f_out:
        for data_path in data_paths:
            logger.info(f"  Processing {data_path}...")
            with open(data_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    sample = json.loads(line)

                    # Extract text from history
                    history = sample.get("history", [])
                    for turn in history:
                        content = turn.get("content", "")
                        if isinstance(content, dict):
                            # Tool call or tool response
                            content = json.dumps(content, ensure_ascii=False)
                        if content:
                            f_out.write(content + "\n")

                    # Extract text from tools
                    tools = sample.get("tools", [])
                    for tool in tools:
                        if "function" in tool:
                            func = tool["function"]
                            if "name" in func:
                                f_out.write(func["name"] + "\n")
                            if "description" in func:
                                f_out.write(func["description"] + "\n")

                    # Extract content (for direct_answer)
                    content = sample.get("content", "")
                    if content:
                        f_out.write(content + "\n")

                    # Extract slot values
                    slots = sample.get("slots", {})
                    for value in slots.values():
                        if value:
                            f_out.write(str(value) + "\n")

    logger.info(f"Text extracted to {output_path}")
    return output_path


def train_tokenizer_from_data(
    data_paths: list[str | Path],
    output_dir: Path,
    vocab_size: int = 32000,
) -> TRMTokenizer:
    """Train a tokenizer from the training data.

    Args:
        data_paths: List of paths to JSONL training files
        output_dir: Directory to save tokenizer
        vocab_size: Vocabulary size

    Returns:
        Trained TRMTokenizer
    """
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    # Extract text to a temporary file
    text_file = tokenizer_dir / "train_text.txt"
    extract_text_for_tokenizer(data_paths, text_file)

    # Train tokenizer
    logger.info(f"Training tokenizer with vocab_size={vocab_size}...")
    tokenizer = TRMTokenizer.train(
        input_files=[text_file],
        output_path=tokenizer_dir / "tokenizer",
        vocab_size=vocab_size,
    )

    logger.info(f"Tokenizer trained and saved to {tokenizer_dir}")
    logger.info(f"Vocabulary size: {len(tokenizer)}")

    # Clean up temp text file
    text_file.unlink()

    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Train TRM model for tool-calling"
    )

    # Data arguments
    parser.add_argument(
        "--train-data",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to training JSONL file(s). Multiple files can be specified.",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation JSONL file (optional)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=None,
        help="Validation split ratio (e.g., 0.1 for 10%%). Used if --val-data not provided",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to SentencePiece tokenizer model. If not provided, will auto-train from dataset.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocabulary size for tokenizer training (default: from config)",
    )

    # Config arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for model, checkpoints and logs",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (effective batch = batch_size * this)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable Automatic Mixed Precision",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--log-sample-interval",
        type=int,
        default=None,
        help="Log sample predictions every N steps (0 = disabled)",
    )

    # Model arguments
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Hidden size",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of layers (default: 2)",
    )

    # Other arguments
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, device = setup_distributed()
    if is_distributed():
        logger.info(f"Distributed training: rank {rank}/{world_size}, device: {device}")
    else:
        logger.info(f"Single GPU/CPU training, device: {device}")

    # Set random seed
    torch.manual_seed(args.seed + rank)  # Different seed per rank for data augmentation
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    # Load configuration
    config_dict = {}
    if args.config:
        config_dict = load_config(Path(args.config))

    # Override with command line arguments
    model_config_dict = config_dict.get("model", {})
    training_config_dict = config_dict.get("training", {})

    if args.hidden_size:
        model_config_dict["hidden_size"] = args.hidden_size
    if args.num_layers:
        model_config_dict["num_layers"] = args.num_layers

    if args.epochs:
        training_config_dict["num_epochs"] = args.epochs
    if args.batch_size:
        training_config_dict["batch_size"] = args.batch_size
    if args.gradient_accumulation_steps:
        training_config_dict["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.no_amp:
        training_config_dict["use_amp"] = False
    if args.learning_rate:
        training_config_dict["learning_rate"] = args.learning_rate
    if args.warmup_steps:
        training_config_dict["warmup_steps"] = args.warmup_steps
    if args.log_sample_interval is not None:
        training_config_dict["log_sample_interval"] = args.log_sample_interval

    training_config_dict["output_dir"] = args.output_dir

    # Create configs
    model_config = TRMConfig(**model_config_dict)
    training_config = TrainingConfig(**training_config_dict)

    logger.info(f"Model config: {model_config}")
    logger.info(f"Training config: {training_config}")

    # Load or train tokenizer (only on rank 0)
    output_dir = Path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Synchronize before tokenizer operations
    if is_distributed():
        barrier()

    tokenizer = None
    tokenizer_path = None

    if args.tokenizer_path:
        tokenizer_path = args.tokenizer_path
    else:
        # Check if tokenizer already exists in output_dir
        existing_tokenizer = output_dir / "tokenizer" / "tokenizer.model"
        if existing_tokenizer.exists():
            tokenizer_path = existing_tokenizer
        elif is_main_process():
            # Train tokenizer from training data (only rank 0)
            logger.info("No tokenizer specified. Training tokenizer from dataset...")
            vocab_size = args.vocab_size or model_config.vocab_size
            train_tokenizer_from_data(
                data_paths=args.train_data,
                output_dir=output_dir,
                vocab_size=vocab_size,
            )
            tokenizer_path = output_dir / "tokenizer" / "tokenizer.model"

    # Wait for rank 0 to finish tokenizer training
    if is_distributed():
        barrier()

    # Now all ranks load the tokenizer
    if tokenizer_path is None:
        tokenizer_path = output_dir / "tokenizer" / "tokenizer.model"

    tokenizer = TRMTokenizer(tokenizer_path)
    logger.info(f"Loaded tokenizer from {tokenizer_path}")

    # Load datasets from multiple files
    all_datasets = []
    tool_name_to_id = None  # Will be set from first dataset

    for i, data_path in enumerate(args.train_data):
        dataset = TRMToolCallingDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_len=model_config.max_seq_len,
            slot_fields=model_config.slot_fields,
            tool_name_to_id=tool_name_to_id,  # Share tool mapping across files
        )
        # Get tool mapping from first dataset
        if tool_name_to_id is None:
            tool_name_to_id = dataset.tool_name_to_id

        all_datasets.append(dataset)
        logger.info(f"Loaded {data_path}: {len(dataset)} samples")

    # Combine all datasets
    if len(all_datasets) == 1:
        full_dataset = all_datasets[0]
    else:
        full_dataset = ConcatDataset(all_datasets)

    total_samples = sum(len(d) for d in all_datasets)
    logger.info(f"Total dataset: {total_samples} samples from {len(args.train_data)} file(s)")

    # Get unified field info from first dataset
    first_dataset = all_datasets[0]
    unified_fields = first_dataset.unified_fields
    num_slots = first_dataset.num_slots
    tool_param_mask = first_dataset.get_tool_param_mask_tensor()

    train_dataset = full_dataset
    val_dataset = None

    if args.val_data:
        # Use separate validation file
        val_dataset = TRMToolCallingDataset(
            data_path=args.val_data,
            tokenizer=tokenizer,
            max_seq_len=model_config.max_seq_len,
            slot_fields=model_config.slot_fields,
            tool_name_to_id=tool_name_to_id,
        )
        logger.info(f"Loaded validation dataset: {len(val_dataset)} samples")

        # Log val dataset extraction stats (only on main process)
        if is_main_process():
            logger.info("Computing validation extraction statistics...")
            val_stats = val_dataset.get_extraction_stats()
            logger.info(f"Validation extraction stats:")
            logger.info(f"  Samples with missing slots: {val_stats['samples_with_missing_slots']}/{val_stats['total_samples']}")
            logger.info(f"  Samples with missing params: {val_stats['samples_with_missing_params']}/{val_stats['total_samples']}")
    elif args.val_split is not None and args.val_split > 0:
        # Split EACH training file and combine
        train_splits = []
        val_splits = []

        for i, dataset in enumerate(all_datasets):
            val_size = int(len(dataset) * args.val_split)
            train_size = len(dataset) - val_size

            train_part, val_part = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(args.seed + i),
            )
            train_splits.append(train_part)
            val_splits.append(val_part)
            logger.info(
                f"  {args.train_data[i]}: {train_size} train, {val_size} val"
            )

        # Combine splits
        if len(train_splits) == 1:
            train_dataset = train_splits[0]
            val_dataset = val_splits[0]
        else:
            train_dataset = ConcatDataset(train_splits)
            val_dataset = ConcatDataset(val_splits)

        total_train = sum(len(s) for s in train_splits)
        total_val = sum(len(s) for s in val_splits)
        logger.info(
            f"Split datasets: {total_train} train, {total_val} val "
            f"({args.val_split:.1%} validation per file)"
        )

    # Update model config with actual tool count and unified fields
    model_config.num_tools = len(tool_name_to_id)
    logger.info(f"Number of tools: {model_config.num_tools}")

    # Update model config with tool_param_fields from dataset
    model_config.set_tool_param_fields(first_dataset.tool_param_fields)
    logger.info(f"Model unified fields: {model_config.num_unified_fields} "
                f"({model_config.num_slots} slots + {model_config.num_tool_params} params)")
    logger.info(f"  Slots: {model_config.slot_fields}")
    logger.info(f"  Tool params: {model_config.tool_param_fields}")

    # Log per-tool parameter masks and find duplicates with slots
    slot_set = set(model_config.slot_fields)
    for tool_name, tool_id in tool_name_to_id.items():
        # Get active unique params (not in slots)
        param_mask = first_dataset.tool_param_mask.get(tool_name, [])
        active_params = [
            first_dataset.tool_param_fields[i]
            for i, m in enumerate(param_mask) if m == 1
        ]

        # Find params that were deduplicated (exist in slots)
        # Need to get original tool params from samples
        duplicated_params: list[str] = []
        found_tool = False
        for sample in first_dataset.samples:
            for tool in sample.get("tools", []):
                func = tool.get("function", {})
                if func.get("name") == tool_name:
                    raw_params = set(func.get("parameters", {}).get("properties", {}).keys())
                    duplicated_params = sorted(raw_params & slot_set)
                    found_tool = True
                    break
            if found_tool:
                break

        log_msg = f"  Tool '{tool_name}' (id={tool_id}): params={active_params}"
        if duplicated_params:
            log_msg += f", duplicated_with_slots={duplicated_params}"
        logger.info(log_msg)

    # Log extraction statistics (only on main process)
    if is_main_process():
        logger.info("Computing extraction statistics...")
        extraction_stats = first_dataset.get_extraction_stats()
        logger.info(f"Extraction stats (samples with missing spans):")
        logger.info(f"  Samples with missing slots: {extraction_stats['samples_with_missing_slots']}/{extraction_stats['total_samples']}")
        logger.info(f"  Samples with missing params: {extraction_stats['samples_with_missing_params']}/{extraction_stats['total_samples']}")

        # Per-slot extraction rates
        logger.info("  Per-slot extraction:")
        for field, field_stats in extraction_stats["slot_extraction"].items():
            if field_stats["total"] > 0:
                rate = field_stats["found"] / field_stats["total"] * 100
                logger.info(f"    {field}: {field_stats['found']}/{field_stats['total']} ({rate:.1f}%) - missing: {field_stats['missing']}")

        # Per-param extraction rates
        if extraction_stats["param_extraction"]:
            logger.info("  Per-param extraction:")
            for field, field_stats in extraction_stats["param_extraction"].items():
                if field_stats["total"] > 0:
                    rate = field_stats["found"] / field_stats["total"] * 100
                    logger.info(f"    {field}: {field_stats['found']}/{field_stats['total']} ({rate:.1f}%) - missing: {field_stats['missing']}")

        # Log examples for problematic fields (< 50% extraction rate)
        analysis = first_dataset.analyze_extraction_failures(max_examples_per_field=2)
        if analysis["problematic_fields"]:
            logger.warning(f"Problematic fields with <50% extraction rate: {analysis['problematic_fields']}")
            logger.warning("These fields may be GENERATED (not extracted from text). Consider:")
            logger.warning("  1. Removing from slot_fields if they're not extractable")
            logger.warning("  2. Using fuzzy matching for format variations")
            logger.warning("  3. Treating as generation task instead of extraction")

            for field in analysis["problematic_fields"]:
                examples = analysis["examples"].get(field, [])
                if examples:
                    logger.warning(f"  Examples for '{field}':")
                    for ex in examples[:2]:
                        logger.warning(f"    Value: '{ex['value']}'")
                        logger.warning(f"    Text: '{ex['text_snippet'][:150]}...'")

    # Create data loaders
    pad_token_id = tokenizer.pad_token_id if tokenizer else 0
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        pad_token_id=pad_token_id,
        distributed=is_distributed(),
    )

    val_dataloader = None
    if val_dataset:
        val_dataloader = create_dataloader(
            val_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            pad_token_id=pad_token_id,
            distributed=is_distributed(),
        )

    # Create model
    model = TRMForToolCalling(model_config)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created model with {num_params:,} parameters")

    # Wrap with DDP if distributed
    if is_distributed():
        local_rank = device.index  # Get GPU index from device
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            static_graph=True,  # Graph structure is fixed, enables optimizations
        )
        logger.info("Wrapped model with DDP")

    # Create trainer
    trainer = TRMTrainer(
        model=model,
        config=model_config,
        training_config=training_config,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        device=device,
        tool_names=list(tool_name_to_id.keys()),
        unified_fields=unified_fields,
        num_slots=num_slots,
        tool_param_mask=tool_param_mask,
        tokenizer=tokenizer,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
