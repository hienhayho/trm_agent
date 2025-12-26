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

    # With validation split (10% of training data):
    uv run python tools/train.py \
        --config configs/default.yaml \
        --train-data data/train.jsonl \
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
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split

from trm_agent.data import TRMCollator, TRMTokenizer, TRMToolCallingDataset
from trm_agent.data.collator import create_dataloader
from trm_agent.models import TRMConfig, TRMForToolCalling
from trm_agent.training import TRMTrainer
from trm_agent.training.trainer import TrainingConfig
from trm_agent.utils import get_logger, is_main_process, get_rank, get_world_size, barrier

logger = get_logger(__name__)


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

        # Set device
        torch.cuda.set_device(local_rank)

        return True, local_rank
    return False, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_text_for_tokenizer(data_path: str | Path, output_path: Path) -> Path:
    """Extract text from JSONL dataset for tokenizer training.

    Args:
        data_path: Path to JSONL training file
        output_path: Path to save extracted text

    Returns:
        Path to the extracted text file
    """
    logger.info(f"Extracting text from {data_path} for tokenizer training...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(data_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

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
    data_path: str | Path,
    output_dir: Path,
    vocab_size: int = 32000,
) -> TRMTokenizer:
    """Train a tokenizer from the training data.

    Args:
        data_path: Path to JSONL training file
        output_dir: Directory to save tokenizer
        vocab_size: Vocabulary size

    Returns:
        Trained TRMTokenizer
    """
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    # Extract text to a temporary file
    text_file = tokenizer_dir / "train_text.txt"
    extract_text_for_tokenizer(data_path, text_file)

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
        required=True,
        help="Path to training JSONL file",
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
    is_distributed, local_rank = setup_distributed()
    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
        logger.info(f"Distributed training: rank {get_rank()}/{get_world_size()}, device: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Single GPU/CPU training, device: {device}")

    # Set random seed
    torch.manual_seed(args.seed + get_rank())  # Different seed per rank for data augmentation
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + get_rank())

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
    if is_distributed:
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
                data_path=args.train_data,
                output_dir=output_dir,
                vocab_size=vocab_size,
            )
            tokenizer_path = output_dir / "tokenizer" / "tokenizer.model"

    # Wait for rank 0 to finish tokenizer training
    if is_distributed:
        barrier()

    # Now all ranks load the tokenizer
    if tokenizer_path is None:
        tokenizer_path = output_dir / "tokenizer" / "tokenizer.model"

    tokenizer = TRMTokenizer(tokenizer_path)
    logger.info(f"Loaded tokenizer from {tokenizer_path}")

    # Load datasets
    full_dataset = TRMToolCallingDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        max_seq_len=model_config.max_seq_len,
        slot_fields=model_config.slot_fields,
    )
    logger.info(f"Loaded dataset: {len(full_dataset)} samples")
    logger.info(f"Label statistics: {full_dataset.get_label_statistics()}")

    train_dataset = full_dataset
    val_dataset = None

    if args.val_data:
        # Use separate validation file
        val_dataset = TRMToolCallingDataset(
            data_path=args.val_data,
            tokenizer=tokenizer,
            max_seq_len=model_config.max_seq_len,
            slot_fields=model_config.slot_fields,
            tool_name_to_id=full_dataset.tool_name_to_id,
        )
        logger.info(f"Loaded validation dataset: {len(val_dataset)} samples")
    elif args.val_split is not None and args.val_split > 0:
        # Split training data
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        logger.info(
            f"Split dataset: {train_size} train, {val_size} val "
            f"({args.val_split:.1%} validation)"
        )

    # Update model config with actual tool count
    model_config.num_tools = len(full_dataset.tool_name_to_id)
    logger.info(f"Number of tools: {model_config.num_tools}")

    # Create data loaders
    pad_token_id = tokenizer.pad_token_id if tokenizer else 0
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        pad_token_id=pad_token_id,
        distributed=is_distributed,
    )

    val_dataloader = None
    if val_dataset:
        val_dataloader = create_dataloader(
            val_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            pad_token_id=pad_token_id,
            distributed=is_distributed,
        )

    # Create model
    model = TRMForToolCalling(model_config)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created model with {num_params:,} parameters")

    # Wrap with DDP if distributed
    if is_distributed:
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
        tool_names=list(full_dataset.tool_name_to_id.keys()),
        slot_fields=model_config.slot_fields,
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
