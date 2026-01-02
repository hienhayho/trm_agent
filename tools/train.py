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
import time
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
    get_world_size,
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
    input_sentence_size: int = 0,  # 0 = use all sentences (no sampling)
) -> TRMTokenizer:
    """Train a tokenizer from the training data.

    If a *_tokenizer.txt file exists alongside the training JSONL,
    it will be used instead of extracting text (avoids duplicates).

    Args:
        data_paths: List of paths to JSONL training files
        output_dir: Directory to save tokenizer
        vocab_size: Vocabulary size
        input_sentence_size: Max sentences to sample for training (default 1M)

    Returns:
        Trained TRMTokenizer
    """
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    # Check for pre-generated tokenizer text files (from convert script)
    # These files have _tokenizer.txt suffix and contain no duplicates
    tokenizer_text_files = []
    for data_path in data_paths:
        data_path = Path(data_path)
        tokenizer_txt = data_path.parent / (data_path.stem + "_tokenizer.txt")
        if tokenizer_txt.exists():
            tokenizer_text_files.append(tokenizer_txt)
            logger.info(f"Found tokenizer text file: {tokenizer_txt}")

    if tokenizer_text_files:
        # Use pre-generated tokenizer text files (no duplicates)
        logger.info(f"Using {len(tokenizer_text_files)} pre-generated tokenizer text file(s)")
        input_files = tokenizer_text_files
        cleanup_text_file = False
    else:
        # Fall back to extracting text from JSONL (has duplicates)
        logger.warning("No *_tokenizer.txt files found. Extracting from JSONL (may have duplicates).")
        logger.warning("Consider running convert_raw_to_dataset.py to generate tokenizer text files.")
        text_file = tokenizer_dir / "train_text.txt"
        extract_text_for_tokenizer(data_paths, text_file)
        input_files = [text_file]
        cleanup_text_file = True

    # Train tokenizer
    if input_sentence_size > 0:
        logger.info(f"Training tokenizer with vocab_size={vocab_size}, "
                    f"sampling {input_sentence_size} sentences...")
    else:
        logger.info(f"Training tokenizer with vocab_size={vocab_size}, "
                    f"using ALL sentences (no sampling)...")
    tokenizer = TRMTokenizer.train(
        input_files=input_files,
        output_path=tokenizer_dir / "tokenizer",
        vocab_size=vocab_size,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
    )

    logger.info(f"Tokenizer trained and saved to {tokenizer_dir}")
    logger.info(f"Vocabulary size: {len(tokenizer)}")

    # Clean up temp text file if we created it
    if cleanup_text_file and input_files:
        input_files[0].unlink()

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
    parser.add_argument(
        "--no-conv-swiglu",
        action="store_true",
        help="Disable ConvSwiGLU (use standard SwiGLU, less memory)",
    )
    parser.add_argument(
        "--no-trm-loop",
        action="store_true",
        help="Disable TRM recursive loop (single pass, let Mamba handle recurrence)",
    )

    # Hybrid architecture arguments (Mamba + MoE + Attention)
    parser.add_argument(
        "--use-hybrid",
        action="store_true",
        help="Use hybrid Mamba+MoE+Attention blocks (requires mamba-ssm)",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Number of experts in MoE layer (default: 4)",
    )
    parser.add_argument(
        "--experts-per-token",
        type=int,
        default=None,
        help="Number of experts per token for top-k routing (default: 2)",
    )
    parser.add_argument(
        "--mamba-version",
        type=int,
        choices=[1, 2],
        default=None,
        help="Mamba version to use: 1 (original) or 2 (improved SSD, default)",
    )
    parser.add_argument(
        "--no-mamba-cuda-kernels",
        action="store_true",
        help="Disable Mamba2 CUDA kernels (use if training hangs)",
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
    if args.no_conv_swiglu:
        model_config_dict["use_conv_swiglu"] = False
    if args.no_trm_loop:
        model_config_dict["use_trm_loop"] = False

    # Hybrid architecture overrides
    if args.use_hybrid:
        model_config_dict["use_hybrid_block"] = True
    if args.num_experts is not None:
        model_config_dict["num_experts"] = args.num_experts
    if args.experts_per_token is not None:
        model_config_dict["num_experts_per_tok"] = args.experts_per_token
    if args.mamba_version is not None:
        model_config_dict["mamba_version"] = args.mamba_version
    if args.no_mamba_cuda_kernels:
        model_config_dict["mamba_use_mem_eff_path"] = False

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

    # Log model configuration
    logger.info("=" * 60)
    logger.info("Model Configuration:")
    logger.info(f"  Architecture: hidden_size={model_config.hidden_size}, "
                f"num_layers={model_config.num_layers}, num_heads={model_config.num_heads}")
    logger.info(f"  Intermediate: {model_config.intermediate_size}, dropout={model_config.dropout}")
    logger.info(f"  Sequence: max_seq_len={model_config.max_seq_len}, vocab_size={model_config.vocab_size}")
    if model_config.use_trm_loop:
        logger.info(f"  TRM recursion: ENABLED (n_latent={model_config.n_latent_recursion}, "
                    f"T_deep={model_config.T_deep_recursion}, N_sup={model_config.N_supervision})")
    else:
        logger.info(f"  TRM recursion: DISABLED (single-pass mode, Mamba handles recurrence)")
    logger.info(f"  URM innovations: use_conv_swiglu={model_config.use_conv_swiglu}, "
                f"conv_kernel_size={model_config.conv_kernel_size}, "
                f"tbptl_no_grad_steps={model_config.tbptl_no_grad_steps}")
    if model_config.use_hybrid_block:
        logger.info(f"  Hybrid block: ENABLED (Mamba + MoE + Attention)")
        if model_config.mamba_version == 2:
            cuda_kernels = "enabled" if model_config.mamba_use_mem_eff_path else "DISABLED"
            logger.info(f"    Mamba2: headdim={model_config.mamba_headdim}, "
                        f"d_conv={model_config.mamba_d_conv}, expand={model_config.mamba_expand}, "
                        f"cuda_kernels={cuda_kernels}")
        else:
            logger.info(f"    Mamba1: d_state={model_config.mamba_d_state}, "
                        f"d_conv={model_config.mamba_d_conv}, expand={model_config.mamba_expand}")
        logger.info(f"    MoE: num_experts={model_config.num_experts}, "
                    f"experts_per_tok={model_config.num_experts_per_tok}, "
                    f"intermediate={model_config.moe_intermediate_size}")
    else:
        logger.info(f"  Hybrid block: disabled (standard Transformer)")
    logger.info(f"  Loss weights: decision={model_config.decision_loss_weight}, "
                f"tool={model_config.tool_loss_weight}, q={model_config.q_loss_weight}")
    logger.info(f"  Focal loss: alpha={model_config.focal_alpha}, gamma={model_config.focal_gamma}")

    # Log training configuration
    world_size = get_world_size()
    effective_batch_size = training_config.batch_size * training_config.gradient_accumulation_steps * world_size
    logger.info("=" * 60)
    logger.info("Training Configuration:")
    logger.info(f"  Batch size: {training_config.batch_size} per GPU")
    logger.info(f"  Gradient accumulation: {training_config.gradient_accumulation_steps} steps")
    logger.info(f"  World size: {world_size} GPU(s)")
    logger.info(f"  Effective batch size: {effective_batch_size} "
                f"({training_config.batch_size} x {training_config.gradient_accumulation_steps} x {world_size})")
    logger.info(f"  Learning rate: {training_config.learning_rate}")
    logger.info(f"  Weight decay: {training_config.weight_decay}")
    logger.info(f"  Warmup steps: {training_config.warmup_steps}")
    logger.info(f"  Epochs: {training_config.num_epochs}")
    logger.info(f"  AMP: {training_config.use_amp} ({training_config.amp_dtype})")
    logger.info(f"  EMA: {training_config.use_ema} (decay={training_config.ema_decay})")
    logger.info(f"  ACT: {training_config.use_act}")
    logger.info("=" * 60)

    # Load or train tokenizer (only on rank 0)
    output_dir = Path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    tokenizer_path = None
    tokenizer_ready_file = output_dir / "tokenizer" / ".ready"

    if args.tokenizer_path:
        tokenizer_path = args.tokenizer_path
    else:
        # Check if tokenizer already exists in output_dir
        existing_tokenizer = output_dir / "tokenizer" / "tokenizer.model"
        if existing_tokenizer.exists():
            tokenizer_path = existing_tokenizer
        elif is_main_process():
            # Train tokenizer from training data (only rank 0)
            # Remove ready file if exists (from previous run)
            if tokenizer_ready_file.exists():
                tokenizer_ready_file.unlink()

            logger.info("No tokenizer specified. Training tokenizer from dataset...")
            logger.info("(Other ranks will wait for tokenizer training to complete)")
            vocab_size = args.vocab_size or model_config.vocab_size
            train_tokenizer_from_data(
                data_paths=args.train_data,
                output_dir=output_dir,
                vocab_size=vocab_size,
            )
            tokenizer_path = output_dir / "tokenizer" / "tokenizer.model"

            # Signal that tokenizer is ready
            tokenizer_ready_file.touch()

    # Wait for rank 0 to finish tokenizer training using file-based sync
    # This avoids NCCL timeout during long tokenizer training
    if is_distributed() and not args.tokenizer_path:
        tokenizer_model_path = output_dir / "tokenizer" / "tokenizer.model"
        if not is_main_process():
            logger.info("Waiting for tokenizer training to complete...")
            while not tokenizer_model_path.exists():
                time.sleep(5)
            # Wait a bit more to ensure file is fully written
            time.sleep(2)
            logger.info("Tokenizer ready, continuing...")

    # Now all ranks load the tokenizer
    if tokenizer_path is None:
        tokenizer_path = output_dir / "tokenizer" / "tokenizer.model"

    tokenizer = TRMTokenizer(tokenizer_path)
    logger.info(f"Loaded tokenizer from {tokenizer_path}")

    # Sync model vocab_size with tokenizer's actual vocabulary size
    model_config.vocab_size = len(tokenizer)
    logger.info(f"Updated model vocab_size to match tokenizer: {model_config.vocab_size}")

    # Sync all ranks before continuing
    if is_distributed():
        barrier()

    # Load datasets from multiple files
    all_datasets = []
    tool_name_to_id = None  # Will be set from first dataset

    for i, data_path in enumerate(args.train_data):
        dataset = TRMToolCallingDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_len=model_config.max_seq_len,
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

    # Get first dataset for shared info
    first_dataset = all_datasets[0]

    train_dataset = full_dataset
    val_dataset = None

    if args.val_data:
        # Use separate validation file
        val_dataset = TRMToolCallingDataset(
            data_path=args.val_data,
            tokenizer=tokenizer,
            max_seq_len=model_config.max_seq_len,
            tool_name_to_id=tool_name_to_id,
        )
        logger.info(f"Loaded validation dataset: {len(val_dataset)} samples")
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

    # Update model config with actual tool count
    model_config.num_tools = len(tool_name_to_id)
    logger.info(f"Number of tools: {model_config.num_tools}")
    for tool_name, tool_id in tool_name_to_id.items():
        logger.info(f"  Tool '{tool_name}' (id={tool_id})")

    # Log dataset statistics
    if is_main_process():
        stats = first_dataset.get_label_statistics()
        logger.info(f"Dataset statistics:")
        logger.info(f"  Total samples: {stats['total_samples']}")
        logger.info(f"  Tool calls: {stats['tool_call']} ({stats['tool_call_ratio']:.1%})")
        logger.info(f"  Direct answers: {stats['direct_answer']}")

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
        # Note: static_graph=True causes hangs with Mamba2's custom CUDA kernels
        # Use find_unused_parameters for hybrid mode, static_graph for standard mode
        if model_config.use_hybrid_block:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,  # Required for Mamba2 compatibility
            )
            logger.info("Wrapped model with DDP (find_unused_parameters=True for hybrid mode)")
        else:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                static_graph=True,  # Graph structure is fixed, enables optimizations
            )
            logger.info("Wrapped model with DDP (static_graph=True)")

    # Warmup for Mamba2 Triton kernels (compile JIT kernels before training)
    if model_config.use_hybrid_block and model_config.mamba_version == 2:
        logger.info("Running Mamba2 warmup to compile Triton kernels...")
        try:
            # Get a sample batch for warmup
            warmup_batch = next(iter(train_dataloader))
            warmup_batch = {k: v.to(device) for k, v in warmup_batch.items()}

            # Run forward pass to trigger kernel compilation
            with torch.no_grad():
                _ = model.module.forward(
                    input_ids=warmup_batch["input_ids"],
                    attention_mask=warmup_batch["attention_mask"],
                    role_ids=warmup_batch["role_ids"],
                ) if is_distributed() else model.forward(
                    input_ids=warmup_batch["input_ids"],
                    attention_mask=warmup_batch["attention_mask"],
                    role_ids=warmup_batch["role_ids"],
                )

            # Sync CUDA to ensure kernels are compiled
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            logger.info("Mamba2 warmup completed successfully!")
        except Exception as e:
            logger.error(f"Mamba2 warmup failed: {e}")
            logger.error("Try using --no-mamba-cuda-kernels or --mamba-version 1")
            raise

    # Create trainer
    trainer = TRMTrainer(
        model=model,
        config=model_config,
        training_config=training_config,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        device=device,
        tool_names=list(tool_name_to_id.keys()),
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
