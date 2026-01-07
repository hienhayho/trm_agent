"""TRM Sudoku Training Script.

Train TRMForSudoku on Sudoku puzzles with deep supervision.

Usage:
    uv run python tools/train_sudoku.py \
        --config configs/sudoku.yaml \
        --data-dir data/sudoku \
        --output-dir outputs/sudoku

    # With custom hyperparameters:
    uv run python tools/train_sudoku.py \
        --data-dir data/sudoku \
        --hidden-size 256 \
        --num-layers 2 \
        --batch-size 64 \
        --epochs 100 \
        --output-dir outputs/sudoku

    # Distributed training with torchrun:
    torchrun --standalone --nproc_per_node=2 tools/train_sudoku.py \
        --config configs/sudoku.yaml \
        --data-dir data/sudoku \
        --output-dir outputs/sudoku
"""

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from rich.pretty import pprint
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from trm_agent.data import SudokuDataset
from trm_agent.models import TRMForSudoku, SudokuConfig
from trm_agent.models.ema import EMA
from trm_agent.utils import (
    get_logger,
    setup_distributed,
    cleanup_distributed,
    is_distributed,
    is_main_process,
    get_world_size,
    barrier,
    all_reduce_sum,
)

logger = get_logger(__name__)


def compute_loss(
    outputs: list,
    targets: torch.Tensor,
    given_mask: torch.Tensor,
    vocab_size: int = 11,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute loss over all supervision steps.

    Args:
        outputs: List of SudokuOutput from N_sup steps
        targets: [batch, 81] ground truth values
        given_mask: [batch, 81] True where cell was given (don't compute loss)
        vocab_size: Number of classes

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    total_loss = 0.0
    total_aux_loss = 0.0
    total_correct = 0
    total_cells = 0

    for output in outputs:
        logits = output.logits  # [batch, 81, vocab_size]

        # Compute cross-entropy loss per cell
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            reduction="none",
        ).view_as(targets)

        # Mask given cells (only compute loss on empty cells)
        empty_mask = ~given_mask
        masked_loss = loss * empty_mask.float()

        # Average over non-given cells
        num_empty = empty_mask.sum()
        if num_empty > 0:
            step_loss = masked_loss.sum() / num_empty
        else:
            step_loss = torch.tensor(0.0, device=loss.device)

        total_loss = total_loss + step_loss

        # Add MoE auxiliary loss (if available)
        if output.aux_loss is not None:
            total_aux_loss = total_aux_loss + output.aux_loss

        # Compute accuracy (on empty cells only)
        preds = logits.argmax(dim=-1)
        correct = ((preds == targets) & empty_mask).sum()
        total_correct += correct.item()
        total_cells += num_empty.item()

    # Average over supervision steps
    avg_loss = total_loss / len(outputs)
    avg_aux_loss = total_aux_loss / len(outputs) if len(outputs) > 0 else 0.0

    # Combine main loss and auxiliary loss
    combined_loss = avg_loss + avg_aux_loss

    # Compute accuracy
    accuracy = total_correct / max(total_cells, 1)

    metrics = {
        "loss": avg_loss.item(),
        "aux_loss": avg_aux_loss.item() if isinstance(avg_aux_loss, torch.Tensor) else avg_aux_loss,
        "accuracy": accuracy,
        "num_steps": len(outputs),
    }

    return combined_loss, metrics


def evaluate(
    model: nn.Module,
    raw_model: TRMForSudoku,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on dataset.

    Args:
        model: Model (possibly wrapped with DDP)
        raw_model: Unwrapped model for accessing config
        dataloader: Evaluation data loader
        device: Device to use

    Returns:
        Dict with metrics (loss, accuracy, puzzle_accuracy)
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_cells = 0
    total_puzzles_correct = 0
    total_puzzles = 0

    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc="Evaluating", leave=False, disable=not is_main_process()
        ):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            given_mask = batch["given_mask"].to(device)

            # Forward pass (single output for inference)
            output = raw_model(inputs, return_all_steps=False)

            # Compute loss
            logits = output.logits
            loss = F.cross_entropy(
                logits.view(-1, raw_model.config.vocab_size),
                targets.view(-1),
                reduction="none",
            ).view_as(targets)

            empty_mask = ~given_mask
            masked_loss = loss * empty_mask.float()
            num_empty = empty_mask.sum()
            if num_empty > 0:
                total_loss += masked_loss.sum().item()
                total_cells += num_empty.item()

            # Compute accuracy
            preds = logits.argmax(dim=-1)
            correct = (preds == targets) & empty_mask
            total_correct += correct.sum().item()

            # Puzzle-level accuracy (all cells correct)
            puzzle_correct = correct.all(dim=1) | (~empty_mask).all(dim=1)
            total_puzzles_correct += puzzle_correct.sum().item()
            total_puzzles += inputs.size(0)

    # Gather metrics across all GPUs in distributed training
    stats = torch.tensor(
        [total_loss, total_correct, total_cells, total_puzzles_correct, total_puzzles],
        dtype=torch.float64,
        device=device,
    )
    all_reduce_sum(stats)
    total_loss, total_correct, total_cells, total_puzzles_correct, total_puzzles = stats.tolist()

    metrics = {
        "loss": total_loss / max(total_cells, 1),
        "accuracy": total_correct / max(total_cells, 1),
        "puzzle_accuracy": total_puzzles_correct / max(total_puzzles, 1),
    }

    return metrics


def train(
    model: nn.Module,
    raw_model: TRMForSudoku,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    config: dict,
    device: torch.device,
    output_dir: Path,
    train_sampler: DistributedSampler | None = None,
    eval_steps: int | None = None,
    resume_path: str | None = None,
    no_epoch_eval: bool = False,
):
    """Training loop.

    Args:
        model: Model (possibly wrapped with DDP)
        raw_model: Unwrapped model for accessing config and saving
        train_loader: Training data loader
        eval_loader: Evaluation data loader
        config: Training configuration
        device: Device to use
        output_dir: Output directory
        train_sampler: DistributedSampler for setting epoch (DDP only)
        eval_steps: Evaluate every N steps (None = only at epoch end)
        resume_path: Path to checkpoint to resume from
        no_epoch_eval: Disable epoch-end evaluation
    """
    # Optimizer
    optimizer = AdamW(
        raw_model.parameters(),
        lr=config["learning_rate"],
        betas=(config.get("beta1", 0.9), config.get("beta2", 0.95)),
        weight_decay=config.get("weight_decay", 0.1),
    )

    # Scheduler
    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = config.get("warmup_steps", 1000)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # EMA (use raw_model, not DDP-wrapped model)
    ema = None
    if config.get("use_ema", True):
        ema = EMA(raw_model, decay=config.get("ema_decay", 0.999))

    # AMP
    use_amp = config.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Training loop
    global_step = 0
    start_epoch = 0
    best_accuracy = 0.0

    # Resume from checkpoint
    if resume_path is not None:
        logger.info(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

        raw_model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer/scheduler if available (full checkpoint)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            logger.warning("  No optimizer state in checkpoint, starting fresh")

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            logger.warning("  No scheduler state in checkpoint, starting fresh")

        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", 0)
        best_accuracy = checkpoint.get("accuracy", 0.0)

        if ema is not None and "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"])

        logger.info(f"  Resumed from epoch {start_epoch}, step {global_step}")
        logger.info(f"  Best accuracy so far: {best_accuracy:.2%}")

    logger.info(f"Starting training for {config['num_epochs']} epochs")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  AMP: {use_amp}")

    progress_bar = tqdm(
        total=total_steps,
        initial=global_step,
        desc="Training",
        disable=not is_main_process(),
    )

    for epoch in range(start_epoch, config["num_epochs"]):
        # Set epoch for DistributedSampler (for proper shuffling)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            given_mask = batch["given_mask"].to(device)

            # Forward pass with AMP
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                outputs = model(inputs, return_all_steps=True)
                loss, metrics = compute_loss(
                    outputs, targets, given_mask, raw_model.config.vocab_size
                )

            # Backward pass
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    raw_model.parameters(), config.get("max_grad_norm", 1.0)
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    raw_model.parameters(), config.get("max_grad_norm", 1.0)
                )
                optimizer.step()

            scheduler.step()

            if ema is not None:
                ema.update()

            # Update metrics
            epoch_loss += metrics["loss"]
            epoch_accuracy += metrics["accuracy"]
            num_batches += 1
            global_step += 1

            # Update progress bar
            progress_bar.update(1)
            if global_step % 10 == 0:
                postfix = {
                    "epoch": f"{epoch + 1}/{config['num_epochs']}",
                    "loss": f"{metrics['loss']:.4f}",
                    "acc": f"{metrics['accuracy']:.2%}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
                # Show aux_loss if non-zero
                if metrics.get("aux_loss", 0) > 0:
                    postfix["aux"] = f"{metrics['aux_loss']:.4f}"
                progress_bar.set_postfix(**postfix)

            # Step-based evaluation
            if eval_steps is not None and global_step % eval_steps == 0:
                barrier()
                if ema is not None:
                    ema.apply_shadow()

                eval_metrics = evaluate(model, raw_model, eval_loader, device)
                logger.info(
                    f"Step {global_step}: eval_loss={eval_metrics['loss']:.4f}, "
                    f"eval_acc={eval_metrics['accuracy']:.2%}, "
                    f"puzzle_acc={eval_metrics['puzzle_accuracy']:.2%}"
                )

                if ema is not None:
                    ema.restore()

                # Save best model
                if is_main_process() and eval_metrics["accuracy"] > best_accuracy:
                    best_accuracy = eval_metrics["accuracy"]
                    best_state = {
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "accuracy": best_accuracy,
                        "config": raw_model.config.to_dict(),
                    }
                    if ema is not None:
                        best_state["ema_state_dict"] = ema.state_dict()
                    torch.save(best_state, output_dir / "best_model.pt")
                    logger.info(f"  Saved best model (accuracy={best_accuracy:.2%})")

                model.train()  # Back to training mode

        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        logger.info(
            f"Epoch {epoch + 1}/{config['num_epochs']}: "
            f"loss={avg_loss:.4f}, accuracy={avg_accuracy:.2%}"
        )

        # Epoch-end evaluation (skip if --no-epoch-eval)
        if not no_epoch_eval:
            barrier()

            if ema is not None:
                ema.apply_shadow()

            eval_metrics = evaluate(model, raw_model, eval_loader, device)
            logger.info(
                f"  Eval: loss={eval_metrics['loss']:.4f}, "
                f"accuracy={eval_metrics['accuracy']:.2%}, "
                f"puzzle_accuracy={eval_metrics['puzzle_accuracy']:.2%}"
            )

            if ema is not None:
                ema.restore()

            # Save best model (only on main process)
            if is_main_process() and eval_metrics["accuracy"] > best_accuracy:
                best_accuracy = eval_metrics["accuracy"]
                best_state = {
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "accuracy": best_accuracy,
                    "config": raw_model.config.to_dict(),
                }
                if ema is not None:
                    best_state["ema_state_dict"] = ema.state_dict()
                torch.save(best_state, output_dir / "best_model.pt")
                logger.info(f"  Saved best model (accuracy={best_accuracy:.2%})")

        # Save checkpoint (only on main process)
        if is_main_process() and (epoch + 1) % config.get("save_interval", 10) == 0:
            save_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            state = {
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "config": raw_model.config.to_dict(),
            }
            if ema is not None:
                state["ema_state_dict"] = ema.state_dict()
            torch.save(state, save_path)

    progress_bar.close()

    # Final evaluation
    barrier()
    logger.info("Running final evaluation...")

    if ema is not None:
        ema.apply_shadow()

    eval_metrics = evaluate(model, raw_model, eval_loader, device)
    logger.info(
        f"Final Eval: loss={eval_metrics['loss']:.4f}, "
        f"accuracy={eval_metrics['accuracy']:.2%}, "
        f"puzzle_accuracy={eval_metrics['puzzle_accuracy']:.2%}"
    )

    if ema is not None:
        ema.restore()

    # Save best model if final is better
    if is_main_process() and eval_metrics["accuracy"] > best_accuracy:
        best_accuracy = eval_metrics["accuracy"]
        best_state = {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": config["num_epochs"],
            "global_step": global_step,
            "accuracy": best_accuracy,
            "config": raw_model.config.to_dict(),
        }
        if ema is not None:
            best_state["ema_state_dict"] = ema.state_dict()
        torch.save(best_state, output_dir / "best_model.pt")
        logger.info(f"  Saved best model (accuracy={best_accuracy:.2%})")

    # Save final model (only on main process)
    if is_main_process():
        torch.save(
            {
                "model_state_dict": raw_model.state_dict(),
                "config": raw_model.config.to_dict(),
                "epoch": config["num_epochs"],
            },
            output_dir / "final_model.pt",
        )
        logger.info(f"Training complete! Best accuracy: {best_accuracy:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Train TRM for Sudoku")

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to preprocessed Sudoku data directory",
    )
    parser.add_argument(
        "--may_dua_can_card_thi_nhan_anh_nha",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sudoku",
        help="Output directory for models and logs",
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )

    # Model
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--n-latent-recursion", type=int, default=None)
    parser.add_argument("--T-deep-recursion", type=int, default=None)
    parser.add_argument("--N-supervision", type=int, default=None)
    parser.add_argument("--no-trm-loop", action="store_true")

    # MoE (DeepSeek-V3 style)
    parser.add_argument(
        "--moe-shared-experts",
        type=int,
        default=None,
        help="Number of shared experts (always active)",
    )
    parser.add_argument(
        "--moe-routed-experts",
        type=int,
        default=None,
        help="Number of routed experts (top-k selection)",
    )
    parser.add_argument(
        "--moe-top-k",
        type=int,
        default=None,
        help="Top-k experts per token",
    )
    parser.add_argument(
        "--moe-intermediate-size",
        type=int,
        default=None,
        help="Expert MLP hidden size",
    )
    parser.add_argument(
        "--no-moe-sigmoid-gating",
        action="store_true",
        help="Use softmax instead of sigmoid gating",
    )
    parser.add_argument(
        "--moe-bias-update-speed",
        type=float,
        default=None,
        help="Bias update speed for load balancing",
    )
    parser.add_argument(
        "--moe-seq-aux-loss",
        type=float,
        default=None,
        help="Sequence-wise auxiliary loss weight (0 = disabled)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Evaluate every N steps (default: only at epoch end)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--no-epoch-eval",
        action="store_true",
        help="Disable epoch-end evaluation (only eval at --eval-steps and end of training)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Save checkpoint every N epochs (default: 10)",
    )

    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, device = setup_distributed()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load config
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    model_config = config.get("model", {})
    training_config = config.get("training", {})

    # Override with CLI arguments
    if args.hidden_size:
        model_config["hidden_size"] = args.hidden_size
    if args.num_layers:
        model_config["num_layers"] = args.num_layers
    if args.num_heads:
        model_config["num_heads"] = args.num_heads
    if args.n_latent_recursion:
        model_config["n_latent_recursion"] = args.n_latent_recursion
    if args.T_deep_recursion:
        model_config["T_deep_recursion"] = args.T_deep_recursion
    if args.N_supervision:
        model_config["N_supervision"] = args.N_supervision
    if args.no_trm_loop:
        model_config["use_trm_loop"] = False

    # MoE CLI overrides
    if args.moe_shared_experts is not None:
        model_config["moe_num_shared_experts"] = args.moe_shared_experts
    if args.moe_routed_experts is not None:
        model_config["moe_num_routed_experts"] = args.moe_routed_experts
    if args.moe_top_k is not None:
        model_config["moe_top_k"] = args.moe_top_k
    if args.moe_intermediate_size is not None:
        model_config["moe_intermediate_size"] = args.moe_intermediate_size
    if args.no_moe_sigmoid_gating:
        model_config["moe_use_sigmoid_gating"] = False
    if args.moe_bias_update_speed is not None:
        model_config["moe_bias_update_speed"] = args.moe_bias_update_speed
    if args.moe_seq_aux_loss is not None:
        model_config["moe_seq_aux_loss_weight"] = args.moe_seq_aux_loss

    if args.epochs:
        training_config["num_epochs"] = args.epochs
    if args.batch_size:
        training_config["batch_size"] = args.batch_size
    if args.learning_rate:
        training_config["learning_rate"] = args.learning_rate
    if args.warmup_steps:
        training_config["warmup_steps"] = args.warmup_steps
    if args.no_amp:
        training_config["use_amp"] = False
    if args.save_interval:
        training_config["save_interval"] = args.save_interval

    # Defaults
    training_config.setdefault("num_epochs", 100)
    training_config.setdefault("batch_size", 64)
    training_config.setdefault("learning_rate", 1e-4)
    training_config.setdefault("warmup_steps", 1000)
    training_config.setdefault("use_amp", True)
    training_config.setdefault("use_ema", True)
    training_config.setdefault("ema_decay", 0.999)
    training_config.setdefault("save_interval", 10)

    # Create output directory (only on main process)
    output_dir = Path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier()  # Wait for directory creation

    # Device is already set by setup_distributed()
    logger.info(f"Using device: {device}")
    if is_distributed():
        logger.info(f"  Distributed training with {world_size} GPUs")

    # Load datasets
    logger.info(f"Loading data from {args.data_dir}...")
    train_dataset = SudokuDataset(args.data_dir, split="train")
    eval_dataset = SudokuDataset(args.data_dir, split="test")
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Eval: {len(eval_dataset)} samples")

    # Create data loaders with DistributedSampler if using DDP
    train_sampler = None
    eval_sampler = None
    if is_distributed():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        sampler=eval_sampler,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    sudoku_config = SudokuConfig(**model_config)
    raw_model = TRMForSudoku(sudoku_config)
    raw_model = raw_model.to(device)

    # Wrap with DDP if distributed
    if is_distributed():
        local_rank = device.index if device.index is not None else 0
        model = DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,  # MoE routing may skip some experts
        )
    else:
        model = raw_model

    num_params = sum(p.numel() for p in raw_model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Print configs and args (main process only)
    if is_main_process():
        print("\n" + "=" * 60)
        print("Arguments:")
        pprint(vars(args))
        print("\nModel Configuration:")
        pprint(sudoku_config.to_dict())
        print("\nTraining Configuration:")
        pprint(training_config)
        print("=" * 60 + "\n")

    # Log config
    logger.info("=" * 60)
    logger.info("Model Configuration:")
    logger.info(f"  hidden_size: {sudoku_config.hidden_size}")
    logger.info(f"  num_layers: {sudoku_config.num_layers}")
    logger.info(f"  num_heads: {sudoku_config.num_heads}")
    logger.info(f"  use_trm_loop: {sudoku_config.use_trm_loop}")
    if sudoku_config.use_trm_loop:
        logger.info(f"  n_latent_recursion: {sudoku_config.n_latent_recursion}")
        logger.info(f"  T_deep_recursion: {sudoku_config.T_deep_recursion}")
        logger.info(f"  N_supervision: {sudoku_config.N_supervision}")
    logger.info("=" * 60)
    logger.info("Training Configuration:")
    logger.info(f"  epochs: {training_config['num_epochs']}")
    logger.info(f"  batch_size: {training_config['batch_size']}")
    logger.info(f"  learning_rate: {training_config['learning_rate']}")
    logger.info("=" * 60)

    # Save config (only on main process)
    if is_main_process():
        sudoku_config.to_yaml(output_dir / "config.yaml")

    # Train
    try:
        train(
            model,
            raw_model,
            train_loader,
            eval_loader,
            training_config,
            device,
            output_dir,
            train_sampler,
            args.eval_steps,
            args.resume,
            args.no_epoch_eval,
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
