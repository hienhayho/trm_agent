"""
TRM Evaluation Script.

Evaluate trained TRM model on test data. Supports single GPU and multi-GPU inference.

Usage:
    # Single GPU:
    uv run python tools/test.py \
        --checkpoint outputs/checkpoint_epoch_1.pt \
        --test-data data/test.json \
        --tokenizer-path outputs/tokenizer/tokenizer.model

    # Multi-GPU with torchrun:
    uv run torchrun --nproc_per_node=2 tools/test.py \
        --checkpoint outputs/checkpoint_epoch_1.pt \
        --test-data data/test.json \
        --tokenizer-path outputs/tokenizer/tokenizer.model \
        --batch-size 16

    # With tools and system prompt:
    uv run python tools/test.py \
        --checkpoint outputs/checkpoint_epoch_1.pt \
        --test-data data/test.json \
        --tokenizer-path outputs/tokenizer/tokenizer.model \
        --tools data/tools.json \
        --system-prompt data/system_prompt.txt \
        --batch-size 16

    # Remove <think>...</think> tags from assistant responses:
    uv run python tools/test.py \
        --checkpoint outputs/checkpoint_epoch_1.pt \
        --test-data data/test.json \
        --tokenizer-path outputs/tokenizer/tokenizer.model \
        --remove-think

    # Save detailed results (conversation, prediction, target) to JSON files:
    uv run python tools/test.py \
        --checkpoint outputs/checkpoint_epoch_1.pt \
        --test-data data/test.json \
        --tokenizer-path outputs/tokenizer/tokenizer.model \
        --output-dir outputs/test_results/
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from trm_agent.data import TRMTokenizer
from trm_agent.models import TRMConfig, TRMForToolCalling
from trm_agent.utils import (
    cleanup_distributed,
    get_logger,
    get_world_size,
    is_distributed,
    is_main_process,
    setup_distributed,
)

logger = get_logger(__name__)
console = Console()


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[TRMForToolCalling, TRMConfig]:
    """Load model from checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config_dict = checkpoint.get("config", {})
    config = TRMConfig(**config_dict)

    model = TRMForToolCalling(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(
        f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    return model, config


def load_tools(tools_path: Optional[Path]) -> list[dict]:
    """Load tool definitions from JSON file."""
    if tools_path is None or not tools_path.exists():
        return []
    with open(tools_path, "r", encoding="utf-8") as f:
        return json.load(f)


def remove_think_tags(text: str) -> str:
    """Remove <think>...</think> tags from text."""
    # Remove <think>...</think> including content
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Also handle unclosed </think> at the end
    text = re.sub(r"</think>\s*$", "", text)
    # Clean up extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_system_prompt(prompt_path: Optional[Path]) -> str:
    """Load system prompt from text file."""
    if prompt_path is None or not prompt_path.exists():
        return ""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def parse_test_data(
    test_data: list[list[dict]],
    tools: list[dict],
    system_prompt: str,
    remove_think: bool = False,
) -> list[dict]:
    """Parse test conversations into evaluation samples."""
    samples = []

    for conv_idx, conversation in enumerate(test_data):
        history = []

        if system_prompt:
            history.append({"role": "system", "content": system_prompt})

        for i, msg in enumerate(conversation):
            role = msg["role"]
            content = msg["content"]

            if role == "tool_call":
                sample = {
                    "conv_idx": conv_idx,
                    "turn_idx": i,
                    "history": history.copy(),
                    "tools": tools,
                    "ground_truth": {
                        "decision": "tool_call",
                        "tool_name": content["name"],
                        "tool_args": content.get("arguments", {}),
                    },
                }
                samples.append(sample)
                history.append({"role": "tool_call", "content": content})

            elif role == "tool_response":
                history.append({"role": "tool_response", "content": content})

            elif role == "assistant":
                # Clean think tags if requested
                if remove_think and isinstance(content, str):
                    content = remove_think_tags(content)

                is_direct_answer = True
                if i + 1 < len(conversation):
                    next_msg = conversation[i + 1]
                    if next_msg["role"] == "tool_call":
                        is_direct_answer = False

                if is_direct_answer:
                    sample = {
                        "conv_idx": conv_idx,
                        "turn_idx": i,
                        "history": history.copy(),
                        "tools": tools,
                        "ground_truth": {
                            "decision": "direct_answer",
                            "tool_name": None,
                            "tool_args": {},
                        },
                    }
                    samples.append(sample)

                history.append({"role": "assistant", "content": content})

            elif role == "user":
                history.append({"role": "user", "content": content})

    return samples


def build_unified_param_mapping(
    tools: list[dict],
    slot_fields: list[str],
) -> tuple[list[str], int, dict[str, int], dict[str, list[int]]]:
    """Build unified parameter mapping for the new architecture.

    Returns:
        - unified_fields: Combined list of slot_fields + tool_param_fields
        - num_slots: Number of slot fields
        - param_name_to_idx: Mapping from param name to unified index
        - tool_param_mask: Per-tool mask for which params are valid
    """
    slot_set = set(slot_fields)

    # Collect all unique params from tools (excluding slots)
    all_params: set[str] = set()
    tool_params: dict[str, set[str]] = {}

    for tool in tools:
        func = tool.get("function", {})
        tool_name = func.get("name", "")
        params = func.get("parameters", {}).get("properties", {})

        if tool_name:
            tool_params[tool_name] = set(params.keys())
            # Only add params that are not already slots
            all_params.update(p for p in params.keys() if p not in slot_set)

    # Build unified fields
    tool_param_fields = sorted(all_params)
    unified_fields = slot_fields + tool_param_fields
    num_slots = len(slot_fields)

    # Build param_name_to_idx (unified index, offset by num_slots)
    param_name_to_idx = {
        name: num_slots + idx for idx, name in enumerate(tool_param_fields)
    }

    # Build per-tool param mask
    tool_param_mask = {}
    for tool_name, params in tool_params.items():
        mask = [0] * len(tool_param_fields)
        for param in params:
            if param in param_name_to_idx:
                # Get index within tool_param_fields (not unified index)
                param_idx = param_name_to_idx[param] - num_slots
                mask[param_idx] = 1
        tool_param_mask[tool_name] = mask

    return unified_fields, num_slots, param_name_to_idx, tool_param_mask


def find_span_in_text(value: str, text: str) -> tuple[int, int] | None:
    """Find character span of value in text."""
    if not value:
        return None

    idx = text.find(value)
    if idx >= 0:
        return (idx, idx + len(value))

    idx = text.lower().find(value.lower())
    if idx >= 0:
        return (idx, idx + len(value))

    return None


class TestDataset(Dataset):
    """Dataset for batched test inference."""

    def __init__(
        self,
        samples: list[dict],
        tokenizer: TRMTokenizer,
        max_seq_len: int,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        encoded = self.tokenizer.encode_conversation_with_offsets(
            sample["history"],
            max_length=self.max_seq_len,
        )

        return {
            "idx": idx,
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "role_ids": torch.tensor(encoded["role_ids"], dtype=torch.long),
            "full_text": encoded["full_text"],
            "offsets": encoded["offsets"],
        }


def collate_test_batch(batch: list[dict]) -> dict:
    """Collate test samples into a batch."""
    batch_size = len(batch)

    # Find max sequence length
    max_len = max(len(b["input_ids"]) for b in batch)

    # Pad sequences
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    role_ids = torch.zeros(batch_size, max_len, dtype=torch.long)

    indices = []
    full_texts = []
    offsets_list = []

    for i, b in enumerate(batch):
        seq_len = len(b["input_ids"])
        input_ids[i, :seq_len] = b["input_ids"]
        attention_mask[i, :seq_len] = b["attention_mask"]
        role_ids[i, :seq_len] = b["role_ids"]
        indices.append(b["idx"])
        full_texts.append(b["full_text"])
        offsets_list.append(b["offsets"])

    return {
        "indices": indices,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "role_ids": role_ids,
        "full_texts": full_texts,
        "offsets_list": offsets_list,
    }


@torch.no_grad()
def run_batched_inference(
    model: TRMForToolCalling,
    dataloader: DataLoader,
    samples: list[dict],
    device: torch.device,
    tool_name_to_id: dict[str, int],
    param_name_to_idx: dict[str, int],
    num_slots: int,
    tool_param_mask: dict[str, list[int]],
) -> list[tuple[int, dict]]:
    """Run batched inference with unified field extraction.

    Args:
        model: TRM model
        dataloader: Test data loader
        samples: List of test samples
        device: Device to run on
        tool_name_to_id: Tool name to ID mapping
        param_name_to_idx: Param name to unified index mapping
        num_slots: Number of slot fields (first num_slots in unified)
        tool_param_mask: Per-tool mask for valid params

    Returns:
        List of (global_index, prediction) tuples for samples processed by this rank.
    """
    id_to_name = {v: k for k, v in tool_name_to_id.items()}
    idx_to_param = {v: k for k, v in param_name_to_idx.items()}
    local_predictions = []

    # Only show progress bar on main process
    iterator = (
        tqdm(dataloader, desc="Running inference") if is_main_process() else dataloader
    )

    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        role_ids = batch["role_ids"].to(device)
        indices = batch["indices"]
        full_texts = batch["full_texts"]
        offsets_list = batch["offsets_list"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            role_ids=role_ids,
            return_all_steps=False,
        )

        batch_size = input_ids.size(0)

        for i in range(batch_size):
            idx = indices[i]
            full_text = full_texts[i]
            token_offsets = offsets_list[i]

            decision_prob = torch.sigmoid(outputs.decision_logits[i]).item()
            decision = "tool_call" if decision_prob > 0.5 else "direct_answer"

            result = {
                "decision": decision,
                "decision_prob": decision_prob,
                "tool_name": None,
                "tool_args": {},
                "arg_spans": {},
            }

            if decision == "tool_call":
                tool_pred_idx = outputs.tool_logits[i].argmax().item()
                tool_name = id_to_name.get(tool_pred_idx, f"tool_{tool_pred_idx}")
                result["tool_name"] = tool_name

                # Extract tool params using unified indices
                if tool_name in tool_param_mask:
                    mask = tool_param_mask[tool_name]
                    num_unified = outputs.param_start_logits.size(2)

                    for param_rel_idx, is_valid in enumerate(mask):
                        if is_valid:
                            unified_idx = num_slots + param_rel_idx
                            if unified_idx < num_unified:
                                # Get param name
                                param_name = idx_to_param.get(unified_idx)
                                if param_name is None:
                                    continue

                                start_pos = (
                                    outputs.param_start_logits[i, :, unified_idx].argmax().item()
                                )
                                end_pos = (
                                    outputs.param_end_logits[i, :, unified_idx].argmax().item()
                                )

                                if start_pos < len(token_offsets) and end_pos < len(token_offsets):
                                    char_start = token_offsets[start_pos][0]
                                    char_end = token_offsets[end_pos][1]
                                    arg_value = full_text[char_start:char_end]
                                    result["tool_args"][param_name] = arg_value
                                    result["arg_spans"][param_name] = (start_pos, end_pos)

            local_predictions.append((idx, result))

    return local_predictions


def gather_predictions(
    local_predictions: list[tuple[int, dict]],
    total_samples: int,
) -> list[dict]:
    """Gather predictions from all ranks.

    Args:
        local_predictions: List of (index, prediction) from this rank
        total_samples: Total number of samples across all ranks

    Returns:
        Complete list of predictions ordered by index (only on rank 0)
    """
    if not is_distributed():
        # Single GPU: directly build predictions list
        predictions = [None] * total_samples
        for idx, pred in local_predictions:
            predictions[idx] = pred
        return predictions

    # Gather from all ranks
    world_size = get_world_size()

    # Serialize local predictions to JSON for gathering
    local_data = json.dumps(local_predictions)

    # Gather all data sizes first
    local_size = torch.tensor([len(local_data)], dtype=torch.long, device="cuda")
    all_sizes = [
        torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(world_size)
    ]
    dist.all_gather(all_sizes, local_size)

    max_size = max(s.item() for s in all_sizes)

    # Pad local data to max size
    padded_data = local_data.ljust(max_size)
    local_tensor = torch.tensor(
        [ord(c) for c in padded_data], dtype=torch.uint8, device="cuda"
    )

    # Gather all tensors
    all_tensors = [
        torch.zeros(max_size, dtype=torch.uint8, device="cuda")
        for _ in range(world_size)
    ]
    dist.all_gather(all_tensors, local_tensor)

    if is_main_process():
        # Decode and merge predictions
        predictions = [None] * total_samples

        for i, (tensor, size) in enumerate(zip(all_tensors, all_sizes)):
            data_str = "".join(chr(c) for c in tensor[: size.item()].cpu().tolist())
            rank_predictions = json.loads(data_str)

            for idx, pred in rank_predictions:
                predictions[idx] = pred

        return predictions
    else:
        return []


def compute_metrics(
    predictions: list[dict],
    ground_truths: list[dict],
    samples: list[dict],
    tool_names: list[str],
    tool_param_mask: dict[str, list[int]],
    tokenizer: TRMTokenizer,
    config: TRMConfig,
) -> dict:
    """Compute evaluation metrics."""
    decision_tp = 0
    decision_fp = 0
    decision_fn = 0
    decision_tn = 0

    tool_correct = 0
    tool_total = 0

    per_tool_correct = {name: 0 for name in tool_names}
    per_tool_total = {name: 0 for name in tool_names}

    arg_span_correct = 0
    arg_span_total = 0

    for pred, gt, sample in zip(predictions, ground_truths, samples):
        pred_is_tool = pred["decision"] == "tool_call"
        gt_is_tool = gt["decision"] == "tool_call"

        if pred_is_tool and gt_is_tool:
            decision_tp += 1
        elif pred_is_tool and not gt_is_tool:
            decision_fp += 1
        elif not pred_is_tool and gt_is_tool:
            decision_fn += 1
        else:
            decision_tn += 1

        if gt_is_tool:
            tool_total += 1
            gt_tool = gt["tool_name"]

            if gt_tool in per_tool_total:
                per_tool_total[gt_tool] += 1

            if pred["tool_name"] == gt_tool:
                tool_correct += 1
                if gt_tool in per_tool_correct:
                    per_tool_correct[gt_tool] += 1

            if pred["tool_name"] == gt_tool and gt_tool in tool_param_mask:
                gt_args = gt["tool_args"]

                encoded = tokenizer.encode_conversation_with_offsets(
                    sample["history"],
                    max_length=config.max_seq_len,
                )
                full_text = encoded["full_text"]
                token_offsets = encoded["offsets"]

                for arg_name, gt_value in gt_args.items():
                    if not gt_value:
                        continue

                    arg_span_total += 1

                    gt_value_str = (
                        str(gt_value) if not isinstance(gt_value, str) else gt_value
                    )
                    char_span = find_span_in_text(gt_value_str, full_text)

                    if char_span is not None:
                        gt_start = None
                        gt_end = None
                        for idx, (cs, ce) in enumerate(token_offsets):
                            if cs <= char_span[0] < ce:
                                gt_start = idx
                            if cs < char_span[1] <= ce:
                                gt_end = idx

                        if gt_start is not None and gt_end is not None:
                            if arg_name in pred.get("arg_spans", {}):
                                pred_start, pred_end = pred["arg_spans"][arg_name]
                                if pred_start == gt_start and pred_end == gt_end:
                                    arg_span_correct += 1

    metrics = {}

    total = decision_tp + decision_fp + decision_fn + decision_tn
    metrics["decision_accuracy"] = (
        (decision_tp + decision_tn) / total if total > 0 else 0.0
    )

    precision = (
        decision_tp / (decision_tp + decision_fp)
        if (decision_tp + decision_fp) > 0
        else 0.0
    )
    recall = (
        decision_tp / (decision_tp + decision_fn)
        if (decision_tp + decision_fn) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    metrics["decision_precision"] = precision
    metrics["decision_recall"] = recall
    metrics["decision_f1"] = f1

    metrics["confusion_matrix"] = {
        "tp": decision_tp,
        "fp": decision_fp,
        "fn": decision_fn,
        "tn": decision_tn,
    }

    metrics["tool_accuracy"] = tool_correct / tool_total if tool_total > 0 else 0.0
    metrics["tool_samples"] = tool_total

    metrics["per_tool"] = {}
    for name in tool_names:
        if per_tool_total[name] > 0:
            metrics["per_tool"][name] = {
                "accuracy": per_tool_correct[name] / per_tool_total[name],
                "correct": per_tool_correct[name],
                "total": per_tool_total[name],
            }

    metrics["arg_span_em"] = (
        arg_span_correct / arg_span_total if arg_span_total > 0 else 0.0
    )
    metrics["arg_span_samples"] = arg_span_total

    return metrics


def save_detailed_results(
    samples: list[dict],
    predictions: list[dict],
    output_dir: Path,
    tokenizer: TRMTokenizer,
    max_seq_len: int,
    max_samples_per_file: int = 100,
) -> None:
    """Save detailed test results to JSON files.

    Each JSON file contains up to max_samples_per_file samples with:
    - conversation history
    - prediction (decision, tool_name, tool_args, ...)
    - target (decision, tool_name, tool_args, ...)

    Args:
        samples: List of test samples with history and ground_truth
        predictions: List of model predictions
        output_dir: Directory to save JSON files
        tokenizer: Tokenizer for encoding conversations
        max_seq_len: Maximum sequence length
        max_samples_per_file: Maximum samples per JSON file (default: 100)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track extraction stats
    total_args = 0
    extractable_args = 0

    # Build detailed results
    detailed_results = []
    for idx, (sample, pred) in enumerate(zip(samples, predictions)):
        # Convert arg_spans tuples to lists for JSON serialization
        arg_spans = {}
        for arg_name, span in pred.get("arg_spans", {}).items():
            arg_spans[arg_name] = list(span) if span else None

        # Check if target args can be extracted from conversation
        target_args = sample["ground_truth"].get("tool_args", {})
        target_arg_extractable = {}
        target_arg_spans = {}

        if target_args:
            # Encode conversation to find spans
            encoded = tokenizer.encode_conversation_with_offsets(
                sample["history"],
                max_length=max_seq_len,
            )
            full_text = encoded["full_text"]
            token_offsets = encoded["offsets"]

            for arg_name, arg_value in target_args.items():
                if not arg_value:
                    target_arg_extractable[arg_name] = False
                    target_arg_spans[arg_name] = None
                    continue

                arg_value_str = str(arg_value) if not isinstance(arg_value, str) else arg_value
                char_span = find_span_in_text(arg_value_str, full_text)

                total_args += 1
                if char_span is not None:
                    # Find token spans
                    token_start = None
                    token_end = None
                    for tok_idx, (cs, ce) in enumerate(token_offsets):
                        if cs <= char_span[0] < ce:
                            token_start = tok_idx
                        if cs < char_span[1] <= ce:
                            token_end = tok_idx

                    if token_start is not None and token_end is not None:
                        target_arg_extractable[arg_name] = True
                        target_arg_spans[arg_name] = [token_start, token_end]
                        extractable_args += 1
                    else:
                        target_arg_extractable[arg_name] = False
                        target_arg_spans[arg_name] = None
                else:
                    target_arg_extractable[arg_name] = False
                    target_arg_spans[arg_name] = None

        result = {
            "sample_id": idx,
            "conv_idx": sample.get("conv_idx", -1),
            "turn_idx": sample.get("turn_idx", -1),
            "history": sample["history"],
            "prediction": {
                "decision": pred["decision"],
                "decision_prob": pred["decision_prob"],
                "tool_name": pred["tool_name"],
                "tool_args": pred["tool_args"],
                "arg_spans": arg_spans,
            },
            "target": {
                "decision": sample["ground_truth"]["decision"],
                "tool_name": sample["ground_truth"].get("tool_name"),
                "tool_args": sample["ground_truth"].get("tool_args", {}),
                "arg_extractable": target_arg_extractable,
                "arg_spans": target_arg_spans,
            },
            "correct": {
                "decision": pred["decision"] == sample["ground_truth"]["decision"],
                "tool_name": pred["tool_name"] == sample["ground_truth"].get("tool_name"),
            },
        }
        detailed_results.append(result)

    # Log extraction stats
    non_extractable = total_args - extractable_args
    if total_args > 0:
        logger.info(
            f"Target arg extraction: {extractable_args}/{total_args} extractable "
            f"({non_extractable} args can't be extracted, {non_extractable/total_args*100:.1f}%)"
        )

    # Split into chunks and save
    num_files = (len(detailed_results) + max_samples_per_file - 1) // max_samples_per_file

    for file_idx in range(num_files):
        start_idx = file_idx * max_samples_per_file
        end_idx = min(start_idx + max_samples_per_file, len(detailed_results))
        chunk = detailed_results[start_idx:end_idx]

        file_path = output_dir / f"results_{file_idx:04d}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(detailed_results)} samples to {num_files} files in {output_dir}")


def print_results(metrics: dict):
    """Print evaluation results using rich tables."""
    main_table = Table(
        title="Evaluation Results", show_header=True, header_style="bold blue"
    )
    main_table.add_column("Metric", style="bold")
    main_table.add_column("Value", justify="right")

    main_table.add_row("Decision Accuracy", f"{metrics['decision_accuracy']:.4f}")
    main_table.add_row("Decision Precision", f"{metrics['decision_precision']:.4f}")
    main_table.add_row("Decision Recall", f"{metrics['decision_recall']:.4f}")
    main_table.add_row(
        "Decision F1", f"[bold cyan]{metrics['decision_f1']:.4f}[/bold cyan]"
    )
    main_table.add_row("", "")
    main_table.add_row(
        "Tool Accuracy",
        f"{metrics['tool_accuracy']:.4f} ({metrics['tool_samples']} samples)",
    )
    main_table.add_row(
        "Arg Span EM",
        f"{metrics['arg_span_em']:.4f} ({metrics['arg_span_samples']} args)",
    )

    console.print(main_table)

    cm = metrics["confusion_matrix"]
    cm_table = Table(title="Decision Confusion Matrix", show_header=True)
    cm_table.add_column("Actual \\ Predicted", style="bold")
    cm_table.add_column("tool_call", justify="center", style="cyan")
    cm_table.add_column("direct_answer", justify="center", style="magenta")

    cm_table.add_row(
        "tool_call", f"[green]{cm['tp']}[/green] (TP)", f"[red]{cm['fn']}[/red] (FN)"
    )
    cm_table.add_row(
        "direct_answer",
        f"[red]{cm['fp']}[/red] (FP)",
        f"[green]{cm['tn']}[/green] (TN)",
    )

    console.print(cm_table)

    if metrics["per_tool"]:
        tool_table = Table(
            title="Per-Tool Accuracy", show_header=True, header_style="bold magenta"
        )
        tool_table.add_column("Tool Name", style="bold")
        tool_table.add_column("Accuracy", justify="right")
        tool_table.add_column("Correct/Total", justify="right")

        for tool_name, tool_metrics in metrics["per_tool"].items():
            acc = tool_metrics["accuracy"]
            if acc >= 0.9:
                acc_str = f"[green]{acc:.4f}[/green]"
            elif acc >= 0.7:
                acc_str = f"[yellow]{acc:.4f}[/yellow]"
            else:
                acc_str = f"[red]{acc:.4f}[/red]"

            tool_table.add_row(
                tool_name,
                acc_str,
                f"{tool_metrics['correct']}/{tool_metrics['total']}",
            )

        console.print(tool_table)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TRM model on test data")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data JSON file",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to tokenizer model",
    )
    parser.add_argument(
        "--tools",
        type=str,
        default=None,
        help="Path to tools JSON file",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Path to system prompt text file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference per GPU (default: 8)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save metrics JSON (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save detailed results (conversation, prediction, target) in JSON files (max 100 samples per file)",
    )
    parser.add_argument(
        "--remove-think",
        action="store_true",
        help="Remove <think>...</think> tags from assistant responses",
    )

    args = parser.parse_args()

    # Setup distributed or single GPU
    rank, world_size, device = setup_distributed()

    if is_main_process():
        if is_distributed():
            logger.info(f"Distributed mode: {world_size} GPUs")
        logger.info(f"Using device: {device}")

    try:
        # Load model
        checkpoint_path = Path(args.checkpoint)
        model, config = load_checkpoint(checkpoint_path, device)

        # Load tokenizer
        tokenizer = TRMTokenizer(args.tokenizer_path)
        if is_main_process():
            logger.info(f"Loaded tokenizer with vocab size: {len(tokenizer)}")

        # Load tools and system prompt
        tools = load_tools(Path(args.tools) if args.tools else None)
        system_prompt = load_system_prompt(
            Path(args.system_prompt) if args.system_prompt else None
        )
        if is_main_process():
            logger.info(f"Loaded {len(tools)} tools")

        # Build tool mappings
        tool_names = [t["function"]["name"] for t in tools if "function" in t]
        tool_name_to_id = {name: idx for idx, name in enumerate(sorted(tool_names))}

        # Build unified param mapping (for new architecture)
        unified_fields, num_slots, param_name_to_idx, tool_param_mask = build_unified_param_mapping(
            tools, config.slot_fields
        )
        if is_main_process():
            logger.info(f"Unified fields: {len(unified_fields)} ({num_slots} slots + {len(unified_fields) - num_slots} params)")

        # Update config
        config.num_tools = len(tool_name_to_id)
        config.set_tool_param_fields([f for f in unified_fields[num_slots:]])

        # Load test data
        with open(args.test_data, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        if is_main_process():
            logger.info(f"Loaded {len(test_data)} conversations")

        # Parse test data
        if is_main_process() and args.remove_think:
            logger.info("Removing <think>...</think> tags from assistant responses")
        samples = parse_test_data(
            test_data, tools, system_prompt, remove_think=args.remove_think
        )
        if is_main_process():
            logger.info(f"Parsed {len(samples)} evaluation samples")

            # Decision distribution
            tool_call_count = sum(
                1 for s in samples if s["ground_truth"]["decision"] == "tool_call"
            )
            direct_answer_count = len(samples) - tool_call_count
            logger.info(f"  Decision distribution:")
            logger.info(f"    - tool_call: {tool_call_count} ({tool_call_count/len(samples)*100:.1f}%)")
            logger.info(f"    - direct_answer: {direct_answer_count} ({direct_answer_count/len(samples)*100:.1f}%)")

            # Per-tool distribution
            tool_counts: dict[str, int] = {}
            for s in samples:
                gt = s["ground_truth"]
                if gt["decision"] == "tool_call" and gt.get("tool_name"):
                    tool_name = gt["tool_name"]
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

            if tool_counts:
                logger.info(f"  Per-tool distribution:")
                for tool_name in sorted(tool_counts.keys()):
                    count = tool_counts[tool_name]
                    logger.info(f"    - {tool_name}: {count} ({count/tool_call_count*100:.1f}%)")

            # Per-tool parameter counts
            logger.info(f"  Per-tool parameters (from test data):")
            for tool_name, mask in tool_param_mask.items():
                active_params = sum(mask)
                logger.info(f"    - {tool_name}: {active_params} params")

        # Create dataset and dataloader
        test_dataset = TestDataset(samples, tokenizer, config.max_seq_len)

        # Use DistributedSampler for multi-GPU
        sampler = None
        if is_distributed():
            sampler = DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=collate_test_batch,
            num_workers=0,
        )

        # Run inference
        local_predictions = run_batched_inference(
            model=model,
            dataloader=test_dataloader,
            samples=samples,
            device=device,
            tool_name_to_id=tool_name_to_id,
            param_name_to_idx=param_name_to_idx,
            num_slots=num_slots,
            tool_param_mask=tool_param_mask,
        )

        # Gather predictions from all ranks
        predictions = gather_predictions(local_predictions, len(samples))

        # Only main process computes metrics and prints results
        if is_main_process():
            ground_truths = [s["ground_truth"] for s in samples]

            # Compute metrics
            metrics = compute_metrics(
                predictions=predictions,
                ground_truths=ground_truths,
                samples=samples,
                tool_names=tool_names,
                tool_param_mask=tool_param_mask,
                tokenizer=tokenizer,
                config=config,
            )

            # Print results
            print_results(metrics)

            # Save metrics
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                results = {
                    "metrics": metrics,
                    "predictions": [
                        {
                            "decision": p["decision"],
                            "decision_prob": p["decision_prob"],
                            "tool_name": p["tool_name"],
                            "tool_args": p["tool_args"],
                        }
                        for p in predictions
                    ],
                    "ground_truths": ground_truths,
                }

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                logger.info(f"Saved metrics to {output_path}")

            # Save detailed results (conversation, prediction, target)
            if args.output_dir:
                save_detailed_results(
                    samples=samples,
                    predictions=predictions,
                    output_dir=Path(args.output_dir),
                    tokenizer=tokenizer,
                    max_seq_len=config.max_seq_len,
                    max_samples_per_file=100,
                )

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
