"""Evaluation metrics for TRM training."""

from dataclasses import dataclass, field
from typing import Optional

import torch

from trm_agent.models.trm import TRMOutput
from trm_agent.utils import gather_metrics


@dataclass
class EvalAccumulators:
    """Accumulators for evaluation metrics."""

    # Decision confusion matrix
    decision_tp: int = 0
    decision_fp: int = 0
    decision_fn: int = 0
    decision_tn: int = 0

    # Tool accuracy
    tool_correct: int = 0
    tool_total: int = 0
    per_tool_correct: list[int] = field(default_factory=list)
    per_tool_total: list[int] = field(default_factory=list)

    # Unified field accuracy
    per_field_presence_correct: list[int] = field(default_factory=list)
    per_field_presence_total: list[int] = field(default_factory=list)
    per_field_span_correct: list[int] = field(default_factory=list)
    per_field_span_total: list[int] = field(default_factory=list)

    total_samples: int = 0

    def init_lists(self, num_tools: int, num_unified: int):
        """Initialize per-item lists."""
        self.per_tool_correct = [0] * num_tools
        self.per_tool_total = [0] * num_tools
        self.per_field_presence_correct = [0] * num_unified
        self.per_field_presence_total = [0] * num_unified
        self.per_field_span_correct = [0] * num_unified
        self.per_field_span_total = [0] * num_unified


def update_decision_metrics(
    accum: EvalAccumulators,
    decision_pred: torch.Tensor,
    decision_true: torch.Tensor,
):
    """Update decision confusion matrix.

    Args:
        accum: Accumulator to update
        decision_pred: Predicted decisions [batch]
        decision_true: True decisions [batch]
    """
    for pred, true in zip(decision_pred, decision_true):
        if pred == 1 and true == 1:
            accum.decision_tp += 1
        elif pred == 1 and true == 0:
            accum.decision_fp += 1
        elif pred == 0 and true == 1:
            accum.decision_fn += 1
        else:
            accum.decision_tn += 1


def update_tool_metrics(
    accum: EvalAccumulators,
    outputs: TRMOutput,
    tool_name_labels: torch.Tensor,
    num_tools: int,
):
    """Update tool prediction metrics.

    Args:
        accum: Accumulator to update
        outputs: Model outputs
        tool_name_labels: True tool indices [batch]
        num_tools: Number of tools
    """
    tool_mask = tool_name_labels >= 0
    if not tool_mask.any():
        return

    tool_pred = outputs.tool_logits.argmax(dim=-1)
    tool_true = tool_name_labels

    accum.tool_correct += (tool_pred[tool_mask] == tool_true[tool_mask]).sum().item()
    accum.tool_total += tool_mask.sum().item()

    for i in range(num_tools):
        tool_i_mask = tool_true == i
        if tool_i_mask.any():
            accum.per_tool_correct[i] += (tool_pred[tool_i_mask] == i).sum().item()
            accum.per_tool_total[i] += tool_i_mask.sum().item()


def update_unified_field_metrics(
    accum: EvalAccumulators,
    outputs: TRMOutput,
    batch: dict[str, torch.Tensor],
    num_slots: int,
    tool_param_mask: Optional[torch.Tensor] = None,
):
    """Update unified field (slots + params) metrics.

    Args:
        accum: Accumulator to update
        outputs: Model outputs
        batch: Batch data with labels
        num_slots: Number of slot fields
        tool_param_mask: [num_tools, num_tool_params] mask
    """
    batch_size = batch["decision_labels"].size(0)
    decision_true = batch["decision_labels"].long()
    num_unified = len(accum.per_field_presence_correct)

    presence_pred = (torch.sigmoid(outputs.param_presence_logits) > 0.5).float()
    presence_true = batch["unified_presence_labels"]
    start_labels = batch["unified_start_labels"]
    end_labels = batch["unified_end_labels"]
    start_pred = outputs.param_start_logits.argmax(dim=1)
    end_pred = outputs.param_end_logits.argmax(dim=1)

    for i in range(batch_size):
        is_tool_call = decision_true[i] == 1
        tool_idx = batch["tool_name_labels"][i].item()

        for f in range(num_unified):
            is_slot = f < num_slots
            is_valid_param = False

            if not is_slot and is_tool_call and tool_idx >= 0:
                if tool_param_mask is not None:
                    param_idx = f - num_slots
                    is_valid_param = tool_param_mask[tool_idx, param_idx].item() > 0
                else:
                    is_valid_param = True

            if is_slot or is_valid_param:
                # Presence accuracy
                accum.per_field_presence_correct[f] += int(
                    presence_pred[i, f] == presence_true[i, f]
                )
                accum.per_field_presence_total[f] += 1

                # Span accuracy
                if start_labels[i, f] >= 0:
                    start_match = start_pred[i, f] == start_labels[i, f]
                    end_match = end_pred[i, f] == end_labels[i, f]
                    accum.per_field_span_correct[f] += int(start_match and end_match)
                    accum.per_field_span_total[f] += 1


def compute_final_metrics(
    accum: EvalAccumulators,
    unified_fields: list[str],
    num_slots: int,
    tool_names: list[str],
    device: torch.device,
) -> dict[str, float]:
    """Compute final metrics from accumulators.

    Args:
        accum: Accumulated metrics
        unified_fields: List of unified field names
        num_slots: Number of slot fields
        tool_names: List of tool names
        device: Device for gather operation

    Returns:
        Dictionary of computed metrics
    """
    num_unified = len(unified_fields)

    # Build raw metrics for gathering
    raw_metrics = {
        "total_samples": accum.total_samples,
        "decision_tp": accum.decision_tp,
        "decision_fp": accum.decision_fp,
        "decision_fn": accum.decision_fn,
        "decision_tn": accum.decision_tn,
        "tool_correct": accum.tool_correct,
        "tool_total": accum.tool_total,
    }

    for i in range(num_unified):
        raw_metrics[f"field_presence_correct_{i}"] = accum.per_field_presence_correct[i]
        raw_metrics[f"field_presence_total_{i}"] = accum.per_field_presence_total[i]
        raw_metrics[f"field_span_correct_{i}"] = accum.per_field_span_correct[i]
        raw_metrics[f"field_span_total_{i}"] = accum.per_field_span_total[i]

    for i in range(len(tool_names)):
        raw_metrics[f"per_tool_correct_{i}"] = accum.per_tool_correct[i]
        raw_metrics[f"per_tool_total_{i}"] = accum.per_tool_total[i]

    # Gather across GPUs
    agg = gather_metrics(raw_metrics, device)

    # Compute final metrics
    metrics = {"eval_samples": int(agg["total_samples"])}

    # Decision metrics
    decision_total = sum([
        agg["decision_tp"], agg["decision_fp"],
        agg["decision_fn"], agg["decision_tn"]
    ])
    metrics["decision_accuracy"] = (
        (agg["decision_tp"] + agg["decision_tn"]) / decision_total
        if decision_total > 0 else 0.0
    )

    precision = (
        agg["decision_tp"] / (agg["decision_tp"] + agg["decision_fp"])
        if (agg["decision_tp"] + agg["decision_fp"]) > 0 else 0.0
    )
    recall = (
        agg["decision_tp"] / (agg["decision_tp"] + agg["decision_fn"])
        if (agg["decision_tp"] + agg["decision_fn"]) > 0 else 0.0
    )
    metrics["decision_precision"] = precision
    metrics["decision_recall"] = recall
    metrics["decision_f1"] = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    # Tool metrics
    metrics["tool_accuracy"] = (
        agg["tool_correct"] / agg["tool_total"]
        if agg["tool_total"] > 0 else 0.0
    )
    metrics["tool_samples"] = int(agg["tool_total"])

    # Aggregate slot and param metrics
    slot_presence_correct = sum(
        agg[f"field_presence_correct_{i}"] for i in range(num_slots)
    )
    slot_presence_total = sum(
        agg[f"field_presence_total_{i}"] for i in range(num_slots)
    )
    slot_span_correct = sum(
        agg[f"field_span_correct_{i}"] for i in range(num_slots)
    )
    slot_span_total = sum(
        agg[f"field_span_total_{i}"] for i in range(num_slots)
    )

    param_presence_correct = sum(
        agg[f"field_presence_correct_{i}"] for i in range(num_slots, num_unified)
    )
    param_presence_total = sum(
        agg[f"field_presence_total_{i}"] for i in range(num_slots, num_unified)
    )
    param_span_correct = sum(
        agg[f"field_span_correct_{i}"] for i in range(num_slots, num_unified)
    )
    param_span_total = sum(
        agg[f"field_span_total_{i}"] for i in range(num_slots, num_unified)
    )

    metrics["slot_presence_accuracy"] = (
        slot_presence_correct / slot_presence_total
        if slot_presence_total > 0 else 0.0
    )
    metrics["slot_span_exact_match"] = (
        slot_span_correct / slot_span_total
        if slot_span_total > 0 else 0.0
    )
    metrics["slot_span_samples"] = int(slot_span_total)

    metrics["param_presence_accuracy"] = (
        param_presence_correct / param_presence_total
        if param_presence_total > 0 else 0.0
    )
    metrics["param_span_exact_match"] = (
        param_span_correct / param_span_total
        if param_span_total > 0 else 0.0
    )
    metrics["param_span_samples"] = int(param_span_total)

    # Per-field accuracy
    for i, field_name in enumerate(unified_fields):
        presence_total = int(agg[f"field_presence_total_{i}"])
        span_total = int(agg[f"field_span_total_{i}"])
        presence_acc = (
            agg[f"field_presence_correct_{i}"] / presence_total
            if presence_total > 0 else 0.0
        )
        span_acc = (
            agg[f"field_span_correct_{i}"] / span_total
            if span_total > 0 else 0.0
        )
        prefix = "slot" if i < num_slots else "param"
        metrics[f"{prefix}_{field_name}_presence_acc"] = presence_acc
        metrics[f"{prefix}_{field_name}_span_acc"] = span_acc

    # Per-tool accuracy
    for i, tool_name in enumerate(tool_names):
        total = int(agg[f"per_tool_total_{i}"])
        acc = agg[f"per_tool_correct_{i}"] / total if total > 0 else 0.0
        metrics[f"tool_{tool_name}_acc"] = acc
        metrics[f"tool_{tool_name}_samples"] = total

    # Store raw confusion matrix for logging
    metrics["_decision_tp"] = int(agg["decision_tp"])
    metrics["_decision_fp"] = int(agg["decision_fp"])
    metrics["_decision_fn"] = int(agg["decision_fn"])
    metrics["_decision_tn"] = int(agg["decision_tn"])

    return metrics
