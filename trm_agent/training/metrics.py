"""Evaluation metrics for TRM training.

Note: Span extraction (slots/params) metrics are handled by GLiNER2 evaluation.
TRM only evaluates decision classification and tool selection.
"""

from dataclasses import dataclass, field

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

    total_samples: int = 0

    def init_lists(self, num_tools: int):
        """Initialize per-item lists."""
        self.per_tool_correct = [0] * num_tools
        self.per_tool_total = [0] * num_tools


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


def compute_final_metrics(
    accum: EvalAccumulators,
    tool_names: list[str],
    device: torch.device,
) -> dict[str, float]:
    """Compute final metrics from accumulators.

    Args:
        accum: Accumulated metrics
        tool_names: List of tool names
        device: Device for gather operation

    Returns:
        Dictionary of computed metrics
    """
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
