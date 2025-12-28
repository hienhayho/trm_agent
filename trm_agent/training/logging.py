"""Logging utilities for TRM training.

Provides rich table formatting for evaluation metrics and confusion matrices.
"""

from rich.console import Console
from rich.table import Table

from trm_agent.utils import is_main_process

console = Console()


def log_confusion_matrix(tp: int, fp: int, fn: int, tn: int):
    """Log confusion matrix for decision classification.

    Args:
        tp: True positives (predicted tool_call, actual tool_call)
        fp: False positives (predicted tool_call, actual direct_answer)
        fn: False negatives (predicted direct_answer, actual tool_call)
        tn: True negatives (predicted direct_answer, actual direct_answer)
    """
    if not is_main_process():
        return

    total = tp + fp + fn + tn
    if total == 0:
        return

    table = Table(title="Decision Confusion Matrix", show_header=True)
    table.add_column("Actual \\ Predicted", style="bold")
    table.add_column("tool_call", justify="center", style="cyan")
    table.add_column("direct_answer", justify="center", style="magenta")

    table.add_row("tool_call", f"[green]{tp}[/green] (TP)", f"[red]{fn}[/red] (FN)")
    table.add_row("direct_answer", f"[red]{fp}[/red] (FP)", f"[green]{tn}[/green] (TN)")

    console.print(table)


def _color_accuracy(value: float) -> str:
    """Color code accuracy value based on threshold."""
    if value >= 0.9:
        return f"[green]{value:.4f}[/]"
    elif value >= 0.7:
        return f"[yellow]{value:.4f}[/]"
    else:
        return f"[red]{value:.4f}[/]"


def log_eval_metrics(
    epoch: int,
    metrics: dict[str, float],
    unified_fields: list[str],
    num_slots: int,
    tool_names: list[str],
):
    """Log evaluation metrics using rich tables.

    Args:
        epoch: Current epoch number
        metrics: Dictionary of evaluation metrics
        unified_fields: List of unified field names
        num_slots: Number of slot fields
        tool_names: List of tool names
    """
    if not is_main_process():
        return

    num_tool_params = len(unified_fields) - num_slots

    # Main metrics table
    main_table = Table(
        title=f"Epoch {epoch} Evaluation Results",
        show_header=True,
        header_style="bold blue",
    )
    main_table.add_column("Metric", style="bold")
    main_table.add_column("Value", justify="right")

    main_table.add_row(
        "Decision Accuracy",
        f"{metrics.get('decision_accuracy', 0):.4f}"
    )
    main_table.add_row(
        "Decision Precision",
        f"{metrics.get('decision_precision', 0):.4f}"
    )
    main_table.add_row(
        "Decision Recall",
        f"{metrics.get('decision_recall', 0):.4f}"
    )
    main_table.add_row(
        "Decision F1",
        f"[bold cyan]{metrics.get('decision_f1', 0):.4f}[/bold cyan]"
    )
    main_table.add_row("", "")

    main_table.add_row(
        "Tool Accuracy",
        f"{metrics.get('tool_accuracy', 0):.4f} ({metrics.get('tool_samples', 0)} samples)"
    )
    main_table.add_row("", "")

    main_table.add_row(
        "Slot Presence Acc",
        f"{metrics.get('slot_presence_accuracy', 0):.4f}"
    )
    main_table.add_row(
        "Slot Span EM",
        f"{metrics.get('slot_span_exact_match', 0):.4f} ({metrics.get('slot_span_samples', 0)} spans)"
    )
    main_table.add_row("", "")

    main_table.add_row(
        "Param Presence Acc",
        f"{metrics.get('param_presence_accuracy', 0):.4f}"
    )
    main_table.add_row(
        "Param Span EM",
        f"{metrics.get('param_span_exact_match', 0):.4f} ({metrics.get('param_span_samples', 0)} spans)"
    )

    console.print(main_table)

    # Per-slot accuracy table
    if num_slots > 0:
        slot_table = Table(
            title="Per-Slot Accuracy",
            show_header=True,
            header_style="bold yellow"
        )
        slot_table.add_column("Slot Field", style="bold")
        slot_table.add_column("Presence Acc", justify="right")
        slot_table.add_column("Span Acc", justify="right")

        for i in range(num_slots):
            field = unified_fields[i]
            presence_acc = metrics.get(f"slot_{field}_presence_acc", 0)
            span_acc = metrics.get(f"slot_{field}_span_acc", 0)
            slot_table.add_row(
                field,
                f"{presence_acc:.4f}",
                _color_accuracy(span_acc)
            )

        console.print(slot_table)

    # Per-param accuracy table
    if num_tool_params > 0:
        param_table = Table(
            title="Per-Parameter Accuracy",
            show_header=True,
            header_style="bold green"
        )
        param_table.add_column("Parameter", style="bold")
        param_table.add_column("Presence Acc", justify="right")
        param_table.add_column("Span Acc", justify="right")

        for i in range(num_slots, len(unified_fields)):
            field = unified_fields[i]
            presence_acc = metrics.get(f"param_{field}_presence_acc", 0)
            span_acc = metrics.get(f"param_{field}_span_acc", 0)
            param_table.add_row(
                field,
                f"{presence_acc:.4f}",
                _color_accuracy(span_acc)
            )

        console.print(param_table)

    # Per-tool accuracy table
    if tool_names:
        tool_table = Table(
            title="Per-Tool Accuracy",
            show_header=True,
            header_style="bold magenta"
        )
        tool_table.add_column("Tool Name", style="bold")
        tool_table.add_column("Accuracy", justify="right")
        tool_table.add_column("Samples", justify="right")

        for tool_name in tool_names:
            acc = metrics.get(f"tool_{tool_name}_acc", 0)
            samples = int(metrics.get(f"tool_{tool_name}_samples", 0))
            tool_table.add_row(
                tool_name,
                _color_accuracy(acc),
                str(samples)
            )

        console.print(tool_table)

    # Log confusion matrix if available
    if "_decision_tp" in metrics:
        log_confusion_matrix(
            metrics["_decision_tp"],
            metrics["_decision_fp"],
            metrics["_decision_fn"],
            metrics["_decision_tn"],
        )


def log_training_start(
    num_epochs: int,
    device: str,
    train_samples: int,
    eval_samples: int,
    unified_fields: list[str],
    num_slots: int,
):
    """Log training start information.

    Args:
        num_epochs: Number of training epochs
        device: Training device
        train_samples: Number of training samples
        eval_samples: Number of evaluation samples
        unified_fields: List of unified field names
        num_slots: Number of slot fields
    """
    if not is_main_process():
        return

    num_tool_params = len(unified_fields) - num_slots

    table = Table(title="Training Configuration", show_header=True)
    table.add_column("Setting", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Epochs", str(num_epochs))
    table.add_row("Device", device)
    table.add_row("Training Samples", str(train_samples))
    table.add_row("Evaluation Samples", str(eval_samples))
    table.add_row("Unified Fields", f"{len(unified_fields)} ({num_slots} slots + {num_tool_params} params)")

    console.print(table)
