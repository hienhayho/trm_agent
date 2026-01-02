"""Training utilities for TRM.

Note: Span extraction (slots/params) is handled by GLiNER2, not TRM.
TRM only handles decision classification and tool selection.
"""

from .config import TrainingConfig
from .losses import (
    DeepSupervisionLoss,
    FocalLoss,
    TRMLoss,
)
from .logging import log_confusion_matrix, log_eval_metrics, log_training_start
from .metrics import (
    EvalAccumulators,
    compute_final_metrics,
    update_decision_metrics,
    update_tool_metrics,
)
from .scheduler import (
    get_constant_warmup_scheduler,
    get_cosine_warmup_scheduler,
    get_linear_warmup_scheduler,
)
from .trainer import TRMTrainer

__all__ = [
    # Config
    "TrainingConfig",
    # Losses
    "FocalLoss",
    "TRMLoss",
    "DeepSupervisionLoss",
    # Logging
    "log_confusion_matrix",
    "log_eval_metrics",
    "log_training_start",
    # Metrics
    "EvalAccumulators",
    "compute_final_metrics",
    "update_decision_metrics",
    "update_tool_metrics",
    # Schedulers
    "get_cosine_warmup_scheduler",
    "get_linear_warmup_scheduler",
    "get_constant_warmup_scheduler",
    # Trainer
    "TRMTrainer",
]
