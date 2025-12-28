"""Training utilities for TRM."""

from .config import TrainingConfig
from .losses import (
    DeepSupervisionLoss,
    FocalLoss,
    TRMLoss,
    UnifiedPresenceLoss,
    UnifiedSpanLoss,
)
from .logging import log_confusion_matrix, log_eval_metrics, log_training_start
from .metrics import (
    EvalAccumulators,
    compute_final_metrics,
    update_decision_metrics,
    update_tool_metrics,
    update_unified_field_metrics,
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
    "UnifiedSpanLoss",
    "UnifiedPresenceLoss",
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
    "update_unified_field_metrics",
    # Schedulers
    "get_cosine_warmup_scheduler",
    "get_linear_warmup_scheduler",
    "get_constant_warmup_scheduler",
    # Trainer
    "TRMTrainer",
]
