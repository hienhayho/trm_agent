"""Training configuration for TRM."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0

    # Scheduler
    warmup_steps: int = 2000
    num_epochs: int = 100

    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # Memory optimization
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" or "bfloat16"

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999

    # ACT (Adaptive Computational Time)
    use_act: bool = True

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    log_sample_interval: int = 0  # Log sample predictions every N steps (0 = disabled)

    # Paths
    output_dir: str = "outputs"
