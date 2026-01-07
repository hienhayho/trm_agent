# Training Process

This document describes the TRM training process, including all components from data loading to optimization.

> **Note:** Span extraction (slots/params) is handled by **GLiNER2**, not TRM. TRM only handles decision classification and tool selection.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Training Pipeline                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  JSONL Dataset ──► Tokenizer ──► Dataset ──► DataLoader ──► Batch          │
│                                                                             │
│       Batch                                                                 │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │    Model    │────►│    Loss     │────►│  Backward   │                   │
│  │ (N_sup steps)│     │(Deep Super) │     │             │                   │
│  │ + TBPTL     │     │             │     │             │                   │
│  └─────────────┘     └─────────────┘     └──────┬──────┘                   │
│                                                  │                          │
│                                                  ▼                          │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │     EMA     │◄────│  Optimizer  │◄────│ Grad Clip   │                   │
│  │   Update    │     │    Step     │     │             │                   │
│  └─────────────┘     └─────────────┘     └─────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Training Configuration

```python
@dataclass
class TrainingConfig:
    # Optimizer (from paper)
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

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000

    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
```

## Training Components

### 1. Optimizer (AdamW)

```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)
```

**Paper settings:**
- Learning rate: 1e-4
- Weight decay: 0.1-1.0
- Betas: (0.9, 0.95)

### 2. Learning Rate Scheduler

Linear warmup then constant:

```
LR
 │
 │                 ┌────────────────────────
 │                /
 │               /
 │              /
 │             /
 │____________/
 └──────────────────────────────────────────► Step
 0        warmup_steps
```

```python
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps  # Linear warmup
    return 1.0  # Constant after warmup
```

### 3. Gradient Clipping

Prevents exploding gradients:

```python
nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm=1.0)
```

### 4. Exponential Moving Average (EMA)

Stabilizes training and improves generalization.

```python
# Update rule
shadow = decay * shadow + (1 - decay) * current_weights
# decay = 0.999 (default)
```

**Paper result:** EMA improves accuracy from 79.9% to 87.4%

```
                    EMA Update Flow

    Step 1         Step 2         Step 3
   weights        weights        weights
      │              │              │
      ▼              ▼              ▼
   shadow ────► shadow ────► shadow ────► ...
   (0.999)      (0.999)      (0.999)

   shadow_new = 0.999 * shadow_old + 0.001 * weights
```

**Usage:**

```python
# Training
for batch in dataloader:
    loss.backward()
    optimizer.step()
    ema.update()  # Update shadow weights

# Evaluation (use EMA weights)
ema.apply_shadow()
evaluate(model)
ema.restore()
```

### 5. URM Innovations

TRM incorporates key optimizations from the Universal Reasoning Model (URM) paper.

#### ConvSwiGLU (+5-8% accuracy)

Adds depthwise 1D convolution to the SwiGLU MLP for local feature mixing:

```python
class ConvSwiGLU(nn.Module):
    """SwiGLU with depthwise short convolution."""
    def __init__(self, hidden_size, intermediate_size, kernel_size=2):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        # Depthwise convolution for local feature mixing
        self.dwconv = nn.Conv1d(
            intermediate_size, intermediate_size,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=intermediate_size,
            bias=False,
        )

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        # Conv1d: (B, T, C) -> (B, C, T) -> conv -> (B, T, C)
        hidden = self.dwconv(hidden.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)
        hidden = F.silu(hidden)
        return self.down_proj(hidden)
```

**Configuration:**
```python
use_conv_swiglu: bool = True   # Enable ConvSwiGLU (default: True)
conv_kernel_size: int = 2      # Short conv kernel size
```

**Trade-off:** ConvSwiGLU uses more memory. Disable with `--no-conv-swiglu` if OOM occurs.

#### TBPTL - Truncated Backprop Through Loops (-12% memory, +2-3% accuracy)

Skips loss computation on the first K supervision steps, reducing memory while improving convergence:

```
                    N_sup = 16 Supervision Steps
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Step 0   Step 1   Step 2   Step 3   ...   Step 15         │
│    ↓        ↓        ↓        ↓              ↓              │
│  no_grad  no_grad  collect  collect  ...   collect         │
│  (skip)   (skip)   (loss)   (loss)          (loss)         │
│                                                             │
│  ←── tbptl_no_grad_steps = 2 ──→                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Configuration:**
```python
tbptl_no_grad_steps: int = 2   # Skip loss on first N steps (default: 2)
                               # Set to 0 to disable TBPTL
```

**Expected Impact (from URM paper):**

| Feature | Memory | Speed | Accuracy Gain |
|---------|--------|-------|---------------|
| ConvSwiGLU | +~20% | -2% | **+5-8%** |
| TBPTL (2 steps) | -12% | +5% | **+2-3%** |
| **Combined** | +8% | +3% | **+7-10%** |

## Training Loop

### Single Training Step

```python
def train_step(batch):
    model.train()

    # 1. Move to device
    batch = {k: v.to(device) for k, v in batch.items()}

    # 2. Forward pass (N_sup supervision steps)
    # Note: TBPTL skips first K steps (no_grad), reducing memory
    all_outputs = model.train_step(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        role_ids=batch["role_ids"],
    )

    # 3. Compute loss over collected steps (after TBPTL skip)
    losses = loss_fn(
        all_outputs,
        batch["decision_labels"],
        batch["tool_name_labels"],
    )

    # 4. Backward pass
    loss = losses["total_loss"] / gradient_accumulation_steps
    loss.backward()

    return losses
```

> **Note:** Slot/parameter extraction is handled by GLiNER2, not TRM. TRM only predicts `decision` (tool_call vs direct_answer) and `tool_name`.

### Epoch Loop

```python
def train_epoch():
    for step, batch in enumerate(dataloader):
        # Training step
        losses = train_step(batch)

        # Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # EMA update
            ema.update()

            global_step += 1

            # Logging
            if global_step % log_interval == 0:
                log(losses)

            # Evaluation
            if global_step % eval_interval == 0:
                evaluate()

            # Checkpoint
            if global_step % save_interval == 0:
                save_checkpoint()
```

### Full Training

```python
def train():
    for epoch in range(num_epochs):
        # Train one epoch
        avg_losses = train_epoch()

        # Evaluate
        eval_metrics = evaluate()

        # Save checkpoint
        save_checkpoint(f"epoch_{epoch}")

    # Save final model
    save_model()
```

## Evaluation

The trainer provides comprehensive evaluation metrics displayed using rich tables for better visualization.

### Evaluation Metrics

```python
@torch.no_grad()
def evaluate():
    model.eval()
    ema.apply_shadow()  # Use EMA weights

    # Collect metrics
    for batch in eval_dataloader:
        outputs = model.inference(...)

        # Decision confusion matrix (tool_call vs direct_answer)
        # Tool name accuracy (for tool_call samples only)

    ema.restore()
    return metrics
```

> **Note:** Slot/parameter extraction metrics are evaluated separately using GLiNER2.

### Confusion Matrix Display (Rich Table)

```
          Decision Confusion Matrix
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Actual \ Predicted   ┃ tool_call  ┃ direct_answer  ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ tool_call            │ 245 (TP)   │  15 (FN)       │
│ direct_answer        │  12 (FP)   │ 128 (TN)       │
└──────────────────────┴────────────┴────────────────┘
```

### Evaluation Results Table

```
       Epoch 5 Evaluation Results
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Metric                 ┃          Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ Decision Accuracy      │         0.9325 │
│ Decision Precision     │         0.9531 │
│ Decision Recall        │         0.9423 │
│ Decision F1            │         0.9477 │
│                        │                │
│ Tool Accuracy          │ 0.8980 (245 s) │
└────────────────────────┴────────────────┘
```

### Metrics Computed

| Metric | Formula | Description |
|--------|---------|-------------|
| Decision Accuracy | (TP + TN) / Total | Overall decision correctness |
| Decision Precision | TP / (TP + FP) | When predicting tool_call, how often correct |
| Decision Recall | TP / (TP + FN) | Of all tool_calls, how many detected |
| Decision F1 | 2 * P * R / (P + R) | Harmonic mean of precision and recall |
| Tool Accuracy | Correct / Total | Tool name classification accuracy |
| Intent Accuracy | Correct / Total | Intent prediction accuracy (if enabled) |

## Checkpointing

### Save Checkpoint

```python
state = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "ema_state_dict": ema.state_dict(),
    "global_step": global_step,
    "epoch": epoch,
    "config": config.to_dict(),
}
torch.save(state, "checkpoints/step_1000.pt")
```

### Load Checkpoint

```python
state = torch.load("checkpoints/step_1000.pt")
model.load_state_dict(state["model_state_dict"])
optimizer.load_state_dict(state["optimizer_state_dict"])
scheduler.load_state_dict(state["scheduler_state_dict"])
ema.load_state_dict(state["ema_state_dict"])
global_step = state["global_step"]
epoch = state["epoch"]
```

### Save Final Model

```python
# Save model weights
torch.save(model.state_dict(), "outputs/model.pt")

# Save EMA weights (often better for inference)
ema.apply_shadow()
torch.save(model.state_dict(), "outputs/model_ema.pt")
ema.restore()

# Save config
config.to_yaml("outputs/config.yaml")
```

## CLI Usage

```bash
uv run python tools/train.py \
    --config configs/default.yaml \
    --train-data data/train.jsonl \
    --val-data data/val.jsonl \
    --tokenizer-path tokenizer/trm.model \
    --output-dir outputs/ \
    --checkpoint-dir checkpoints/ \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4
```

### With Intent Prediction

Enable intent prediction by providing an intent mapping file:

```bash
uv run python tools/train.py \
    --config configs/default.yaml \
    --train-data data/train.jsonl \
    --intent-map data/intend.json \
    --output-dir outputs/
```

**Intent mapping file format (JSON):**
```json
{
  "greeting": "User greets the assistant",
  "check_balance": "User wants to check account balance",
  "make_payment": "User wants to make a payment",
  "get_support": "User needs technical support"
}
```

Intent IDs are assigned alphabetically, so:
- `check_balance` → 0
- `get_support` → 1
- `greeting` → 2
- `make_payment` → 3

The model will predict the next expected intent alongside decision and tool predictions. Intent loss uses **Focal Cross-Entropy** to handle class imbalance.

### With Validation Split

If you don't have a separate validation file, use `--val-split` to split training data:

```bash
uv run python tools/train.py \
    --config configs/default.yaml \
    --train-data data/train.jsonl \
    --val-split 0.1 \
    --output-dir outputs/
```

This splits 10% of training data for validation.

### URM Flags

```bash
# Disable ConvSwiGLU (use standard SwiGLU) - saves memory
uv run python tools/train.py \
    --config configs/default.yaml \
    --train-data data/train.jsonl \
    --no-conv-swiglu \
    --output-dir outputs/
```

### Distributed Training (DDP)

```bash
# Multi-GPU training with torchrun
torchrun --nproc_per_node=4 tools/train.py \
    --config configs/default.yaml \
    --train-data data/train.jsonl \
    --output-dir outputs/
```

**DDP Optimizations:**
- Gradient sync is skipped during accumulation steps (`model.no_sync()`)
- Tokenizer training uses file-based sync to avoid NCCL timeout
- Only rank 0 performs tokenizer training (others wait)

## Hyperparameter Recommendations

| Parameter | Small Dataset | Large Dataset |
|-----------|--------------|---------------|
| `learning_rate` | 1e-4 | 1e-4 to 3e-4 |
| `batch_size` | 16-32 | 64-768 |
| `warmup_steps` | 500-1000 | 2000-5000 |
| `num_epochs` | 50-100 | 10-30 |
| `weight_decay` | 0.1 | 0.1-1.0 |
| `ema_decay` | 0.999 | 0.999-0.9999 |

---

## Complete End-to-End Example

This example shows one sample flowing through all components.

### Step 1: Raw Sample (JSONL)

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_customer_info",
        "description": "Get customer information",
        "parameters": {"type": "object", "properties": {"phone": {"type": "string"}}}
      }
    }
  ],
  "history": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Check info for 0901234567"}
  ],
  "decision": "tool_call",
  "tool": {"name": "get_customer_info", "arguments": {"phone": "0901234567"}},
  "slots": {"phone": "0901234567", "name": "", "address": "", "device_number": "", "intent_of_user": "check_info", "contract_id": ""},
  "content": ""
}
```

> **Note:** The `slots` field is present in the dataset format but processed by **GLiNER2**, not TRM. TRM only uses `decision` (tool_call vs direct_answer) and `tool` (tool name).

### Step 2: Tokenization

```python
from trm_agent.data import TRMTokenizer

tokenizer = TRMTokenizer("tokenizer/trm.model")

history = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Check info for 0901234567"}
]

encoded = tokenizer.encode_conversation(history, max_length=2048)
```

**Output:**

```python
encoded = {
    "input_ids": [2, 102, 567, 234, 891, ..., 103, 445, 678, 901, 3],
    #            BOS <sys> tokens...       <user> tokens...      EOS
    "role_ids":  [0,   1,   1,   1,   1, ...,   0,   0,   0,   0, 0],
    #            BOS  system tokens        user tokens           EOS
    "attention_mask": [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1],
}

# Shapes
# input_ids: [45] (sequence length)
# role_ids: [45]
# attention_mask: [45]
```

### Step 3: Dataset __getitem__

```python
from trm_agent.data import TRMToolCallingDataset

dataset = TRMToolCallingDataset("data/train.jsonl", tokenizer)
sample = dataset[0]
```

**Output:**

```python
sample = {
    "input_ids": tensor([2, 102, 567, ..., 3]),        # [45]
    "attention_mask": tensor([1, 1, 1, ..., 1]),       # [45]
    "role_ids": tensor([0, 1, 1, ..., 0]),             # [45]
    "decision_label": tensor(1.0),                     # scalar (tool_call=1)
    "tool_name_label": tensor(0),                      # scalar (get_customer_info=0)
}
```

> **Note:** Slot extraction is handled by GLiNER2, not the TRM dataset.

### Step 4: Batch Collation

```python
from trm_agent.data.collator import TRMCollator

collator = TRMCollator(pad_token_id=0)
batch = collator([dataset[0], dataset[1], dataset[2], dataset[3]])
```

**Output (batch_size=4, max_len=64):**

```python
batch = {
    "input_ids": tensor([
        [2, 102, 567, ..., 3, 0, 0, 0],  # Sample 0 (padded)
        [2, 102, 891, ..., 3, 0, 0, 0],  # Sample 1 (padded)
        [2, 102, 234, ..., 3, 0, 0, 0],  # Sample 2 (padded)
        [2, 102, 456, ..., 3, 0, 0, 0],  # Sample 3 (padded)
    ]),  # Shape: [4, 64]

    "attention_mask": tensor([
        [1, 1, 1, ..., 1, 0, 0, 0],
        [1, 1, 1, ..., 1, 0, 0, 0],
        [1, 1, 1, ..., 1, 0, 0, 0],
        [1, 1, 1, ..., 1, 0, 0, 0],
    ]),  # Shape: [4, 64]

    "role_ids": tensor([
        [0, 1, 1, ..., 0, 0, 0, 0],
        [0, 1, 1, ..., 0, 0, 0, 0],
        [0, 1, 1, ..., 0, 0, 0, 0],
        [0, 1, 1, ..., 0, 0, 0, 0],
    ]),  # Shape: [4, 64]

    "decision_labels": tensor([1., 0., 1., 0.]),  # [4]
    "tool_name_labels": tensor([0, -1, 2, -1]),   # [4] (-1 = direct_answer)
}
```

### Step 5: Model Forward Pass

```python
from trm_agent.models import TRMConfig, TRMForToolCalling

config = TRMConfig(
    hidden_size=512,
    num_layers=2,
    num_tools=3,
    n_latent_recursion=6,
    T_deep_recursion=3,
    N_supervision=16,
    # URM innovations (enabled by default)
    use_conv_swiglu=True,
    conv_kernel_size=2,
    tbptl_no_grad_steps=2,
)
model = TRMForToolCalling(config)

# Move to device
device = torch.device("cuda")
model = model.to(device)
batch = {k: v.to(device) for k, v in batch.items()}

# Training forward (returns N_sup - tbptl_no_grad_steps outputs)
all_outputs = model.train_step(
    input_ids=batch["input_ids"],           # [4, 64]
    attention_mask=batch["attention_mask"], # [4, 64]
    role_ids=batch["role_ids"],             # [4, 64]
)
```

**Output (list of 14 TRMOutput, with TBPTL skipping first 2):**

```python
len(all_outputs)  # 14 (N_supervision - tbptl_no_grad_steps)

# Each TRMOutput contains:
output = all_outputs[0]
output.decision_logits      # [4, 1]
output.tool_logits          # [4, 3] (num_tools=3)
output.q_logits             # [4, 1]
output.y                    # [4, 64, 512]
output.z                    # [4, 64, 512]
```

> **Note:** Span extraction outputs (slot/arg logits) are removed. Use GLiNER2 for entity extraction.

### Step 6: Loss Computation

```python
from trm_agent.training.losses import DeepSupervisionLoss

loss_fn = DeepSupervisionLoss(config)

losses = loss_fn(
    all_outputs,                        # List of 14 TRMOutput (after TBPTL)
    batch["decision_labels"],           # [4]
    batch["tool_name_labels"],          # [4]
)
```

**Output:**

```python
losses = {
    "decision_loss": tensor(0.4532),  # Focal Loss
    "tool_loss": tensor(1.0234),      # Cross Entropy (masked)
    "q_loss": tensor(0.6789),         # BCE
    "total_loss": tensor(1.8234),     # Weighted sum
}

# How total_loss is computed:
# total_loss = 1.0 * decision_loss + 1.0 * tool_loss + 0.5 * q_loss
# = 1.0 * 0.4532 + 1.0 * 1.0234 + 0.5 * 0.6789
# = 0.4532 + 1.0234 + 0.3395
# = 1.8161 (approximately)
```

> **Note:** Slot loss is removed since span extraction is handled by GLiNER2.

### Step 7: Backward and Optimization

```python
# Backward pass
losses["total_loss"].backward()

# Gradient clipping
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Optimizer step
optimizer.step()
scheduler.step()
optimizer.zero_grad()

# EMA update
ema.update()
```

### Step 8: Inference

```python
# For inference, use model.inference()
with torch.no_grad():
    model.eval()
    ema.apply_shadow()  # Use EMA weights

    output = model.inference(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        role_ids=batch["role_ids"],
    )

    # Get predictions
    decision_probs = torch.sigmoid(output.decision_logits)
    decisions = (decision_probs > 0.5).long()
    # tensor([[1], [0], [1], [0]])  # tool_call, direct, tool_call, direct

    tool_ids = output.tool_logits.argmax(dim=-1)
    # tensor([0, 1, 2, 0])  # Predicted tool IDs

    ema.restore()  # Restore original weights

# For slot/parameter extraction, use GLiNER2 separately
# from gliner import GLiNER
# gliner = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
# entities = gliner.predict_entities(text, labels=["phone", "address", ...])
```

### Complete Training Script Example

```python
import torch
from torch.optim import AdamW

from trm_agent.data import TRMTokenizer, TRMToolCallingDataset
from trm_agent.data.collator import create_dataloader
from trm_agent.models import TRMConfig, TRMForToolCalling
from trm_agent.models.ema import EMA
from trm_agent.training.losses import DeepSupervisionLoss

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load tokenizer and data
tokenizer = TRMTokenizer("tokenizer/trm.model")
dataset = TRMToolCallingDataset("data/train.jsonl", tokenizer)
dataloader = create_dataloader(dataset, batch_size=32, shuffle=True)

# 3. Create model (with URM innovations)
config = TRMConfig(
    num_tools=len(dataset.tool_name_to_id),
    use_conv_swiglu=True,       # ConvSwiGLU for local feature mixing
    tbptl_no_grad_steps=2,      # Skip loss on first 2 supervision steps
)
model = TRMForToolCalling(config).to(device)

# 4. Setup training
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
loss_fn = DeepSupervisionLoss(config)
ema = EMA(model, decay=0.999)

# 5. Training loop
for epoch in range(100):
    model.train()
    total_loss = 0

    for batch in dataloader:
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward (TBPTL: first 2 steps run without gradients)
        all_outputs = model.train_step(
            batch["input_ids"],
            batch["attention_mask"],
            batch["role_ids"],
        )

        # Loss (computed on collected outputs only)
        losses = loss_fn(
            all_outputs,
            batch["decision_labels"],
            batch["tool_name_labels"],
        )

        # Backward
        losses["total_loss"].backward()

        # Optimize
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        ema.update()

        total_loss += losses["total_loss"].item()

    print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")

# 6. Save model
ema.apply_shadow()
torch.save(model.state_dict(), "model_ema.pt")
```

## Summary

| Step | Input | Output | Shape Change |
|------|-------|--------|--------------|
| 1. Raw JSON | JSONL file | dict | - |
| 2. Tokenize | history | input_ids, role_ids | → [L] |
| 3. Dataset | sample dict | tensors | → [L], scalar |
| 4. Collate | list of samples | batch | → [B, L] |
| 5. Forward | batch tensors | N_sup outputs | → [B, L, D] |
| 6. Loss | outputs + labels | loss dict | → scalar |
| 7. Backward | loss | gradients | - |
| 8. Optimize | gradients | updated weights | - |
