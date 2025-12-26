# Training Process

This document describes the TRM training process, including all components from data loading to optimization.

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

## Training Loop

### Single Training Step

```python
def train_step(batch):
    model.train()

    # 1. Move to device
    batch = {k: v.to(device) for k, v in batch.items()}

    # 2. Forward pass (N_sup supervision steps)
    all_outputs = model.train_step(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        role_ids=batch["role_ids"],
    )

    # 3. Compute loss over all steps
    losses = loss_fn(
        all_outputs,
        batch["decision_labels"],
        batch["tool_name_labels"],
        batch["slot_presence_labels"],
    )

    # 4. Backward pass
    loss = losses["total_loss"] / gradient_accumulation_steps
    loss.backward()

    return losses
```

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

        # Decision confusion matrix
        # Tool name accuracy (for tool_call samples)
        # Slot presence accuracy (per-slot and overall)

    ema.restore()
    return metrics
```

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
│                        │                │
│ Slot Accuracy (Overall)│         0.8756 │
└────────────────────────┴────────────────┘

        Per-Slot Accuracy
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Slot Field      ┃ Accuracy ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ address         │   0.9200 │  (green: >= 0.9)
│ phone           │   0.8500 │  (yellow: >= 0.7)
│ device_number   │   0.9100 │  (green: >= 0.9)
│ intent_of_user  │   0.7800 │  (yellow: >= 0.7)
│ name            │   0.8900 │  (yellow: >= 0.7)
│ contract_id     │   0.6500 │  (red: < 0.7)
└─────────────────┴──────────┘
```

### Metrics Computed

| Metric | Formula | Description |
|--------|---------|-------------|
| Decision Accuracy | (TP + TN) / Total | Overall decision correctness |
| Decision Precision | TP / (TP + FP) | When predicting tool_call, how often correct |
| Decision Recall | TP / (TP + FN) | Of all tool_calls, how many detected |
| Decision F1 | 2 * P * R / (P + R) | Harmonic mean of precision and recall |
| Tool Accuracy | Correct / Total | Tool name classification accuracy |
| Slot Accuracy | Correct slots / Total | Overall slot presence accuracy |

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
    "slot_presence_labels": tensor([0, 1, 0, 0, 1, 0]) # [6] (phone, intent filled)
}
```

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
    "slot_presence_labels": tensor([
        [0., 1., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0.],
    ]),  # Shape: [4, 6]
}
```

### Step 5: Model Forward Pass

```python
from trm_agent.models import TRMConfig, TRMForToolCalling

config = TRMConfig(
    hidden_size=512,
    num_layers=2,
    num_tools=3,
    num_slots=6,
    n_latent_recursion=6,
    T_deep_recursion=3,
    N_supervision=16,
)
model = TRMForToolCalling(config)

# Move to device
device = torch.device("cuda")
model = model.to(device)
batch = {k: v.to(device) for k, v in batch.items()}

# Training forward (returns N_sup outputs)
all_outputs = model.train_step(
    input_ids=batch["input_ids"],           # [4, 64]
    attention_mask=batch["attention_mask"], # [4, 64]
    role_ids=batch["role_ids"],             # [4, 64]
)
```

**Output (list of 16 TRMOutput):**

```python
len(all_outputs)  # 16 (N_supervision steps)

# Each TRMOutput contains:
output = all_outputs[0]
output.decision_logits      # [4, 1]
output.tool_logits          # [4, 3] (num_tools=3)
output.arg_start_logits     # [4, 64, 10]
output.arg_end_logits       # [4, 64, 10]
output.slot_start_logits    # [4, 64, 6]
output.slot_end_logits      # [4, 64, 6]
output.slot_presence_logits # [4, 6]
output.q_logits             # [4, 1]
output.y                    # [4, 64, 512]
output.z                    # [4, 64, 512]
```

### Step 6: Loss Computation

```python
from trm_agent.training.losses import DeepSupervisionLoss

loss_fn = DeepSupervisionLoss(config)

losses = loss_fn(
    all_outputs,                        # List of 16 TRMOutput
    batch["decision_labels"],           # [4]
    batch["tool_name_labels"],          # [4]
    batch["slot_presence_labels"],      # [4, 6]
)
```

**Output:**

```python
losses = {
    "decision_loss": tensor(0.4532),  # Focal Loss
    "tool_loss": tensor(1.0234),      # Cross Entropy (masked)
    "slot_loss": tensor(0.2341),      # BCE
    "q_loss": tensor(0.6789),         # BCE
    "total_loss": tensor(1.8234),     # Weighted sum
}

# How total_loss is computed:
# total_loss = 1.0 * decision_loss + 1.0 * tool_loss
#            + 0.5 * slot_loss + 0.5 * q_loss
# = 1.0 * 0.4532 + 1.0 * 1.0234 + 0.5 * 0.2341 + 0.5 * 0.6789
# = 0.4532 + 1.0234 + 0.1171 + 0.3395
# = 1.9332 (approximately)
```

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

    slot_presence = torch.sigmoid(output.slot_presence_logits) > 0.5
    # tensor([[F, T, F, F, T, F],
    #         [F, F, F, F, F, F],
    #         [T, T, F, F, T, F],
    #         [F, F, F, F, F, F]])

    ema.restore()  # Restore original weights
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

# 3. Create model
config = TRMConfig(num_tools=len(dataset.tool_name_to_id))
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

        # Forward
        all_outputs = model.train_step(
            batch["input_ids"],
            batch["attention_mask"],
            batch["role_ids"],
        )

        # Loss
        losses = loss_fn(
            all_outputs,
            batch["decision_labels"],
            batch["tool_name_labels"],
            batch["slot_presence_labels"],
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
