# Loss Functions

This document describes the loss functions used for training the TRM model.

## Overview

TRM training uses multiple loss functions combined into a weighted sum:

```
Total Loss = w_d * L_decision + w_t * L_tool + w_s * L_slot + w_q * L_q + w_ss * L_slot_span + w_as * L_arg_span
```

| Loss | Weight | Description |
|------|--------|-------------|
| `L_decision` | 1.0 | Decision classification (Focal Loss) |
| `L_tool` | 1.0 | Tool name prediction (Cross Entropy) |
| `L_slot` | 0.5 | Slot presence prediction (BCE) |
| `L_q` | 0.5 | Halting probability (BCE) |
| `L_slot_span` | 0.5 | Slot value span extraction (Cross Entropy) |
| `L_arg_span` | 0.5 | Tool argument span extraction (Cross Entropy) |

## Loss Components

### 1. Decision Loss (Focal Loss)

Handles imbalanced classification between `tool_call` and `direct_answer`.

#### Why Focal Loss?

In tool-calling datasets, the distribution is often imbalanced:
- Some datasets have more `tool_call` samples
- Some have more `direct_answer` samples

Standard BCE would bias toward the majority class. Focal Loss addresses this by:
1. Down-weighting easy examples
2. Focusing on hard examples

#### Formula

```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

Where:
- `p_t` = probability of correct class
- `α` = class weight (default: 0.25 for positive class)
- `γ` = focusing parameter (default: 2.0)

#### Visualization

```
                    Standard BCE vs Focal Loss
    Loss
      │
    1.0│   ╲
      │    ╲  BCE
      │     ╲
    0.5│      ╲___________
      │        ╲
      │    FL   ╲__________ (γ=2)
    0.0│__________╲__________
      └──────────────────────► p_t
       0.0      0.5       1.0

When p_t is high (easy example):
  - BCE: Still contributes significantly
  - Focal: Contribution reduced by (1-p_t)^γ

When p_t is low (hard example):
  - Both losses are high
  - Focal focuses training on these
```

#### Implementation

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # Standard BCE
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Probability of correct class
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Final focal loss
        return (alpha_weight * focal_weight * bce_loss).mean()
```

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.25 | Weight for positive class (tool_call) |
| `gamma` | 2.0 | Focusing parameter (higher = more focus on hard examples) |

#### Effect of Gamma

```
γ = 0: Standard BCE (no focusing)
γ = 1: Moderate focusing
γ = 2: Strong focusing (default)
γ = 5: Very strong focusing (may be unstable)
```

### 2. Tool Loss (Cross Entropy)

Classifies which tool to call when decision is `tool_call`.

#### Formula

```
L_tool = CrossEntropy(tool_logits, tool_labels)
```

#### Masking

Only computed for samples where `decision == tool_call`:

```python
tool_mask = tool_name_labels >= 0  # -1 indicates direct_answer
if tool_mask.any():
    loss = CrossEntropyLoss(outputs.tool_logits[tool_mask],
                            tool_name_labels[tool_mask])
else:
    loss = 0.0
```

#### Label Format

```python
tool_name_labels = [
    3,   # Sample 0: tool_call, tool ID = 3
    -1,  # Sample 1: direct_answer (ignored)
    0,   # Sample 2: tool_call, tool ID = 0
    -1,  # Sample 3: direct_answer (ignored)
]
```

### 3. Slot Loss (BCE)

Binary classification for each slot's presence.

#### Formula

```
L_slot = BCEWithLogits(slot_presence_logits, slot_presence_labels)
```

#### Label Format

```python
slot_presence_labels = [
    [1, 1, 0, 0, 1, 0],  # Sample 0: slots 0,1,4 are filled
    [0, 1, 0, 0, 0, 1],  # Sample 1: slots 1,5 are filled
    ...
]
# Shape: [batch_size, num_slots]
```

#### Slot Fields

Default slots tracked:

| Index | Field |
|-------|-------|
| 0 | address |
| 1 | phone |
| 2 | device_number |
| 3 | intent_of_user |
| 4 | name |
| 5 | contract_id |

### 4. Q Loss (BCE)

Trains the halting head for Adaptive Computational Time (ACT).

#### Purpose

The Q head learns to predict whether the current answer is correct:
- Q → 1: Model is confident, can stop early
- Q → 0: Model needs more iterations

#### Target Computation

```python
with torch.no_grad():
    # Get model's decision prediction
    pred_decision = (sigmoid(outputs.decision_logits) > 0.5).float()

    # Target: 1 if prediction matches ground truth, 0 otherwise
    is_correct = (pred_decision == decision_labels).float()

q_loss = BCEWithLogits(outputs.q_logits, is_correct)
```

#### Flow

```
                          Q Head Training Target

    Model Prediction          Ground Truth           Q Target
    ─────────────────         ────────────           ────────
    tool_call (1)      ==     tool_call (1)    →       1 (correct)
    tool_call (1)      !=     direct_answer (0) →      0 (incorrect)
    direct_answer (0)  ==     direct_answer (0) →      1 (correct)
    direct_answer (0)  !=     tool_call (1)     →      0 (incorrect)
```

## Combined Loss (TRMLoss)

### Structure

```python
class TRMLoss(nn.Module):
    def __init__(self, config):
        self.decision_loss = FocalLoss(alpha=config.focal_alpha,
                                        gamma=config.focal_gamma)
        self.tool_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.slot_loss = nn.BCEWithLogitsLoss()
        self.q_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, decision_labels, tool_name_labels, slot_presence_labels):
        losses = {}

        # 1. Decision loss
        losses["decision_loss"] = self.decision_loss(outputs.decision_logits,
                                                      decision_labels)

        # 2. Tool loss (masked)
        tool_mask = tool_name_labels >= 0
        if tool_mask.any():
            losses["tool_loss"] = self.tool_loss(outputs.tool_logits[tool_mask],
                                                  tool_name_labels[tool_mask])
        else:
            losses["tool_loss"] = 0.0

        # 3. Slot loss
        losses["slot_loss"] = self.slot_loss(outputs.slot_presence_logits,
                                              slot_presence_labels)

        # 4. Q loss
        is_correct = compute_correctness(outputs, decision_labels)
        losses["q_loss"] = self.q_loss(outputs.q_logits, is_correct)

        # 5. Total
        losses["total_loss"] = (
            w_d * losses["decision_loss"] +
            w_t * losses["tool_loss"] +
            w_s * losses["slot_loss"] +
            w_q * losses["q_loss"]
        )

        return losses
```

### Output

```python
losses = {
    "decision_loss": 0.234,   # Focal loss value
    "tool_loss": 1.456,       # Cross entropy value
    "slot_loss": 0.123,       # BCE value
    "q_loss": 0.567,          # BCE value
    "total_loss": 1.892,      # Weighted sum
}
```

## Deep Supervision Loss

Aggregates loss over all N_sup supervision steps.

### Algorithm

```python
class DeepSupervisionLoss(nn.Module):
    def forward(self, all_outputs, decision_labels, tool_labels, slot_labels):
        total_losses = {"decision_loss": 0, "tool_loss": 0,
                        "slot_loss": 0, "q_loss": 0, "total_loss": 0}

        num_steps = len(all_outputs)  # N_sup = 16

        for outputs in all_outputs:
            step_losses = self.step_loss(outputs, decision_labels,
                                          tool_labels, slot_labels)
            for key in total_losses:
                total_losses[key] += step_losses[key]

        # Average over steps
        for key in total_losses:
            total_losses[key] /= num_steps

        return total_losses
```

### Visualization

```
Step 1    Step 2    Step 3    ...    Step N_sup
  │         │         │                  │
  ▼         ▼         ▼                  ▼
┌─────┐  ┌─────┐  ┌─────┐           ┌─────┐
│Loss1│  │Loss2│  │Loss3│    ...    │LossN│
└──┬──┘  └──┬──┘  └──┬──┘           └──┬──┘
   │        │        │                  │
   └────────┴────────┴───────┬──────────┘
                             │
                             ▼
                    Average Loss = Σ(Loss_i) / N_sup
```

### Why Average?

- Each supervision step should improve the answer
- Averaging ensures all steps contribute equally
- Prevents overfitting to later steps

## Configuration

### Default Weights

```yaml
# In config
decision_loss_weight: 1.0
tool_loss_weight: 1.0
slots_loss_weight: 0.5
q_loss_weight: 0.5
slot_span_loss_weight: 0.5
arg_span_loss_weight: 0.5

focal_alpha: 0.25
focal_gamma: 2.0
```

### Adjusting Weights

```python
config = TRMConfig(
    # Increase decision importance
    decision_loss_weight=2.0,

    # Decrease slot importance
    slots_loss_weight=0.25,

    # Adjust focal loss for different imbalance
    focal_alpha=0.5,   # Equal weight for both classes
    focal_gamma=1.0,   # Less aggressive focusing
)
```

## Usage Example

```python
from trm_agent.training.losses import DeepSupervisionLoss
from trm_agent.models import TRMConfig

config = TRMConfig()
loss_fn = DeepSupervisionLoss(config)

# Training loop
for batch in dataloader:
    # Forward pass with deep supervision
    all_outputs = model.train_step(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        role_ids=batch["role_ids"],
    )

    # Compute loss over all supervision steps
    losses = loss_fn(
        all_outputs,
        batch["decision_labels"],
        batch["tool_name_labels"],
        batch["slot_presence_labels"],
    )

    # Backward pass
    losses["total_loss"].backward()

    # Log individual losses
    print(f"Decision: {losses['decision_loss']:.4f}")
    print(f"Tool: {losses['tool_loss']:.4f}")
    print(f"Slot: {losses['slot_loss']:.4f}")
    print(f"Q: {losses['q_loss']:.4f}")
    print(f"Total: {losses['total_loss']:.4f}")
```

### 5. Span Extraction Loss

Extracts values from input context by predicting start/end token positions.

#### Purpose

Instead of generating text, span extraction locates values in the input:
- Slot values (e.g., phone number, address mentioned in conversation)
- Tool argument values (e.g., query mentioned by user)

#### Formula

```
L_span = (L_start + L_end) / 2

where:
L_start = CrossEntropy(start_logits, start_labels)
L_end = CrossEntropy(end_logits, end_labels)
```

#### Label Format

Labels are token positions (0 to seq_len-1), or -1 if not found:

```python
# Slot span labels
slot_start_labels = [42, -1, 15, 28, -1, 7]   # [num_slots]
slot_end_labels = [45, -1, 18, 32, -1, 12]    # [num_slots]

# Argument span labels
arg_start_labels = [42, 67, -1, -1, ...]      # [max_tool_args]
arg_end_labels = [45, 72, -1, -1, ...]        # [max_tool_args]
```

#### Implementation

```python
class SpanExtractionLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, start_logits, end_logits, start_labels, end_labels):
        batch_size, seq_len, num_spans = start_logits.shape

        # Transpose: [batch, seq_len, num_spans] → [batch, num_spans, seq_len]
        start_logits = start_logits.transpose(1, 2)
        end_logits = end_logits.transpose(1, 2)

        # Reshape for cross entropy: [batch * num_spans, seq_len]
        start_logits = start_logits.reshape(-1, seq_len)
        end_logits = end_logits.reshape(-1, seq_len)

        # Flatten labels: [batch * num_spans]
        start_labels = start_labels.view(-1)
        end_labels = end_labels.view(-1)

        # Compute losses (ignore_index=-1 handles missing spans)
        start_loss = self.ce_loss(start_logits, start_labels)
        end_loss = self.ce_loss(end_logits, end_labels)

        return (start_loss + end_loss) / 2
```

#### Handling Missing Spans

When a value cannot be found in the input context:
- Labels set to -1
- CrossEntropyLoss ignores these with `ignore_index=-1`
- No gradient contribution for unfindable values

#### Slot vs Argument Spans

| Type | Shape | When Computed |
|------|-------|---------------|
| Slot spans | `[batch, num_slots]` | Always (all samples) |
| Arg spans | `[batch, max_args]` | Only for `tool_call` samples |

## Training Phases

### Current Training

Full loss with span extraction:
- Decision loss (Focal)
- Tool loss (only for tool_call)
- Slot presence loss
- Q loss (halting)
- **Slot span extraction loss** (extracting slot values)
- **Argument span extraction loss** (extracting tool arguments)

### Future Phase

Add content generation:
- Content loss (Cross Entropy over vocabulary)

```python
# Future addition
content_loss = CrossEntropyLoss(content_logits, content_labels)
```

## Monitoring Training

### Healthy Training Signs

| Metric | Expected Behavior |
|--------|-------------------|
| `decision_loss` | Decreases steadily |
| `tool_loss` | Decreases (may fluctuate if few tool_call samples) |
| `slot_loss` | Decreases |
| `q_loss` | Starts high, decreases as model improves |
| `slot_span_loss` | Decreases as model learns to locate values |
| `arg_span_loss` | Decreases (only computed for tool_call samples) |
| `total_loss` | Overall decreasing trend |

### Warning Signs

| Issue | Possible Cause |
|-------|----------------|
| `decision_loss` not decreasing | Learning rate too low, or severe imbalance |
| `tool_loss` = 0 always | No tool_call samples in batch |
| `q_loss` stuck at ~0.69 | Model predicting random (log(2) ≈ 0.69) |
| `slot_span_loss` not decreasing | Values not in context, or offset mapping issues |
| `arg_span_loss` not decreasing | Few tool_call samples, or value not in context |
| Loss exploding | Learning rate too high, gradient issues |
