# Loss Functions

This document describes the loss functions used for training the TRM model.

> **Note:** Span extraction (slots/params) is handled by **GLiNER2**, not TRM. TRM only handles decision classification and tool selection.

## Overview

TRM training uses loss functions combined into a weighted sum:

```
Total Loss = w_d * L_decision + w_t * L_tool + w_i * L_intent + w_q * L_q
```

| Loss | Weight | Description |
|------|--------|-------------|
| `L_decision` | 1.0 | Decision classification (Focal Loss) |
| `L_tool` | 1.0 | Tool name prediction (Cross Entropy) |
| `L_intent` | 1.0 | Intent prediction (Focal Cross-Entropy, optional) |
| `L_q` | 0.5 | Halting probability (BCE) |

> **Note:** Slot/arg span extraction losses are no longer used in TRM. See GLiNER2 documentation for entity extraction.
> **Note:** Intent loss is only computed when `--intent-map` is provided during training.

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

### 3. Intent Loss (Focal Cross-Entropy)

Multi-class classification for predicting the next assistant intent.

#### Why Focal Cross-Entropy?

Intent distribution is often highly imbalanced:
- Some intents (e.g., "greeting") are very common
- Others (e.g., "escalate_to_human") are rare

Standard Cross-Entropy biases toward majority classes. **Focal Cross-Entropy** addresses this by focusing on hard-to-classify examples.

#### Formula

```
FL(p_t) = -(1 - p_t)^γ * log(p_t)
```

Where:
- `p_t` = probability of the correct class
- `γ` = focusing parameter (default: 2.0)

#### Implementation

```python
class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=-1):
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # Filter out ignored indices
        mask = targets != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0)
        logits = logits[mask]
        targets = targets[mask]

        # Softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # Get probability of target class
        batch_size = targets.shape[0]
        pt = probs[torch.arange(batch_size), targets]

        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Apply focal weight
        return (focal_weight * ce_loss).mean()
```

#### Label Format

```python
intent_labels = [
    2,   # Sample 0: intent ID = 2 (e.g., "greeting")
    -1,  # Sample 1: unknown intent (ignored)
    0,   # Sample 2: intent ID = 0 (e.g., "check_balance")
    3,   # Sample 3: intent ID = 3 (e.g., "make_payment")
]
```

#### Enabling Intent Training

Intent training is enabled by providing `--intent-map` flag:

```bash
uv run python tools/train.py \
    --train-data data/train.jsonl \
    --intent-map data/intend.json \
    --output-dir outputs/
```

### 4. Slot Loss (Deprecated)

> **Deprecated:** Slot extraction is now handled by **GLiNER2**, not TRM. This section is kept for historical reference.

Binary classification for each slot's presence (no longer used in TRM training).

### 5. Q Loss (BCE)

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
        self.intent_loss = FocalCrossEntropyLoss(gamma=config.focal_gamma,
                                                  ignore_index=-1)
        self.q_loss = nn.BCEWithLogitsLoss()
        self.num_intents = config.num_intents

    def forward(self, outputs, decision_labels, tool_name_labels, intent_labels=None):
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

        # 3. Intent loss (if enabled)
        if self.num_intents > 0 and intent_labels is not None:
            intent_mask = intent_labels >= 0
            if intent_mask.any():
                losses["intent_loss"] = self.intent_loss(
                    outputs.intent_logits[intent_mask],
                    intent_labels[intent_mask]
                )
            else:
                losses["intent_loss"] = 0.0
        else:
            losses["intent_loss"] = 0.0

        # 4. Q loss
        is_correct = compute_correctness(outputs, decision_labels)
        losses["q_loss"] = self.q_loss(outputs.q_logits, is_correct)

        # 5. Total
        losses["total_loss"] = (
            w_d * losses["decision_loss"] +
            w_t * losses["tool_loss"] +
            w_i * losses["intent_loss"] +
            w_q * losses["q_loss"]
        )

        return losses
```

### Output

```python
losses = {
    "decision_loss": 0.234,   # Focal loss value
    "tool_loss": 1.456,       # Cross entropy value
    "intent_loss": 0.892,     # Focal cross-entropy value (if enabled)
    "q_loss": 0.567,          # BCE value
    "total_loss": 2.784,      # Weighted sum
}
```

## Deep Supervision Loss

Aggregates loss over all N_sup supervision steps.

> **Note:** With TBPTL enabled, only `N_sup - tbptl_no_grad_steps` outputs are collected for loss computation.

### Algorithm

```python
class DeepSupervisionLoss(nn.Module):
    def forward(self, all_outputs, decision_labels, tool_labels):
        total_losses = {"decision_loss": 0, "tool_loss": 0,
                        "q_loss": 0, "total_loss": 0}

        num_steps = len(all_outputs)  # N_sup - tbptl_no_grad_steps

        for outputs in all_outputs:
            step_losses = self.step_loss(outputs, decision_labels, tool_labels)
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
intent_loss_weight: 1.0  # Only used if num_intents > 0
q_loss_weight: 0.5

focal_alpha: 0.25
focal_gamma: 2.0
```

### Adjusting Weights

```python
config = TRMConfig(
    # Increase decision importance
    decision_loss_weight=2.0,

    # Enable intent prediction (num_intents set automatically from --intent-map)
    num_intents=10,
    intent_loss_weight=1.0,

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
    # Forward pass with deep supervision (TBPTL: first K steps no gradients)
    all_outputs = model.train_step(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        role_ids=batch["role_ids"],
    )

    # Compute loss over collected supervision steps
    losses = loss_fn(
        all_outputs,
        batch["decision_labels"],
        batch["tool_name_labels"],
    )

    # Backward pass
    losses["total_loss"].backward()

    # Log individual losses
    print(f"Decision: {losses['decision_loss']:.4f}")
    print(f"Tool: {losses['tool_loss']:.4f}")
    print(f"Q: {losses['q_loss']:.4f}")
    print(f"Total: {losses['total_loss']:.4f}")
```

### 5. Span Extraction Loss (Deprecated)

> **Deprecated:** Span extraction is now handled by **GLiNER2**, not TRM. This section is kept for historical reference.

Span extraction was previously used to locate values in the input context by predicting start/end token positions. This has been replaced by GLiNER2 for more accurate entity extraction.

## Training Phases

### Current Training

TRM training uses:
- **Decision loss** (Focal) - classify tool_call vs direct_answer
- **Tool loss** (Cross Entropy) - predict tool name for tool_call samples
- **Q loss** (BCE) - halting probability

> **Note:** Slot presence and span extraction losses have been removed. Use GLiNER2 for entity extraction.

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
| `intent_loss` | Decreases (if enabled, may fluctuate with imbalanced intents) |
| `q_loss` | Starts high, decreases as model improves |
| `total_loss` | Overall decreasing trend |

### Warning Signs

| Issue | Possible Cause |
|-------|----------------|
| `decision_loss` not decreasing | Learning rate too low, or severe imbalance |
| `tool_loss` = 0 always | No tool_call samples in batch |
| `intent_loss` = 0 always | No samples with intent labels in batch |
| `q_loss` stuck at ~0.69 | Model predicting random (log(2) ≈ 0.69) |
| Loss exploding | Learning rate too high, gradient issues |
