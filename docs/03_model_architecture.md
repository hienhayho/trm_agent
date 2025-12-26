# Model Architecture

This document describes the TRM (Tiny Recursive Model) architecture for tool-calling, based on the paper "Less is More: Recursive Reasoning with Tiny Networks".

## Overview

TRM is a recursive reasoning model that uses a **single tiny network** (2 layers, ~7M parameters) to iteratively refine answers through recursive computation.

```
                    ┌─────────────────────────────────────────────────┐
                    │                 TRMForToolCalling               │
                    ├─────────────────────────────────────────────────┤
                    │                                                 │
   input_ids ──────►│  InputEmbedding ──► x                          │
   role_ids ───────►│                                                 │
                    │  LatentEmbedding ──► y_init, z_init            │
                    │                                                 │
                    │         ┌──────────────────────┐                │
                    │         │    Deep Recursion    │                │
                    │         │  ┌────────────────┐  │                │
                    │    x ──►│  │    TRMBlock    │  │                │
                    │    y ──►│  │   (2 layers)   │──┼──► y', z'     │
                    │    z ──►│  └────────────────┘  │                │
                    │         └──────────────────────┘                │
                    │                    │                            │
                    │         ┌──────────┴──────────┐                 │
                    │         ▼                     ▼                 │
                    │    OutputHead              QHead                │
                    │    ┌────────┐           ┌────────┐              │
                    │    │Decision│           │Halting │              │
                    │    │  Tool  │           │  Prob  │              │
                    │    │ Slots  │           └────────┘              │
                    │    └────────┘                                   │
                    └─────────────────────────────────────────────────┘
```

## Core Concepts

### Key Variables

| Variable | Description | Shape |
|----------|-------------|-------|
| `x` | Embedded input (history + tools) | `[B, L, D]` |
| `y` | Current answer (decision + tool/content) | `[B, L, D]` |
| `z` | Latent reasoning state | `[B, L, D]` |

Where: B = batch_size, L = seq_len, D = hidden_size

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n` | 6 | Latent recursion iterations |
| `T` | 3 | Deep recursion iterations |
| `N_sup` | 16 | Supervision steps (training) |
| `num_layers` | 2 | Network depth (optimal per paper) |
| `hidden_size` | 512 | Embedding dimension |

## Configuration

```python
from trm_agent.models import TRMConfig

config = TRMConfig(
    # Architecture
    hidden_size=512,
    num_layers=2,
    num_heads=8,
    intermediate_size=2048,
    dropout=0.1,
    use_attention=True,

    # Vocabulary
    vocab_size=32000,
    max_seq_len=2048,

    # TRM specific
    n_latent_recursion=6,
    T_deep_recursion=3,
    N_supervision=16,

    # Output dimensions
    num_tools=10,
    num_slots=6,
    max_tool_args=10,
)
```

## Model Components

### 1. Input Embedding

Combines token, role, and learnable bias embeddings.

```
input_ids ──► TokenEmbedding ──┐
                               ├──► + ──► LayerNorm ──► Dropout ──► x
role_ids ───► RoleEmbedding ──┘
                    ▲
                    │
              input_bias (learnable)
```

**Components:**

| Component | Description |
|-----------|-------------|
| `TokenEmbedding` | Maps token IDs to vectors `[vocab_size, hidden_size]` |
| `RoleEmbedding` | Maps role IDs to vectors `[num_roles, hidden_size]` |
| `input_bias` | Learnable bias `[1, 1, hidden_size]` |

### 2. Latent Embedding

Initializes `y` and `z` with learnable parameters.

```python
y_init = nn.Parameter([1, 1, hidden_size])  # Expanded to [B, L, D]
z_init = nn.Parameter([1, 1, hidden_size])  # Expanded to [B, L, D]
```

### 3. TRM Block (Core Network)

Single tiny network used for both `z` and `y` updates.

```
                    ┌──────────────────────────────────────┐
                    │              TRMBlock                │
                    │  ┌────────────────────────────────┐  │
                    │  │       TransformerLayer 1       │  │
                    │  │  ┌──────────────────────────┐  │  │
    x ─────────────►│  │  │ RMSNorm ──► Attention   │  │  │
                    │  │  │            ▼            │  │  │
                    │  │  │        + residual       │  │  │
                    │  │  │            ▼            │  │  │
                    │  │  │ RMSNorm ──► SwiGLU MLP  │  │  │
                    │  │  │            ▼            │  │  │
                    │  │  │        + residual       │  │  │
                    │  │  └──────────────────────────┘  │  │
                    │  └────────────────────────────────┘  │
                    │                  ▼                   │
                    │  ┌────────────────────────────────┐  │
                    │  │       TransformerLayer 2       │  │
                    │  │          (same structure)      │  │
                    │  └────────────────────────────────┘  │
                    │                  ▼                   │
                    │             RMSNorm                  │
                    └──────────────────────────────────────┘
                                       ▼
                                    output
```

**Sub-components:**

#### RMSNorm

```python
def forward(x):
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rms * self.weight
```

#### SwiGLU

```python
def forward(x):
    gate = F.silu(self.gate_proj(x))
    up = self.up_proj(x)
    return self.down_proj(gate * up)
```

#### Multi-Head Attention with RoPE

```
Q, K, V = proj(x)
Q, K = apply_rotary(Q, K)
attn = softmax(Q @ K.T / sqrt(d)) @ V
output = proj(attn)
```

### 4. Output Heads

```
           y
           │
           ▼
    ┌──────────────┐
    │  OutputHead  │
    ├──────────────┤
    │              │
    │ ┌──────────┐ │
    │ │ Decision │─┼──► decision_logits [B, 1]
    │ │   Head   │ │    (tool_call=1, direct_answer=0)
    │ └──────────┘ │
    │              │
    │ ┌──────────┐ │
    │ │   Tool   │─┼──► tool_logits [B, num_tools]
    │ │   Head   │ │    arg_start_logits [B, L, max_args]
    │ └──────────┘ │    arg_end_logits [B, L, max_args]
    │              │
    │ ┌──────────┐ │
    │ │  Slots   │─┼──► slot_start_logits [B, L, num_slots]
    │ │   Head   │ │    slot_end_logits [B, L, num_slots]
    │ └──────────┘ │    slot_presence_logits [B, num_slots]
    │              │
    └──────────────┘

    ┌──────────────┐
    │    QHead     │──► q_logits [B, 1]
    │  (Halting)   │    (halting probability)
    └──────────────┘
```

#### DecisionHead

Binary classification for tool_call vs direct_answer.

```python
y_pooled = y.mean(dim=1)  # Global average pooling
logits = MLP(y_pooled)     # [B, hidden] -> [B, 1]
```

#### ToolHead

Predicts tool name and argument positions.

```python
# Tool name (classification)
tool_logits = MLP(y.mean(dim=1))  # [B, num_tools]

# Argument spans (per token)
arg_start = Linear(y)  # [B, L, max_args]
arg_end = Linear(y)    # [B, L, max_args]
```

#### SlotsHead

Predicts slot presence and positions.

```python
# Slot presence
slot_presence = MLP(y.mean(dim=1))  # [B, num_slots]

# Slot spans
slot_start = Linear(y)  # [B, L, num_slots]
slot_end = Linear(y)    # [B, L, num_slots]
```

#### QHead

Predicts halting probability for Adaptive Computational Time (ACT).

```python
q = MLP(y.mean(dim=1))  # [B, 1]
# Target: whether current prediction matches ground truth
```

## Recursion Algorithm

### Latent Recursion

```python
def latent_recursion(x, y, z, n=6):
    """
    n iterations of latent reasoning, then refine y.
    """
    # Update z with x present (reasoning step)
    for _ in range(n):
        z = net(x + y + z)

    # Refine y without x (answer refinement step)
    y = net(y + z)

    return y, z
```

### Deep Recursion

```python
def deep_recursion(x, y, z, n=6, T=3):
    """
    T-1 times without grad, 1 time with grad.
    """
    # T-1 recursions without gradients (save memory)
    with torch.no_grad():
        for _ in range(T - 1):
            y, z = latent_recursion(x, y, z, n)

    # Final recursion with gradients
    y, z = latent_recursion(x, y, z, n)

    # Get predictions
    outputs = output_head(y)
    q = q_head(y)

    return y.detach(), z.detach(), outputs, q
```

### Deep Supervision Training

```python
def train_step(x, y, z, N_sup=16):
    """
    Training with deep supervision.
    """
    all_outputs = []

    for step in range(N_sup):
        y, z, outputs, q = deep_recursion(x, y, z)
        all_outputs.append(outputs)

    return all_outputs  # Loss computed over all steps
```

## Forward Pass Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Forward Pass                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Input Embedding                                                     │
│     x = embed(input_ids, role_ids)                                     │
│                                                                         │
│  2. Initialize Latent States                                            │
│     y = y_init.expand(batch, seq_len, hidden)                          │
│     z = z_init.expand(batch, seq_len, hidden)                          │
│                                                                         │
│  3. Deep Supervision Loop (Training)                                    │
│     for step in range(N_sup):                                          │
│         ┌─────────────────────────────────────────────────────────┐    │
│         │ Deep Recursion (T iterations)                           │    │
│         │                                                         │    │
│         │   for t in range(T-1):  # No gradients                  │    │
│         │       ┌───────────────────────────────────────────┐     │    │
│         │       │ Latent Recursion (n iterations)           │     │    │
│         │       │   for i in range(n):                      │     │    │
│         │       │       z = net(x + y + z)  # Reasoning     │     │    │
│         │       │   y = net(y + z)          # Refinement    │     │    │
│         │       └───────────────────────────────────────────┘     │    │
│         │                                                         │    │
│         │   # Final with gradients                                │    │
│         │   y, z = latent_recursion(x, y, z, n)                  │    │
│         │   outputs = output_head(y)                              │    │
│         │   q = q_head(y)                                         │    │
│         │   y, z = y.detach(), z.detach()                        │    │
│         └─────────────────────────────────────────────────────────┘    │
│         all_outputs.append(outputs)                                    │
│                                                                         │
│  4. Compute Loss (over all supervision steps)                          │
│     loss = sum(loss_fn(output, target) for output in all_outputs)      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Output Structure

```python
@dataclass
class TRMOutput:
    decision_logits: Tensor      # [B, 1] - tool_call vs direct_answer
    tool_logits: Tensor          # [B, num_tools] - tool classification
    arg_start_logits: Tensor     # [B, L, max_args] - argument start
    arg_end_logits: Tensor       # [B, L, max_args] - argument end
    slot_start_logits: Tensor    # [B, L, num_slots] - slot start
    slot_end_logits: Tensor      # [B, L, num_slots] - slot end
    slot_presence_logits: Tensor # [B, num_slots] - slot presence
    q_logits: Tensor             # [B, 1] - halting probability
    y: Tensor                    # [B, L, D] - final answer embedding
    z: Tensor                    # [B, L, D] - final latent embedding
```

## Usage Examples

### Creating Model

```python
from trm_agent.models import TRMConfig, TRMForToolCalling

config = TRMConfig(
    hidden_size=512,
    num_layers=2,
    num_tools=15,
    num_slots=6,
)

model = TRMForToolCalling(config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Training Forward Pass

```python
# Training with deep supervision
outputs = model.train_step(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    role_ids=batch["role_ids"],
)
# Returns list of N_sup TRMOutput objects
```

### Inference

```python
# Inference (full N_sup steps)
with torch.no_grad():
    output = model.inference(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        role_ids=batch["role_ids"],
    )

# Get predictions
decision = torch.sigmoid(output.decision_logits) > 0.5
tool_id = output.tool_logits.argmax(dim=-1)
slot_presence = torch.sigmoid(output.slot_presence_logits) > 0.5
```

## Why This Architecture Works

1. **2 layers is optimal**: More layers overfit on small datasets
2. **Single network**: Same network for reasoning (z) and refinement (y)
3. **Full backprop**: No 1-step gradient approximation needed
4. **Separate y and z**:
   - `y` stores current solution
   - `z` stores reasoning state (prevents forgetting)
5. **Deep supervision**: Model learns to improve any (y, z) state
6. **ACT with Q-head**: Enables early stopping when confident

## Model Size

With default configuration:

| Component | Parameters |
|-----------|------------|
| Token Embedding | 16.4M |
| Role Embedding | 2K |
| TRM Block (2 layers) | 6.3M |
| Output Heads | 1.2M |
| **Total** | **~24M** |

For smaller model (paper's 7M):
- Reduce hidden_size to 256
- Reduce vocab_size
