# Model Architecture

This document describes the TRM (Tiny Recursive Model) architecture for tool-calling, based on the paper "Less is More: Recursive Reasoning with Tiny Networks".

## Overview

TRM is a recursive reasoning model that uses a **single tiny network** (2 layers, ~7M parameters) to iteratively refine answers through recursive computation.

```mermaid
flowchart TB
    subgraph Input["Input Processing"]
        Conversation["Conversation History"]
        input_ids["input_ids<br/>[B, L]"]
        role_ids["role_ids<br/>[B, L]"]
        InputEmbed["InputEmbedding"]
    end

    subgraph Latent["Latent Initialization"]
        LatentEmbed["LatentEmbedding"]
        y_init["y_init"]
        z_init["z_init"]
    end

    subgraph Recursion["Deep Recursion Loop"]
        direction TB
        TRMBlock["TRMBlock<br/>(2 Layers)"]

        subgraph LatentRecursion["Latent Recursion (n=6)"]
            z_update["z = net(x + y + z)"]
        end

        subgraph Refinement["Answer Refinement"]
            y_update["y = net(y + z)"]
        end
    end

    subgraph Outputs["Output Heads"]
        DecisionHead["DecisionHead<br/>→ decision_logits"]
        ToolHead["ToolHead<br/>→ tool_logits"]
        UnifiedParamHead["UnifiedParamHead<br/>→ param_start/end/presence"]
        QHead["QHead<br/>→ q_logits"]
    end

    Conversation --> input_ids
    Conversation --> role_ids
    input_ids --> InputEmbed
    role_ids --> InputEmbed
    InputEmbed --> |"x"| TRMBlock
    LatentEmbed --> y_init
    LatentEmbed --> z_init
    y_init --> |"y"| TRMBlock
    z_init --> |"z"| TRMBlock

    TRMBlock --> z_update
    z_update --> |"n times"| z_update
    z_update --> y_update
    y_update --> |"y'"| DecisionHead
    y_update --> |"y'"| ToolHead
    y_update --> |"y'"| UnifiedParamHead
    y_update --> |"y'"| QHead
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

    # Unified fields (slots + tool params)
    slot_fields=["address", "phone", "device_number", "name"],
)
```

## Model Components

### Component Hierarchy

```mermaid
classDiagram
    class TRMForToolCalling {
        +InputEmbedding input_embedding
        +LatentEmbedding latent_embedding
        +TRMBlock net
        +OutputHead output_head
        +QHead q_head
        +forward()
        +train_step()
        +inference()
    }

    class InputEmbedding {
        +TokenEmbedding token_embedding
        +RoleEmbedding role_embedding
        +ToolEmbedding tool_embedding
        +Parameter input_bias
        +forward()
    }

    class LatentEmbedding {
        +Parameter y_init
        +Parameter z_init
        +get_initial_y()
        +get_initial_z()
    }

    class TRMBlock {
        +ModuleList~TransformerLayer~ layers
        +RMSNorm final_norm
        +forward()
    }

    class TransformerLayer {
        +MultiHeadAttention attn
        +SwiGLU mlp
        +RMSNorm norm1
        +RMSNorm norm2
        +forward()
    }

    class OutputHead {
        +DecisionHead decision_head
        +ToolHead tool_head
        +UnifiedParamHead unified_param_head
        +ContentHead content_head
        +forward()
    }

    TRMForToolCalling --> InputEmbedding
    TRMForToolCalling --> LatentEmbedding
    TRMForToolCalling --> TRMBlock
    TRMForToolCalling --> OutputHead
    TRMForToolCalling --> QHead
    TRMBlock --> TransformerLayer
    OutputHead --> DecisionHead
    OutputHead --> ToolHead
    OutputHead --> UnifiedParamHead
```

### 1. Input Embedding

Combines token, role, and learnable bias embeddings.

```mermaid
flowchart LR
    subgraph Inputs
        input_ids["input_ids<br/>[B, L]"]
        role_ids["role_ids<br/>[B, L]"]
    end

    subgraph Embeddings
        TokenEmb["TokenEmbedding<br/>[vocab, D]"]
        RoleEmb["RoleEmbedding<br/>[num_roles, D]"]
        Bias["input_bias<br/>[1, 1, D]"]
    end

    subgraph Processing
        Add(("+"))
        LN["LayerNorm"]
        Drop["Dropout"]
    end

    input_ids --> TokenEmb
    role_ids --> RoleEmb
    TokenEmb --> Add
    RoleEmb --> Add
    Bias --> Add
    Add --> LN --> Drop --> x["x<br/>[B, L, D]"]
```

**Components:**

| Component | Description | Shape |
|-----------|-------------|-------|
| `TokenEmbedding` | Maps token IDs to vectors | `[vocab_size, hidden_size]` |
| `RoleEmbedding` | Maps role IDs to vectors | `[num_roles, hidden_size]` |
| `input_bias` | Learnable bias | `[1, 1, hidden_size]` |

### 2. Latent Embedding

Initializes `y` and `z` with learnable parameters.

```python
y_init = nn.Parameter([1, 1, hidden_size])  # Expanded to [B, L, D]
z_init = nn.Parameter([1, 1, hidden_size])  # Expanded to [B, L, D]
```

### 3. TRM Block (Core Network)

Single tiny network used for both `z` and `y` updates.

```mermaid
flowchart TB
    subgraph TRMBlock["TRMBlock"]
        Input["Input<br/>[B, L, D]"]

        subgraph Layer1["TransformerLayer 1"]
            direction TB
            N1_1["RMSNorm"]
            Attn1["MultiHeadAttention<br/>+ RoPE"]
            Add1(("+"))
            N2_1["RMSNorm"]
            MLP1["SwiGLU MLP"]
            Add2(("+"))
        end

        subgraph Layer2["TransformerLayer 2"]
            direction TB
            N1_2["RMSNorm"]
            Attn2["MultiHeadAttention<br/>+ RoPE"]
            Add3(("+"))
            N2_2["RMSNorm"]
            MLP2["SwiGLU MLP"]
            Add4(("+"))
        end

        FinalNorm["RMSNorm"]
        Output["Output<br/>[B, L, D]"]
    end

    Input --> N1_1 --> Attn1 --> Add1
    Input --> Add1
    Add1 --> N2_1 --> MLP1 --> Add2
    Add1 --> Add2

    Add2 --> N1_2 --> Attn2 --> Add3
    Add2 --> Add3
    Add3 --> N2_2 --> MLP2 --> Add4
    Add3 --> Add4

    Add4 --> FinalNorm --> Output
```

**Sub-components:**

#### RMSNorm

```python
def forward(x):
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rms * self.weight
```

#### SwiGLU

```mermaid
flowchart LR
    x["x"] --> gate_proj["gate_proj"]
    x --> up_proj["up_proj"]
    gate_proj --> silu["SiLU"]
    silu --> mul(("×"))
    up_proj --> mul
    mul --> down_proj["down_proj"] --> out["output"]
```

```python
def forward(x):
    gate = F.silu(self.gate_proj(x))
    up = self.up_proj(x)
    return self.down_proj(gate * up)
```

#### Multi-Head Attention with RoPE

```mermaid
flowchart TB
    x["x [B, L, D]"] --> Q["Q = q_proj(x)"]
    x --> K["K = k_proj(x)"]
    x --> V["V = v_proj(x)"]

    Q --> Reshape1["Reshape<br/>[B, H, L, d]"]
    K --> Reshape2["Reshape<br/>[B, H, L, d]"]
    V --> Reshape3["Reshape<br/>[B, H, L, d]"]

    Reshape1 --> RoPE1["Apply RoPE"]
    Reshape2 --> RoPE2["Apply RoPE"]

    RoPE1 --> Scores["Q @ K.T / √d"]
    RoPE2 --> Scores

    Scores --> Mask["Apply Mask"]
    Mask --> Softmax["Softmax"]
    Softmax --> Dropout["Dropout"]
    Dropout --> Attn["@ V"]
    Reshape3 --> Attn

    Attn --> Reshape4["Reshape<br/>[B, L, D]"]
    Reshape4 --> o_proj["o_proj"] --> Output["Output"]
```

### 4. Output Heads

```mermaid
flowchart TB
    y["y<br/>[B, L, D]"]

    subgraph OutputHead["OutputHead"]
        subgraph Decision["DecisionHead"]
            D_Norm["RMSNorm"]
            D_Pool["Mean Pool"]
            D_MLP["MLP"]
            D_Out["decision_logits<br/>[B, 1]"]
        end

        subgraph Tool["ToolHead"]
            T_Norm["RMSNorm"]
            T_Pool["Mean Pool"]
            T_MLP["MLP"]
            T_Out["tool_logits<br/>[B, num_tools]"]
        end

        subgraph Unified["UnifiedParamHead"]
            U_Norm["RMSNorm"]
            U_Start["start_classifier"]
            U_End["end_classifier"]
            U_Pool["Mean Pool"]
            U_Presence["presence_classifier"]
            U_Out1["param_start_logits<br/>[B, L, num_unified]"]
            U_Out2["param_end_logits<br/>[B, L, num_unified]"]
            U_Out3["param_presence_logits<br/>[B, num_unified]"]
        end
    end

    subgraph QH["QHead"]
        Q_Norm["RMSNorm"]
        Q_Pool["Mean Pool"]
        Q_MLP["MLP"]
        Q_Out["q_logits<br/>[B, 1]"]
    end

    y --> D_Norm --> D_Pool --> D_MLP --> D_Out
    y --> T_Norm --> T_Pool --> T_MLP --> T_Out
    y --> U_Norm
    U_Norm --> U_Start --> U_Out1
    U_Norm --> U_End --> U_Out2
    U_Norm --> U_Pool --> U_Presence --> U_Out3
    y --> Q_Norm --> Q_Pool --> Q_MLP --> Q_Out
```

#### DecisionHead

Binary classification for tool_call vs direct_answer.

```python
y_pooled = y.mean(dim=1)  # Global average pooling
logits = MLP(y_pooled)     # [B, hidden] -> [B, 1]
```

#### ToolHead

Predicts tool name.

```python
# Tool name (classification)
tool_logits = MLP(y.mean(dim=1))  # [B, num_tools]
```

#### UnifiedParamHead

Unified extraction for both content slots and tool parameters with decision-based + tool-based masking.

```mermaid
flowchart TB
    subgraph Fields["Unified Fields"]
        Slots["Slot Fields<br/>(always valid)"]
        Params["Tool Param Fields<br/>(tool-specific mask)"]
    end

    subgraph Masking["Masking Strategy"]
        DA["direct_answer:<br/>slots=✓, params=✗"]
        TC["tool_call:<br/>slots=✓, params=tool-specific"]
    end

    Slots --> DA
    Slots --> TC
    Params --> TC
```

**Unified Fields:**
- `unified_fields = slot_fields + tool_param_fields` (deduplicated)
- Slots are always extracted regardless of decision
- Tool params are only valid for tool_call with tool-specific masking

```python
# Span prediction (per token)
param_start = Linear(y)  # [B, L, num_unified_fields]
param_end = Linear(y)    # [B, L, num_unified_fields]

# Presence prediction (pooled)
param_presence = MLP(y.mean(dim=1))  # [B, num_unified_fields]
```

#### QHead

Predicts halting probability for Adaptive Computational Time (ACT).

```python
q = MLP(y.mean(dim=1))  # [B, 1]
# Target: whether current prediction matches ground truth
```

## Recursion Algorithm

### Latent Recursion

```mermaid
flowchart TB
    subgraph LatentRecursion["latent_recursion(x, y, z, n=6)"]
        direction TB

        subgraph Loop["for i in range(n):"]
            Combine1["combined = x + y + z"]
            Net1["z = net(combined)"]
        end

        Combine2["combined = y + z"]
        Net2["y = net(combined)"]

        Return["return y, z"]
    end

    Loop --> Combine2 --> Net2 --> Return
```

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

```mermaid
flowchart TB
    subgraph DeepRecursion["deep_recursion(x, y, z, n=6, T=3)"]
        direction TB

        subgraph NoGrad["torch.no_grad() - T-1 times"]
            LR1["y, z = latent_recursion(x, y, z, n)"]
        end

        subgraph WithGrad["With Gradients - 1 time"]
            LR2["y, z = latent_recursion(x, y, z, n)"]
        end

        Outputs["outputs = output_head(y)"]
        Q["q = q_head(y)"]
        Detach["y, z = y.detach(), z.detach()"]
        Return["return y, z, outputs, q"]
    end

    NoGrad --> WithGrad --> Outputs --> Q --> Detach --> Return
```

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

```mermaid
flowchart TB
    subgraph Training["train_step with Deep Supervision"]
        Init["x = embed(input)<br/>y = y_init, z = z_init"]

        subgraph Loop["for step in range(N_sup=16):"]
            DR["y, z, outputs, q = deep_recursion(x, y, z)"]
            Collect["all_outputs.append(outputs)"]
        end

        Loss["loss = Σ loss_fn(output, target)"]
        Return["return all_outputs"]
    end

    Init --> Loop --> Loss --> Return
```

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

```mermaid
sequenceDiagram
    participant I as Input
    participant E as Embeddings
    participant R as Recursion
    participant O as Outputs

    Note over I,O: Forward Pass

    I->>E: input_ids, role_ids
    E->>E: x = InputEmbedding(input_ids, role_ids)
    E->>E: y = y_init.expand(B, L, D)
    E->>E: z = z_init.expand(B, L, D)

    loop N_sup times (Deep Supervision)
        Note over R: Deep Recursion
        loop T-1 times (no grad)
            loop n times (Latent Recursion)
                R->>R: z = net(x + y + z)
            end
            R->>R: y = net(y + z)
        end
        Note over R: Final with grad
        loop n times
            R->>R: z = net(x + y + z)
        end
        R->>R: y = net(y + z)
        R->>O: outputs = output_head(y)
        R->>O: q = q_head(y)
        R->>R: y, z = detach(y, z)
    end

    O->>O: Compute loss over all steps
```

## Output Structure

```python
@dataclass
class TRMOutput:
    decision_logits: Tensor        # [B, 1] - tool_call vs direct_answer
    tool_logits: Tensor            # [B, num_tools] - tool classification
    param_start_logits: Tensor     # [B, L, num_unified] - param span start
    param_end_logits: Tensor       # [B, L, num_unified] - param span end
    param_presence_logits: Tensor  # [B, num_unified] - param presence
    q_logits: Tensor               # [B, 1] - halting probability
    y: Tensor                      # [B, L, D] - final answer embedding
    z: Tensor                      # [B, L, D] - final latent embedding
```

## Unified Parameter Extraction

The model uses a unified approach for extracting both content slots and tool parameters:

```mermaid
flowchart LR
    subgraph UnifiedFields["unified_fields"]
        direction TB
        S1["address"]
        S2["phone"]
        S3["device_number"]
        S4["name"]
        P1["product"]
        P2["reason"]
        P3["..."]
    end

    subgraph Slots["slot_fields<br/>(always valid)"]
        S1
        S2
        S3
        S4
    end

    subgraph Params["tool_param_fields<br/>(tool-specific)"]
        P1
        P2
        P3
    end
```

### Masking Example

| Decision | Tool | address | phone | name | product | reason |
|----------|------|---------|-------|------|---------|--------|
| direct_answer | - | ✓ | ✓ | ✓ | ✗ | ✗ |
| tool_call | tool_A | ✓ | ✓ | ✓ | ✓ | ✗ |
| tool_call | tool_B | ✓ | ✓ | ✓ | ✗ | ✓ |

## Usage Examples

### Creating Model

```python
from trm_agent.models import TRMConfig, TRMForToolCalling

config = TRMConfig(
    hidden_size=512,
    num_layers=2,
    num_tools=15,
    slot_fields=["address", "phone", "device_number", "name"],
)

# Set tool param fields (auto-collected from dataset)
config.set_tool_param_fields(["product", "reason", "quantity"])

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
param_presence = torch.sigmoid(output.param_presence_logits) > 0.5
```

## Why This Architecture Works

```mermaid
mindmap
  root((TRM))
    2 Layers
      Prevents overfitting
      Optimal per paper
    Single Network
      Same for z and y updates
      Task determined by input
    Full Backprop
      No 1-step approximation
      No fixed-point theorem
    Separate y and z
      y: current solution
      z: reasoning state
      Prevents forgetting
    Deep Supervision
      Learns to improve any state
      Progressive refinement
    ACT with Q-head
      Early stopping
      Confidence estimation
```

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
