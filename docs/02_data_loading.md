# Data Loading and Tokenization

This document describes how training data is loaded, tokenized, and prepared for the TRM model.

> **Note:** Slot/parameter extraction is handled by **GLiNER2**, not TRM. TRM only handles decision classification (tool_call vs direct_answer) and tool selection.

## Overview

The data pipeline consists of three main components:

1. **TRMTokenizer** - SentencePiece tokenizer with special tokens
2. **TRMToolCallingDataset** - PyTorch Dataset for loading samples
3. **TRMCollator** - Batch collation with padding

## Pipeline Flow

```
JSONL File
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  TRMToolCallingDataset                                  │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 1. Load samples from JSONL                        │  │
│  │ 2. Build tool name → ID mapping                   │  │
│  └───────────────────────────────────────────────────┘  │
│                         │                               │
│                         ▼                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │ __getitem__(idx)                                  │  │
│  │ ┌─────────────────────────────────────────────┐   │  │
│  │ │ TRMTokenizer.encode_conversation()          │   │  │
│  │ │ - Add role tokens                           │   │  │
│  │ │ - Encode text with SentencePiece            │   │  │
│  │ │ - Track role IDs per position               │   │  │
│  │ └─────────────────────────────────────────────┘   │  │
│  │                      │                            │  │
│  │                      ▼                            │  │
│  │ ┌─────────────────────────────────────────────┐   │  │
│  │ │ Build labels                                │   │  │
│  │ │ - decision_label (0 or 1)                   │   │  │
│  │ │ - tool_name_label (tool ID or -1)           │   │  │
│  │ └─────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  TRMCollator                                            │
│  - Pad sequences to max length in batch                 │
│  - Create attention masks                               │
│  - Stack all tensors                                    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
                   Batch Tensors
```

## Tokenizer

### Special Tokens

| Token | Purpose |
|-------|---------|
| `<pad>` | Padding token (ID: 0) |
| `<unk>` | Unknown token (ID: 1) |
| `<bos>` | Beginning of sequence (ID: 2) |
| `<eos>` | End of sequence (ID: 3) |
| `<user>` | User turn marker |
| `<assistant>` | Assistant turn marker |
| `<system>` | System message marker |
| `<tool_call>` | Start of tool call |
| `</tool_call>` | End of tool call |
| `<tool_response>` | Start of tool response |
| `</tool_response>` | End of tool response |
| `<tool_name>` | Tool name marker |
| `<tool_args>` | Tool arguments marker |
| `<slot>` | Slot value marker |

### Role IDs

Role IDs are used for role embeddings in the model:

| Role | ID |
|------|-----|
| `user` | 0 |
| `assistant` | 1 |
| `system` | 1 |
| `tool_call` | 2 |
| `tool_response` | 3 |

### Training a Tokenizer

```python
from trm_agent.data import TRMTokenizer

# Train new tokenizer
tokenizer = TRMTokenizer.train(
    input_files=["data/corpus.txt"],
    output_path="tokenizer/trm",
    vocab_size=32000,
    character_coverage=0.9995,
    model_type="unigram",
)
```

### Loading a Tokenizer

```python
from trm_agent.data import TRMTokenizer

tokenizer = TRMTokenizer("tokenizer/trm.model")
```

## Conversation Encoding

### Input Format

```python
history = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, I need help."},
    {"role": "assistant", "content": "How can I help you?"},
    {"role": "user", "content": "Check my bill."},
    {"role": "tool_call", "content": {"name": "get_bill", "arguments": {"phone": "0901234567"}}},
    {"role": "tool_response", "content": {"status": "success", "amount": 150000}},
]
```

### Encoding Process

```python
encoded = tokenizer.encode_conversation(history, max_length=2048)
# Returns:
# {
#     "input_ids": [2, 101, 45, 67, ..., 3],      # Token IDs
#     "role_ids": [0, 1, 1, 1, ..., 0],           # Role ID per position
#     "attention_mask": [1, 1, 1, 1, ..., 1],     # Attention mask
# }
```

### Token Sequence Structure

```
<bos> <system> System message tokens... <user> User message tokens... <assistant> Response tokens... <eos>
  │      │              │                  │            │                  │              │           │
  │      │              │                  │            │                  │              │           │
role=0 role=1        role=1             role=0       role=0            role=1         role=1      role=0
```

### Example Encoding

For a simple conversation:

```python
history = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there"},
]
```

Produces:

```
Tokens:    <bos>  <user>  Hello  <assistant>  Hi  there  <eos>
Token IDs:   2     104     567      105      234   456     3
Role IDs:    0      0       0        1        1     1      0
Attn Mask:   1      1       1        1        1     1      1
```

## Dataset

### Loading Dataset

```python
from trm_agent.data import TRMToolCallingDataset, TRMTokenizer

tokenizer = TRMTokenizer("tokenizer/trm.model")

dataset = TRMToolCallingDataset(
    data_path="data/train.jsonl",
    tokenizer=tokenizer,
    max_seq_len=2048,
    slot_fields=["address", "phone", "name"],  # Optional
)
```

### Sample Structure

Each sample from `dataset[idx]` returns:

```python
{
    "input_ids": torch.Tensor,           # [seq_len] Token IDs
    "attention_mask": torch.Tensor,      # [seq_len] Attention mask (1 = valid, 0 = pad)
    "role_ids": torch.Tensor,            # [seq_len] Role ID per position
    "decision_label": torch.Tensor,      # Scalar: 0 (direct_answer) or 1 (tool_call)
    "tool_name_label": torch.Tensor,     # Scalar: Tool ID or -1 if not tool_call
}
```

> **Note:** Slot extraction is handled by GLiNER2, not TRM dataset.

### Sample Building Process

```
JSONL Sample:
{
    "history": [...],
    "decision": "tool_call",
    "tool": {"name": "get_info", "arguments": {...}},
    "slots": {"phone": "0901234567", "name": "John"}
}
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│ Step 1: Encode conversation history                     │
│                                                         │
│ history = sample["history"]                             │
│ encoded = tokenizer.encode_conversation(history)        │
│                                                         │
│ input_ids = [2, 104, 567, 105, 234, ..., 3]            │
│ role_ids  = [0,   0,   0,   1,   1, ..., 0]            │
│ attn_mask = [1,   1,   1,   1,   1, ..., 1]            │
└─────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: Create decision label                           │
│                                                         │
│ decision = sample["decision"]  # "tool_call"            │
│ decision_label = 1  # (0 for "direct_answer")           │
└─────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: Create tool name label                          │
│                                                         │
│ tool_name = sample["tool"]["name"]  # "get_info"        │
│ tool_name_label = tool_name_to_id["get_info"]  # e.g. 3 │
│                                                         │
│ If decision is "direct_answer":                         │
│     tool_name_label = -1  # Ignored in loss             │
└─────────────────────────────────────────────────────────┘
                │
                ▼
            Final Sample

Note: The "slots" field in JSONL is preserved for GLiNER2 usage but
not processed by TRMToolCallingDataset. Slot extraction uses GLiNER2.
```

### Tool Name Mapping

The dataset automatically builds a mapping from tool names to IDs:

```python
# Auto-built from samples
dataset.tool_name_to_id = {
    "get_customer_info": 0,
    "check_bill": 1,
    "create_ticket": 2,
    ...
}

# Get list of tool names
tool_names = dataset.get_tool_names()

# Get label statistics
stats = dataset.get_label_statistics()
# {
#     "total_samples": 10000,
#     "tool_call": 6000,
#     "direct_answer": 4000,
#     "tool_call_ratio": 0.6,
#     "num_tools": 15,
#     "tool_names": ["get_customer_info", "check_bill", ...]
# }
```

## Batch Collation

### Collator

```python
from trm_agent.data import TRMCollator

collator = TRMCollator(
    pad_token_id=0,
    max_seq_len=2048,
)
```

### Collation Process

```
Sample 1: seq_len = 50     Sample 2: seq_len = 80     Sample 3: seq_len = 65
    │                           │                           │
    └───────────────────────────┼───────────────────────────┘
                                │
                                ▼
                        max_len = 80 (max in batch)
                                │
                                ▼
┌─────────────────────────────────────────────────────────┐
│ Pad all sequences to max_len                            │
│                                                         │
│ Sample 1: [tokens...] + [PAD] * 30                      │
│ Sample 2: [tokens...]                                   │
│ Sample 3: [tokens...] + [PAD] * 15                      │
│                                                         │
│ attention_mask:                                         │
│ Sample 1: [1,1,1,...,1,0,0,...,0]                       │
│ Sample 2: [1,1,1,...,1]                                 │
│ Sample 3: [1,1,1,...,1,0,0,...,0]                       │
└─────────────────────────────────────────────────────────┘
                                │
                                ▼
                          Batch Tensors
```

### Batch Output

```python
batch = {
    "input_ids": torch.Tensor,           # [batch_size, max_len]
    "attention_mask": torch.Tensor,      # [batch_size, max_len]
    "role_ids": torch.Tensor,            # [batch_size, max_len]
    "decision_labels": torch.Tensor,     # [batch_size]
    "tool_name_labels": torch.Tensor,    # [batch_size]
}
```

## DataLoader

### Creating DataLoader

```python
from trm_agent.data import TRMToolCallingDataset, TRMTokenizer
from trm_agent.data.collator import create_dataloader

tokenizer = TRMTokenizer("tokenizer/trm.model")

dataset = TRMToolCallingDataset(
    data_path="data/train.jsonl",
    tokenizer=tokenizer,
    max_seq_len=2048,
)

dataloader = create_dataloader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pad_token_id=tokenizer.pad_token_id,
)
```

### Training Loop

```python
for batch in dataloader:
    input_ids = batch["input_ids"]              # [B, L]
    attention_mask = batch["attention_mask"]    # [B, L]
    role_ids = batch["role_ids"]                # [B, L]
    decision_labels = batch["decision_labels"]  # [B]
    tool_name_labels = batch["tool_name_labels"] # [B]

    # Forward pass
    outputs = model(input_ids, attention_mask, role_ids)

    # Compute loss
    loss = loss_fn(outputs, decision_labels, tool_name_labels)
```

## Complete Example

```python
from trm_agent.data import TRMTokenizer, TRMToolCallingDataset, TRMCollator
from trm_agent.data.collator import create_dataloader

# 1. Load tokenizer
tokenizer = TRMTokenizer("tokenizer/trm.model")

# 2. Load dataset
train_dataset = TRMToolCallingDataset(
    data_path="data/train.jsonl",
    tokenizer=tokenizer,
    max_seq_len=2048,
)

# 3. Check statistics
print(f"Samples: {len(train_dataset)}")
print(f"Stats: {train_dataset.get_label_statistics()}")

# 4. Get single sample
sample = train_dataset[0]
print(f"Input shape: {sample['input_ids'].shape}")
print(f"Decision: {sample['decision_label'].item()}")

# 5. Create dataloader
dataloader = create_dataloader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    pad_token_id=tokenizer.pad_token_id,
)

# 6. Iterate batches
for batch in dataloader:
    print(f"Batch input_ids: {batch['input_ids'].shape}")
    print(f"Batch decision_labels: {batch['decision_labels'].shape}")
    break
```

## Memory Optimization

### Lazy Loading

Samples are loaded entirely into memory during initialization. For very large datasets, consider:

1. Using memory-mapped files
2. Implementing lazy loading with caching
3. Using streaming datasets

### Sequence Length

Truncate long sequences to reduce memory:

```python
dataset = TRMToolCallingDataset(
    data_path="data/train.jsonl",
    tokenizer=tokenizer,
    max_seq_len=512,  # Shorter sequences = less memory
)
```

### Batch Size

Adjust batch size based on available GPU memory:

```python
# For 16GB GPU with 2048 seq_len
dataloader = create_dataloader(dataset, batch_size=8)

# For 24GB GPU with 2048 seq_len
dataloader = create_dataloader(dataset, batch_size=16)
```
