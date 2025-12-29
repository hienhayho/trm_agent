# TRM Agent - Project Documentation

## Project Overview

This project implements **TRM (Tiny Recursive Model) aware language model training** for tool-calling and response generation. The goal is to train small, efficient models that can recursively reason about when to call tools versus provide direct answers.

## TRM Concept (from the paper "Less is More: Recursive Reasoning with Tiny Networks")

### Core Idea

TRM is a recursive reasoning approach that achieves high generalization using a **single tiny network** (2 layers, ~7M parameters) that:

1. **Recursively improves latent reasoning** `z` given input `x`, current answer `y`, and current latent `z`
2. **Refines the output answer** `y` given current `y` and latent `z`
3. Uses **deep supervision** to progressively improve answers across multiple supervision steps

### Key Components

```
x = embedded input (question/context)
y = current answer embedding (solution being refined)
z = latent reasoning feature (like chain-of-thought)
```

### Algorithm (Pseudocode)

```python
def latent_recursion(x, y, z, n=6):
    for i in range(n):  # latent reasoning
        z = net(x, y, z)
    y = net(y, z)  # refine output answer
    return y, z

def deep_recursion(x, y, z, n=6, T=3):
    # Recurse T-1 times without gradients
    with torch.no_grad():
        for j in range(T-1):
            y, z = latent_recursion(x, y, z, n)
    # Final recursion with gradients
    y, z = latent_recursion(x, y, z, n)
    return (y.detach(), z.detach()), output_head(y), Q_head(y)
```

### Why TRM Works

1. **Less is more**: 2-layer networks prevent overfitting on small datasets
2. **Deep supervision**: Model learns to improve any (y, z) through recursion
3. **Separate y and z**:
   - `y` stores the current solution
   - `z` stores reasoning state (prevents forgetting how we got to `y`)
4. **No fixed-point theorem needed**: Full recursion with backprop, no 1-step gradient approximation

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| n | 6 | Latent recursion steps |
| T | 3 | Deep recursion iterations |
| N_sup | 16 | Max supervision steps |
| Layers | 2 | Network depth |
| Hidden size | 512 | Embedding dimension |
| EMA | 0.999 | Exponential moving average |

## Application to Tool-Calling

### Decision Framework

For tool-calling, the TRM approach maps to:

- **x**: Embedded conversation history + available tools
- **y**: Current decision (tool_call vs direct_answer) + content/tool info
- **z**: Latent reasoning about slot filling, prerequisites, and decision logic

### Training Objective

The model learns to:
1. **Classify decision**: `tool_call` or `direct_answer`
2. **Extract slots**: Accumulate information from conversation
3. **Generate tool calls**: Name + arguments when decision is `tool_call`
4. **Generate responses**: Content when decision is `direct_answer`

## Dataset Format

### Target Format (JSONL)

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "tool_name",
        "description": "Tool description",
        "parameters": {
          "type": "object",
          "properties": {...},
          "required": [...]
        }
      }
    }
  ],
  "history": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "tool_call", "content": {"name": "...", "arguments": {...}}},
    {"role": "tool_response", "content": {...}}
  ],
  "decision": "tool_call|direct_answer",
  "slots": {
    "address": "...",
    "phone": "...",
    "device_number": "...",
    "intent_of_user": "...",
    "name": "...",
    "contract_id": "..."
  },
  "tool": {
    "name": "tool_name",
    "arguments": {"arg1": "value1"}
  },
  "content": "Response text if direct_answer"
}
```

### Raw Data Format

```json
[
  {"role": "user", "content": "User message"},
  {
    "role": "assistant",
    "content": {
      "think": "Reasoning trace...",
      "response": "Assistant response text",
      "address": "", "phone": "", ...
    }
  },
  {
    "role": "tool_call",
    "content": {
      "think": "Reasoning trace...",
      "response": {"name": "tool_name", "arguments": {...}},
      "address": "...", "phone": "", ...
    }
  },
  {
    "role": "tool_response",
    "content": {
      "think": "Reasoning trace...",
      "response": {"info": "Tool result..."},
      "address": "...", "phone": "", ...
    }
  }
]
```

### Roles in History

| Role | Description |
|------|-------------|
| `user` | User messages |
| `assistant` | Direct responses from the model |
| `tool_call` | Tool invocation with name and arguments |
| `tool_response` | Result returned from tool execution |

## Project Structure

```
trm_agent/
├── pyproject.toml           # Project configuration (uv)
├── CLAUDE.md                 # This file
├── trm_agent/
│   ├── __init__.py
│   └── utils/
│       ├── __init__.py
│       └── data_processor.py # Core data processing utilities
├── tools/
│   └── convert_raw_to_dataset.py  # CLI conversion script
└── tests/
    └── sample_data/          # Test data
```

## Usage

### Convert Raw Data to Training Format

```bash
uv run python tools/convert_raw_to_dataset.py <input_folder> <output_file> [options]

# Options:
#   --tools <file>       JSON file with tool definitions
#   --slot-fields <...>  Custom slot field names
#   --recursive          Search subdirectories
```

### Example

```bash
uv run python tools/convert_raw_to_dataset.py \
    data/raw \
    data/dataset.jsonl \
    --tools data/tools.json
```

### Programmatic Usage

```python
from trm_agent.utils import process_raw_conversation, RawDataProcessor

# Simple conversion
samples = process_raw_conversation(
    raw_data=conversation_list,
    tools=tool_definitions,
    slot_fields=["address", "phone", "name"]
)

# Using processor class
processor = RawDataProcessor(tools=tools, slot_fields=slot_fields)
samples = processor.process_conversation(raw_data)
```

## Development Guidelines

### Code Style

- Use Python 3.11+
- Type hints for all functions
- Dataclasses for structured data
- Keep functions focused and small

### Testing

```bash
uv run python tools/convert_raw_to_dataset.py tests/sample_data tests/output.jsonl
```

### Adding New Slot Fields

1. Update `DEFAULT_SLOT_FIELDS` in `trm_agent/utils/data_processor.py`
2. Or pass custom fields via `--slot-fields` CLI argument

## Future Work

1. **TRM Model Implementation**: Implement the actual TRM architecture for training
2. **Content Generation**: Extend beyond tool-calling to content generation
3. **Multi-turn Reasoning**: Implement recursive improvement across conversation turns
4. **Evaluation Pipeline**: Add benchmarks for tool-calling accuracy

## References

- Paper: "Less is More: Recursive Reasoning with Tiny Networks" (Jolicoeur-Martineau, 2025)
- arXiv: 2510.04871v1
- Key insight: Small recursive models (7M params) can outperform LLMs on reasoning tasks

## IMPORTANT NOTES:
- Always use uv for running python script, uv add for installing packages
- Use pathlib for checking file/folder exists, mkdir, ...
- Use trm_agent import instead of using sys.path.insert
- With args, always have prefix: --
- Don't add: #!/usr/bin/env python3 at the begining of python script files.
- Example usage must be used with: uv run ...
- When implemented or modified something, always check docs folder to if need to update any document or not.
- use builtin logger of trm-agent for logging in DDP environment.