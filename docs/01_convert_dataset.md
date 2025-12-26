# Dataset Conversion

This document describes the process of converting raw conversation data to TRM training format.

## Overview

The conversion script transforms raw conversation JSON files into a JSONL dataset suitable for training the TRM model. Each conversation is split into multiple training samples at decision points (assistant responses or tool calls).

## Usage

```bash
uv run python tools/convert_raw_to_dataset.py --input <folder> --output <file> [options]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--input` | Yes | Path to folder containing raw JSON conversation files |
| `--output` | Yes | Path to output JSONL file |
| `--tools` | No | Path to JSON file containing tool definitions |
| `--system` | No | Path to system prompt text file (prepended to history) |
| `--slot-fields` | No | List of slot field names to extract |
| `--recursive` | No | Search for JSON files recursively in subdirectories |
| `--num-input-files` | No | Limit number of input files (for testing) |

## Examples

### Basic conversion

```bash
uv run python tools/convert_raw_to_dataset.py \
    --input data/raw \
    --output data/dataset.jsonl
```

### With tool definitions

```bash
uv run python tools/convert_raw_to_dataset.py \
    --input data/raw \
    --output data/dataset.jsonl \
    --tools data/tools.json
```

### With system prompt

```bash
uv run python tools/convert_raw_to_dataset.py \
    --input data/raw \
    --output data/dataset.jsonl \
    --system prompts/system.txt
```

### Testing with small dataset

```bash
uv run python tools/convert_raw_to_dataset.py \
    --input data/raw \
    --output data/test.jsonl \
    --num-input-files 10
```

### Custom slot fields

```bash
uv run python tools/convert_raw_to_dataset.py \
    --input data/raw \
    --output data/dataset.jsonl \
    --slot-fields name phone email address
```

### Recursive search

```bash
uv run python tools/convert_raw_to_dataset.py \
    --input data/raw \
    --output data/dataset.jsonl \
    --recursive
```

## Input Format

### Raw Conversation JSON

Each input file should contain a JSON array of conversation turns:

```json
[
  {
    "role": "user",
    "content": "User message text"
  },
  {
    "role": "assistant",
    "content": {
      "think": "Internal reasoning (optional)",
      "response": "Assistant response text",
      "address": "",
      "phone": "",
      "device_number": "",
      "intent_of_user": "",
      "name": "",
      "contract_id": ""
    }
  },
  {
    "role": "tool_call",
    "content": {
      "think": "Internal reasoning (optional)",
      "response": {
        "name": "tool_name",
        "arguments": {"arg1": "value1"}
      },
      "address": "123 Main St",
      "phone": "0901234567",
      "device_number": "",
      "intent_of_user": "check_bill",
      "name": "John Doe",
      "contract_id": ""
    }
  },
  {
    "role": "tool_response",
    "content": {
      "think": "Internal reasoning (optional)",
      "response": {"status": "success", "data": {...}}
    }
  }
]
```

### Roles

| Role | Description |
|------|-------------|
| `user` | User messages |
| `assistant` | Direct responses from the model |
| `tool_call` | Tool invocation with name and arguments |
| `tool_response` | Result returned from tool execution |

### Tool Definitions JSON

```json
[
  {
    "type": "function",
    "function": {
      "name": "get_customer_info",
      "description": "Get customer information by phone number",
      "parameters": {
        "type": "object",
        "properties": {
          "phone": {
            "type": "string",
            "description": "Customer phone number"
          }
        },
        "required": ["phone"]
      }
    }
  }
]
```

### System Prompt Text

Plain text file containing the system prompt:

```
You are a helpful customer service assistant for FPT Telecom.
Your task is to help customers with their inquiries about internet services, billing, and technical support.
```

## Output Format

### JSONL Sample

Each line in the output JSONL file is a training sample:

```json
{
  "tools": [...],
  "history": [
    {"role": "system", "content": "System prompt..."},
    {"role": "user", "content": "User message"},
    {"role": "assistant", "content": "Previous response"},
    {"role": "tool_call", "content": {"name": "...", "arguments": {...}}},
    {"role": "tool_response", "content": {...}}
  ],
  "decision": "tool_call",
  "slots": {
    "address": "123 Main St",
    "phone": "0901234567",
    "device_number": "",
    "intent_of_user": "check_bill",
    "name": "John Doe",
    "contract_id": ""
  },
  "tool": {
    "name": "get_customer_info",
    "arguments": {"phone": "0901234567"}
  },
  "content": ""
}
```

### Output Fields

| Field | Description |
|-------|-------------|
| `tools` | List of available tool definitions |
| `history` | Conversation history up to this decision point |
| `decision` | Either `"tool_call"` or `"direct_answer"` |
| `slots` | Extracted slot values from the conversation |
| `tool` | Tool name and arguments (empty if `direct_answer`) |
| `content` | Response text (empty if `tool_call`) |

## Processing Logic

### Sample Generation

The converter creates one training sample for each decision point:

1. **User messages** - Added to history, no sample generated
2. **Assistant responses** - Creates `direct_answer` sample
3. **Tool calls** - Creates `tool_call` sample
4. **Tool responses** - Added to history, no sample generated

### History Accumulation

History accumulates progressively:

```
Turn 1: user     -> history: [user1]
Turn 2: assistant -> SAMPLE 1 (history: [user1])
                  -> history: [user1, assistant1]
Turn 3: user     -> history: [user1, assistant1, user2]
Turn 4: tool_call -> SAMPLE 2 (history: [user1, assistant1, user2])
                  -> history: [user1, assistant1, user2, tool_call1]
Turn 5: tool_resp -> history: [user1, assistant1, user2, tool_call1, tool_resp1]
Turn 6: assistant -> SAMPLE 3 (history: [user1, assistant1, user2, tool_call1, tool_resp1])
```

### Slot Extraction

Slots are extracted from the `content` object at each decision point. Default slot fields:

- `address`
- `phone`
- `device_number`
- `intent_of_user`
- `name`
- `contract_id`

## Default Values

| Parameter | Default |
|-----------|---------|
| Slot fields | `address`, `phone`, `device_number`, `intent_of_user`, `name`, `contract_id` |
| Recursive | `False` |
| System prompt | None (not prepended) |

## Error Handling

- Invalid JSON files are skipped with error logging
- Files not containing a list are skipped with warning
- Processing continues even if some files fail
- Final summary shows processed files, errors, and total samples
