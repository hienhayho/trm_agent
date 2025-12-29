# GLiNER2 Integration for Entity Extraction

This document describes the GLiNER2 co-network integration for entity extraction (slots and tool arguments).

## Architecture Overview

The system uses a hybrid approach where:
- **TRM**: Handles decision prediction (tool_call/direct_answer) and tool name selection
- **GLiNER2**: Handles entity extraction for both slots and tool arguments

```
┌─────────────────────────────────────────────────────┐
│                  Input Text                          │
│          (conversation history → full_text)          │
└─────────────────────┬───────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌───────────────┐           ┌───────────────┐
│  TRM Model    │           │   GLiNER2     │
│  - decision   │           │  - slots      │
│  - tool_name  │           │  - tool_args  │
└───────┬───────┘           └───────┬───────┘
        │                           │
        └─────────────┬─────────────┘
                      ▼
              Combined Output
```

## Benefits

1. **Better NER accuracy**: GLiNER2 is trained specifically for entity extraction
2. **Zero-shot capability**: Can extract new entity types without retraining
3. **LoRA fine-tuning**: Domain-specific adaptation without modifying base model
4. **Multilingual**: `gliner2-multi-v1` supports Vietnamese
5. **Decoupled**: Can update GLiNER2 independently of TRM

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GLINER2_MODEL` | `fastino/gliner2-multi-v1` | Base GLiNER2 model |
| `GLINER2_ADAPTER` | (empty) | Path to LoRA adapter directory |
| `GLINER2_THRESHOLD` | `0.5` | Confidence threshold for extraction |

## GLiNER2Extractor

### Location
`trm_agent/inference/gliner2_extractor.py`

### Usage

```python
from trm_agent.inference import GLiNER2Extractor

# Using pre-trained model
extractor = GLiNER2Extractor()

# Using fine-tuned LoRA adapter
extractor = GLiNER2Extractor(
    adapter_path="outputs/gliner2/final"
)

# Extract slots and tool arguments
slots, tool_args = extractor.extract_all(
    text="Tôi là Nguyễn Văn A, muốn lắp mạng ở 123 Nguyễn Huệ",
    tool_name="get_product_price",
    tool_params={"get_product_price": ["product", "address"]}
)
```

### Key Methods

| Method | Description |
|--------|-------------|
| `extract(text, labels)` | Extract entities for given labels |
| `extract_slots(text)` | Extract only slot fields |
| `extract_all(text, tool_name, tool_params)` | Extract both slots and tool args |
| `load_adapter(path)` | Load a LoRA adapter |
| `unload_adapter()` | Unload the current adapter |

## LoRA Fine-tuning

### Training

The training script supports:
- TRM JSONL format (auto-converts) and pre-converted GLiNER2 format
- Multiple training files (concatenated)
- Single GPU and multi-GPU (DDP) training via torchrun

```bash
# Single GPU training (auto-converts TRM dataset):
uv run python tools/train_gliner2.py \
    --train-data data/train.jsonl \
    --val-data data/test.jsonl \
    --output-dir outputs/gliner2

# Multiple training files (concatenated):
uv run python tools/train_gliner2.py \
    --train-data data/train1.jsonl data/train2.jsonl data/train3.jsonl \
    --val-data data/test.jsonl \
    --output-dir outputs/gliner2

# Split validation from training data (10% per file):
uv run python tools/train_gliner2.py \
    --train-data data/train1.jsonl data/train2.jsonl \
    --val-split 0.1 \
    --output-dir outputs/gliner2

# Multi-GPU training with torchrun:
torchrun --nproc_per_node=4 tools/train_gliner2.py \
    --train-data data/train.jsonl \
    --val-data data/test.jsonl \
    --output-dir outputs/gliner2
```

The script will:
1. Auto-detect TRM JSONL format and convert to GLiNER2 InputExample format
2. Concatenate multiple training files if provided
3. Log sample examples for verification
4. Apply LoRA adapters to the encoder
5. Train with mixed precision (FP16) by default
6. Save checkpoints and best model based on validation loss

### Multi-GPU Setup

```bash
# 4 GPUs on single node:
torchrun --nproc_per_node=4 tools/train_gliner2.py \
    --train-data data/train.jsonl \
    --output-dir outputs/gliner2

# All available GPUs:
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) tools/train_gliner2.py \
    --train-data data/train.jsonl \
    --output-dir outputs/gliner2

# Multi-node training:
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
    --master_addr=192.168.1.1 --master_port=29500 \
    tools/train_gliner2.py --train-data data/train.jsonl
```

### Manual Conversion (Optional)

If you prefer to pre-convert the dataset:

```bash
uv run python tools/convert_to_gliner2.py data/train.jsonl data/gliner2_train.pkl
uv run python tools/convert_to_gliner2.py data/test.jsonl data/gliner2_test.pkl

# Then train with pre-converted files:
uv run python tools/train_gliner2.py \
    --train-data data/gliner2_train.pkl \
    --val-data data/gliner2_test.pkl \
    --output-dir outputs/gliner2
```

#### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--base-model` | `fastino/gliner2-multi-v1` | Base model |
| `--epochs` | 10 | Training epochs |
| `--batch-size` | 8 | Batch size |
| `--val-split` | 0.0 | Split ratio for validation from training data (per-file) |
| `--lora-r` | 8 | LoRA rank (4, 8, 16, 32) |
| `--lora-alpha` | 16.0 | LoRA alpha (usually 2*r) |
| `--lora-dropout` | 0.0 | LoRA dropout |
| `--encoder-lr` | 1e-5 | Encoder learning rate |
| `--task-lr` | 5e-4 | Task layer learning rate |

### Using Trained Adapter

```bash
GLINER2_ADAPTER=outputs/gliner2/final uv run chainlit run app.py
```

## Slot and Tool Param Extraction

### Slot Fields (Always Extracted)

- `address`
- `phone`
- `device_number`
- `intent_of_user`
- `name`
- `contract_id`

### Tool Param Fields (Tool-specific)

Tool parameters are extracted based on the predicted tool. The mapping is built from `tools.json`:

```python
tool_param_mapping = {
    "get_product_price": ["product", "address"],
    "describe_product": ["product"],
    "request_agent": ["reason"],
}
```

### Extraction Logic

```python
# When decision is "tool_call":
# - Extract all slot fields
# - Extract tool-specific param fields

# When decision is "direct_answer":
# - Extract all slot fields only
# - No tool param extraction
```

## Integration with App

The Chainlit app integrates GLiNER2 in `predict_with_trm()`:

```python
def predict_with_trm(history):
    # Build full text for GLiNER2
    full_text = conversation_to_text(history)

    # TRM: decision + tool selection
    decision, tool_name = trm_predict(history)

    # GLiNER2: entity extraction
    slots, tool_args = gliner2_extractor.extract_all(
        text=full_text,
        tool_name=tool_name if decision == "tool_call" else None,
        tool_params=tool_param_mapping,
    )

    return decision, tool_name, tool_args, slots
```

## Model Size

| Component | Size |
|-----------|------|
| GLiNER2 base model | ~400MB |
| LoRA adapter | ~10-50MB (depending on rank) |
| Total (with adapter) | ~450MB |

## Performance Considerations

1. **Latency**: Two model inferences (TRM + GLiNER2) per request
2. **Memory**: Additional ~400MB GPU memory for GLiNER2
3. **Caching**: Consider caching for repeated entity types

## Future Improvements

1. **Batched extraction**: Process multiple requests in parallel
2. **Confidence filtering**: Use threshold tuning per entity type
3. **Entity linking**: Connect extracted entities to knowledge base
