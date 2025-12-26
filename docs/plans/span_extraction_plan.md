# Span Extraction for Tool Arguments and Slot Values

**Status**: ✅ COMPLETED

## Overview

Add span extraction capability to TRM model to extract tool argument values and slot values from input context by predicting start/end token positions.

**Approach**: Span Extraction (extractive, not generative)
**Scope**: Tool arguments + slot values only (no content generation)
**Integration**: Final supervision step only

## Current State

The model already has the necessary output heads:
- `ToolHead.arg_start_classifier`, `arg_end_classifier` → `[batch, seq_len, max_tool_args]`
- `SlotsHead.slot_start_classifier`, `slot_end_classifier` → `[batch, seq_len, num_slots]`

But these outputs currently use **dummy losses** (0.0 weight) for DDP gradient flow.

## Implementation Steps

### Step 1: Create Span Utilities
**File**: `trm_agent/utils/span_utils.py` (new)

- `SpanPosition` dataclass with start/end token indices
- `find_span_in_text()` - find character-level span of value in text
- `char_span_to_token_span()` - convert character span to token positions

### Step 2: Enhance Tokenizer
**File**: `trm_agent/data/tokenizer.py`

Add method `encode_with_offsets()`:
- Return token IDs with character offset mapping `[(char_start, char_end), ...]`
- Needed to map extracted values to token positions

### Step 3: Update Dataset
**File**: `trm_agent/data/dataset.py`

Modify `TRMToolCallingDataset`:
- Build argument name → index mapping per tool
- Compute span labels in `__getitem__`:
  - `slot_start_labels`: `[num_slots]` - start token position per slot (-1 if not found)
  - `slot_end_labels`: `[num_slots]` - end token position per slot
  - `arg_start_labels`: `[max_tool_args]` - start position per argument
  - `arg_end_labels`: `[max_tool_args]` - end position per argument
- Use `find_span_in_text()` + `char_span_to_token_span()` to locate values

### Step 4: Update Collator
**File**: `trm_agent/data/collator.py`

Add batching for new labels:
```python
"slot_start_labels": [batch, num_slots]
"slot_end_labels": [batch, num_slots]
"arg_start_labels": [batch, max_tool_args]
"arg_end_labels": [batch, max_tool_args]
```

### Step 5: Add Span Extraction Loss
**File**: `trm_agent/training/losses.py`

Create `SpanExtractionLoss` class:
```python
class SpanExtractionLoss(nn.Module):
    # CrossEntropyLoss with ignore_index=-1
    # Transpose logits: [batch, seq_len, num_spans] → [batch, num_spans, seq_len]
    # Compute start_loss + end_loss, average
```

Update `TRMLoss`:
- Add `self.slot_span_loss` and `self.arg_span_loss`
- Add losses to total (with configurable weights)
- Remove dummy losses for span outputs

### Step 6: Update Config
**File**: `trm_agent/models/config.py`

Add new parameters:
```python
slot_span_loss_weight: float = 0.5
arg_span_loss_weight: float = 0.5
```

### Step 7: Update DeepSupervisionLoss
**File**: `trm_agent/training/losses.py`

Pass new labels through `DeepSupervisionLoss.forward()` to `TRMLoss`.

### Step 8: Add Evaluation Metrics
**File**: `trm_agent/training/trainer.py`

Add `_compute_span_metrics()`:
- Exact match accuracy (both start and end correct)
- Start accuracy, end accuracy
- Only evaluate where labels exist (not -1)

Update `evaluate()`:
- Compute slot span exact match
- Compute arg span exact match (tool_call samples only)
- Log metrics in evaluation tables

### Step 9: Create Span Decoder (Inference)
**File**: `trm_agent/inference/span_decoder.py` (new)

```python
def decode_span_predictions(
    start_logits, end_logits, input_ids, tokenizer
) -> list[ExtractedSpan]:
    # Argmax start/end positions
    # Decode token IDs to text
    # Return list of (value, start, end, confidence)
```

## Key Technical Details

### Span Label Format
- Position labels are per-slot/per-arg, NOT per-token
- Shape: `[batch, num_slots]` or `[batch, max_tool_args]`
- Value: Token index (0 to seq_len-1) or -1 if not found

### Loss Computation
- Transpose logits from `[batch, seq_len, num_spans]` to `[batch, num_spans, seq_len]`
- Each span becomes independent classification over sequence positions
- Use `ignore_index=-1` to skip missing spans

### Handling Edge Cases
- Value not in context → label = -1, ignored in loss
- Value appears multiple times → use first occurrence
- Semantic slots (e.g., `intent_of_user`) → may not be extractable, label = -1

## Files to Modify

| File | Changes |
|------|---------|
| `trm_agent/utils/span_utils.py` | NEW - span finding utilities |
| `trm_agent/data/tokenizer.py` | Add `encode_with_offsets()` |
| `trm_agent/data/dataset.py` | Add span label computation |
| `trm_agent/data/collator.py` | Batch new labels |
| `trm_agent/training/losses.py` | Add `SpanExtractionLoss`, update `TRMLoss` |
| `trm_agent/models/config.py` | Add loss weight configs |
| `trm_agent/training/trainer.py` | Add span metrics |
| `trm_agent/inference/span_decoder.py` | NEW - inference decoding |
| `trm_agent/utils/__init__.py` | Export new utilities |

## Verification

1. Check span labels are computed correctly for sample data
2. Verify loss decreases during training
3. Check span exact match metrics improve
4. Test inference decoder produces correct text values
