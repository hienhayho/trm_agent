# Content Generation Module for Direct Answer

**Status:** PENDING
**Priority:** Medium
**Created:** 2024-12-26

## Overview

Add content generation capability for `direct_answer` responses using teacher-forcing during training and single-pass greedy decode at inference.

**Approach:** Causal Language Model with teacher forcing (not autoregressive at inference)

**Key Insight:** The `ContentHead` already exists in `heads.py:204-227` but is disabled. We need to enable it and connect the data/training pipeline.

## Current State

| Component | Status |
|-----------|--------|
| `ContentHead` class | Exists in `heads.py:204-227` but disabled |
| JSONL `content` field | Exists but ignored in dataset |
| `content_ids` in batch | Not implemented |
| Content loss | Not implemented |
| `TRMOutput.content_logits` | Not included |

## Implementation Steps

### Step 1: Update Config
**File:** `trm_agent/models/config.py`

Add parameters:
```python
max_content_len: int = 256
content_loss_weight: float = 1.0
```

### Step 2: Update Dataset
**File:** `trm_agent/data/dataset.py`

In `__getitem__`:
- Extract `content` field for `direct_answer` samples
- Tokenize with BOS/EOS tokens
- Pad/truncate to `max_content_len`
- Return `content_ids` tensor (padding for `tool_call` samples)

```python
# For direct_answer:
content_ids = tokenizer.encode(content, add_bos=True, add_eos=True)
# For tool_call:
content_ids = [pad_token_id] * max_content_len
```

### Step 3: Update Collator
**File:** `trm_agent/data/collator.py`

Add `content_ids` to batch output with padding.

### Step 4: Enable ContentHead in OutputHead
**File:** `trm_agent/models/heads.py`

In `OutputHead.forward()`:
- Call `self.content_head(y)`
- Include `content_logits` in return dict

### Step 5: Update TRMOutput
**File:** `trm_agent/models/trm.py`

Add to dataclass:
```python
content_logits: Optional[torch.Tensor] = None  # [batch, seq_len, vocab_size]
```

Update forward to include content_logits in output.

### Step 6: Add Content Loss
**File:** `trm_agent/training/losses.py`

In `TRMLoss`:
```python
# CrossEntropyLoss with ignore_index for padding
self.content_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)

# In forward():
# Only compute for direct_answer samples (decision_labels == 0)
content_mask = decision_labels == 0
if content_mask.any() and outputs.content_logits is not None:
    # Shift for next-token prediction
    shift_logits = outputs.content_logits[content_mask, :-1, :]
    shift_labels = content_ids[content_mask, 1:]
    content_loss = self.content_loss(
        shift_logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1)
    )
```

Update `DeepSupervisionLoss` to pass `content_ids`.

### Step 7: Update Trainer
**File:** `trm_agent/training/trainer.py`

- Pass `content_ids` to model forward
- Pass `content_ids` to loss function
- Add content loss to logging
- Add perplexity metric for evaluation

### Step 8: Update Inference Decoder
**File:** `trm_agent/inference/span_decoder.py`

For `direct_answer` decisions:
- Greedy decode `content_logits.argmax(dim=-1)`
- Decode tokens to text
- Return in result dict

## Files to Modify

| File | Changes |
|------|---------|
| `trm_agent/models/config.py` | Add `max_content_len`, `content_loss_weight` |
| `trm_agent/data/dataset.py` | Extract and tokenize content, return `content_ids` |
| `trm_agent/data/collator.py` | Batch `content_ids` |
| `trm_agent/models/heads.py` | Enable ContentHead in OutputHead.forward() |
| `trm_agent/models/trm.py` | Add `content_logits` to TRMOutput |
| `trm_agent/training/losses.py` | Add content CrossEntropyLoss |
| `trm_agent/training/trainer.py` | Pass content_ids, add metrics |
| `trm_agent/inference/span_decoder.py` | Decode content for direct_answer |

## Loss Formula

```
Total Loss = w_d * L_decision
           + w_t * L_tool
           + w_s * L_slot
           + w_q * L_q
           + w_ss * L_slot_span
           + w_as * L_arg_span
           + w_c * L_content  # NEW
```

Where `L_content` is CrossEntropyLoss on shifted tokens, masked to `direct_answer` samples only.

## Evaluation Metrics

- **Content Perplexity**: `exp(content_loss)` - lower is better
- Optional: BLEU score against ground truth (future)

## Design Decisions

### Why Teacher-Forcing (not Autoregressive)?

1. **Simpler implementation**: Single forward pass at inference
2. **TRM's recursive refinement**: Model already iteratively improves `y` embedding
3. **Faster inference**: No token-by-token generation loop
4. **Can upgrade later**: If quality is poor, autoregressive can be added

### Why CrossEntropyLoss?

- Standard for language modeling
- `ignore_index` handles padding naturally
- Per-token loss allows fine-grained learning

### Masking Strategy

- `tool_call` samples: `content_ids = [PAD] * max_len`, ignored in loss
- `direct_answer` samples: Actual content tokens, included in loss

## Future Enhancements

If single-pass decode quality is poor:

1. **Autoregressive generation**:
   - Create `ContentGenerator` class with token-by-token loop
   - Use temperature/sampling for diversity
   - Add beam search option

2. **Quality improvements**:
   - Add repetition penalty
   - Implement nucleus sampling (top-p)
   - Length penalty for balanced outputs

## Action Items

- [ ] Update config with content parameters
- [ ] Add content_ids extraction to dataset
- [ ] Update collator for content batching
- [ ] Enable ContentHead in OutputHead
- [ ] Add content_logits to TRMOutput
- [ ] Implement content loss in TRMLoss
- [ ] Update trainer for content training
- [ ] Add content decoding to inference
- [ ] Add perplexity metric to evaluation
- [ ] Update documentation
