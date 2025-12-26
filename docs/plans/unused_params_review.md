# Review Unused Parameters in InputEmbedding

**Status**: PENDING
**Priority**: Low
**Created**: 2024-12-26

## Overview

The `InputEmbedding` class has unused parameters that were likely intended for future features. This plan outlines what needs to be examined and decisions to be made.

## Current State

### InputEmbedding.forward() signature:
```python
def forward(
    self,
    input_ids: torch.Tensor,
    role_ids: Optional[torch.Tensor] = None,
    tool_ids: Optional[torch.Tensor] = None,      # UNUSED
    attention_mask: Optional[torch.Tensor] = None, # UNUSED (but passed through)
) -> torch.Tensor:
```

### Unused Components:

| Component | Location | Status |
|-----------|----------|--------|
| `tool_ids` parameter | `embeddings.py:104` | Never passed, always None |
| `ToolEmbedding` class | `embeddings.py:60-73` | Initialized but never called |
| `attention_mask` in embedding | `embeddings.py:105` | Passed but not used in embedding |

## Questions to Answer

### 1. Tool Embedding Purpose
- Should tool definitions influence the input embeddings?
- Options:
  - **A) Remove**: Tools don't need to be embedded at input level
  - **B) Append**: Add tool embeddings as prefix/suffix to sequence
  - **C) Context**: Pool tool embeddings and add as global context

### 2. Attention Mask in Embedding
- Is there a use case for masking during embedding creation?
- Currently attention_mask is used in attention layers, not embedding
- Likely safe to remove from InputEmbedding signature

## Potential Implementations

### Option A: Remove Unused Code
```python
# InputEmbedding.forward() - simplified
def forward(
    self,
    input_ids: torch.Tensor,
    role_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
```

Remove:
- `tool_ids` parameter
- `attention_mask` parameter from InputEmbedding
- `ToolEmbedding` class (if not used elsewhere)

### Option B: Implement Tool Context
```python
def forward(
    self,
    input_ids: torch.Tensor,
    role_ids: Optional[torch.Tensor] = None,
    tool_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    embeddings = self.token_embedding(input_ids)

    if role_ids is not None:
        embeddings = embeddings + self.role_embedding(role_ids)

    # Add tool context as global bias
    if tool_ids is not None:
        tool_emb = self.tool_embedding(tool_ids)  # [batch, num_tools, hidden]
        tool_context = tool_emb.mean(dim=1, keepdim=True)  # [batch, 1, hidden]
        embeddings = embeddings + tool_context

    embeddings = embeddings + self.input_bias
    embeddings = self.layer_norm(embeddings)
    embeddings = self.dropout(embeddings)

    return embeddings
```

### Option C: Tool-Conditioned Embedding (Advanced)
- Use tool embeddings to modulate token embeddings
- Similar to conditional generation in diffusion models
- More complex but potentially more expressive

## Files to Modify

| File | Changes |
|------|---------|
| `trm_agent/models/embeddings.py` | Remove/implement tool_ids usage |
| `trm_agent/models/trm.py` | Update forward signatures |
| `trm_agent/data/dataset.py` | Add tool_ids to batch if needed |
| `trm_agent/data/collator.py` | Collate tool_ids if needed |

## Decision Criteria

1. **Does tool context improve performance?**
   - Run ablation: with vs without tool embeddings
   - Measure tool accuracy difference

2. **Is the complexity worth it?**
   - Current model already has tool logits head
   - Tool info is in the input text already

3. **Memory/compute cost?**
   - Additional embedding lookup per forward pass
   - May not be significant

## Action Items

- [ ] Decide: Remove or implement tool_ids
- [ ] If remove: Clean up dead code
- [ ] If implement: Run ablation experiments
- [ ] Update documentation accordingly

## Notes

- The model currently predicts tools via `ToolHead` which uses the final hidden states
- Tool information is encoded in the input text (tool definitions in conversation)
- Adding explicit tool embeddings might be redundant or helpful - needs testing
