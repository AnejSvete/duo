# Validation & Test Metrics Analysis - Summary

## Documentation Created

Three comprehensive analysis documents have been created to help you understand the validation and testing metrics pipeline in the duo codebase:

1. **VALIDATION_TEST_METRICS_ANALYSIS.md** (605 lines, 19KB)
   - Complete, detailed analysis of all metrics computation
   - Organized by topic with line number references to source code
   - Best for: Deep understanding and comprehensive reference

2. **METRICS_QUICK_REFERENCE.md** (210 lines, 6KB)
   - Quick lookup guide for key functions and metrics
   - Code snippets and configuration values
   - Best for: Fast reference during development

3. **METRICS_ARCHITECTURE.md** (392 lines, 21KB)
   - Visual ASCII diagrams of the entire system
   - Data flow diagrams and process flows
   - Best for: Understanding the big picture and relationships

---

## Key Findings

### 1. Metrics Computation Happens at Two Levels

**Per-Batch Level** (in `validation_step()` / `test_step()`):
- Loss metrics: NLL computed and accumulated
- Accuracy metrics: Three types calculated
  - `acc_exact`: Sequence-level exact match
  - `acc_token`: Token-level accuracy
  - `correct_prediction`: Final answer token accuracy

**Per-Epoch Level** (in `on_validation_epoch_end()` / `on_test_epoch_end()`):
- Loss metrics aggregated from running sums
- All metrics logged to W&B and JSON files

### 2. Three Core Metrics

**Loss Metrics** (trainer_base.py:368-371):
- `val/nll`: Negative Log-Likelihood (raw)
- `val/bpd`: Bits Per Dimension (nll / log(2))
- `val/ppl`: Perplexity (exp(nll / num_tokens))

**Accuracy Metrics** (trainer_base.py:224-226):
- `acc_exact`: % of sequences with all target tokens correct
- `acc_token`: % of target tokens predicted correctly
- `correct_prediction`: % of sequences with correct final answer

### 3. Two Output Channels

**Weights & Biases (W&B)**:
- Real-time monitoring during training
- Scalar metrics per epoch
- Sample tables showing generation results
- Accessible via W&B web dashboard

**JSON Files**:
- `validation_metrics.json`: Cumulative array (append-only)
  - One entry per validation epoch
  - Contains both loss and accuracy metrics
- `test_metrics.json`: Single object (overwrite)
  - Final test results only
  - Contains all metrics from final test run

### 4. Generation Modes

**AR (Autoregressive)**:
- Single mode: `default`
- Greedy token-by-token decoding

**LT (Looping Transformer)**:
- Single mode: `default`
- Single-pass non-autoregressive generation

**MDLM (Masked Diffusion)**:
- 5 modes: `random`, `top_k`, `one_level`, `all_at_once`, `one_at_a_time`
- Each mode evaluated separately
- `top_k` mapped to `default` for comparison with AR/LT

### 5. Sequence Length Information

**Currently Available but NOT Logged**:
```python
seq_lengths = (batch["attention_mask"] == 1).sum(dim=1)
prompt_lengths = batch["do_not_mask"].sum(dim=1)
target_lengths = (~batch["do_not_mask"] & 
                  (batch["input_ids"] != pad_token_id)).sum(dim=1)
```

**Gap**: No length-stratified metrics
- Cannot analyze generalization across sequence lengths
- All metrics aggregated globally
- No per-length-bin accuracy tracking

---

## Critical Code Locations

| Task | File | Lines | Function |
|------|------|-------|----------|
| Loss computation | trainer_base.py | 650-704 | `_loss()` |
| Metrics update | metrics.py | 103-109 | `update_train/update_valid()` |
| Prompt/target split | trainer_base.py | 299-315 | `_extract_prompts_and_targets()` |
| Accuracy calculation | trainer_base.py | 317-357 | `_compute_accuracy()` |
| Per-batch logging | trainer_base.py | 227-262 | `validation_step()` accuracy section |
| Epoch aggregation | trainer_base.py | 367-403 | `on_validation_epoch_end()` |
| JSON persistence | trainer_base.py | 373-403 | JSON save logic |
| AR generation | algo.py | 9-58 | `AR.generate_conditioned()` |
| LT generation | algo.py | 125-150 | `LT.generate_conditioned()` |
| MDLM generation | diffusion.py + algo.py | various | Multiple generation modes |

---

## Data Flow Summary

```
Batch Input:
  batch["input_ids"]      - All tokens
  batch["attention_mask"] - Real tokens mask
  batch["do_not_mask"]    - Prompt protection mask
        ↓
  Loss Computation:
  - _loss() → nlls per token
  - metrics.update_valid(nlls, num_tokens)
        ↓
  Accuracy Computation:
  - Extract prompts/targets using do_not_mask
  - generate_conditioned() → generated tokens
  - _compute_accuracy() → 3 metrics
        ↓
  Per-Batch Logging:
  - self.log() → W&B accumulation
        ↓
  [Repeat for all batches]
        ↓
  Epoch Aggregation:
  - metrics.valid_nlls.compute() → final values
  - self.log() → W&B final logging
  - Save to validation_metrics.json (append)
```

---

## Important Configuration Parameters

```yaml
# trainer (validation intervals)
trainer.val_check_interval: 0.5        # Validate every half epoch
trainer.num_sanity_val_steps: 2        # Sanity check batches

# loader (batch sizes)
loader.eval_batch_size: 2048           # Validation batch size

# eval (generation settings)
eval.top_k: 4                          # Top-k for MDLM generation
eval.gen_ppl_eval_model_name_or_path: gpt2-large  # For PPL eval

# sampling (logging)
sampling.num_sample_log: 4             # Sample count per epoch

# checkpointing (output)
checkpointing.save_dir: ${cwd:}        # Metrics save location
```

---

## Missing Features (Gaps)

### 1. Sequence Length Metrics
- No length-stratified accuracy metrics
- Cannot analyze "accuracy vs. sequence length" curves
- Cannot identify if model struggles with long sequences

### 2. Per-Sample Information
- Metrics only aggregated globally
- No per-sample loss or accuracy tracking
- Cannot identify which types of problems the model struggles with

### 3. Generation Strategy Analysis
- Multiple generation modes logged separately (MDLM)
- No direct comparison mechanism in JSON
- Would benefit from explicit ranking/comparison metrics

### To Add Length-Stratified Analysis:
```python
# In validation_step() / test_step()
target_lengths = (~batch["do_not_mask"] & 
                  (batch["input_ids"] != pad_token_id)).sum(dim=1)

# Create length bins
length_bins = [(0, 64), (64, 128), (128, 256)]

# Log per-bin metrics
for bin_start, bin_end in length_bins:
    mask = (target_lengths >= bin_start) & (target_lengths < bin_end)
    if mask.any():
        acc_token_binned = (
            ((generated == targets) & target_mask)[mask].float().mean()
        )
        self.log(f"val/acc_token_len_{bin_start}_{bin_end}", 
                 acc_token_binned)
```

---

## Quick Navigation Guide

**For Understanding**:
1. Start with METRICS_QUICK_REFERENCE.md (overview)
2. Read METRICS_ARCHITECTURE.md (visual flows)
3. Consult VALIDATION_TEST_METRICS_ANALYSIS.md (deep dive)

**For Implementation**:
1. Check METRICS_QUICK_REFERENCE.md for function locations
2. Use line numbers to find code in source files
3. Refer to METRICS_ARCHITECTURE.md for context

**For Debugging**:
1. Find your issue in METRICS_QUICK_REFERENCE.md
2. Get line numbers for source code
3. Cross-reference with VALIDATION_TEST_METRICS_ANALYSIS.md for detailed explanation

---

## Key Takeaways

1. **Metrics are computed in two phases**: per-batch updates + epoch-end aggregation
2. **Two persistence mechanisms**: W&B (real-time) + JSON (historical)
3. **Sequence length information is available but not logged**
4. **Generation modes are algorithm-dependent**: AR/LT have 1, MDLM has 5
5. **The system is well-structured** but could benefit from length-stratified analysis

---

## Files Generated

- `/Users/anej/repos/projects/duo/VALIDATION_TEST_METRICS_ANALYSIS.md` - Complete reference
- `/Users/anej/repos/projects/duo/METRICS_QUICK_REFERENCE.md` - Quick lookup
- `/Users/anej/repos/projects/duo/METRICS_ARCHITECTURE.md` - Visual diagrams
- `/Users/anej/repos/projects/duo/ANALYSIS_SUMMARY.md` - This file

All files are in the repository root and ready for reference.

