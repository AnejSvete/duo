# Metrics Computation Quick Reference

## Key Files
- **`trainer_base.py`**: Main metrics computation and logging
- **`metrics.py`**: Metric aggregation logic
- **`analyze_metrics.py`**: Post-hoc analysis and visualization
- **`config.yaml`**: Configuration (validation interval, batch sizes, etc.)

## Critical Functions

### Loss & Metrics Update
```python
# trainer_base.py:193-200
losses = self._loss(x0, valid_tokens, do_not_mask, train_mode=False)
self.metrics.update_valid(losses.nlls, losses.prior_loss, losses.num_tokens)
```
**Path to Metrics Object**: `self.metrics.valid_nlls` contains NLL, BPD, Perplexity

### Accuracy Computation
```python
# trainer_base.py:204-226
prompts, targets = self._extract_prompts_and_targets(batch["input_ids"], batch["do_not_mask"])
generated = self.generate_conditioned(prompts, targets, mode=gen_mode, top_k=top_k)
acc_exact, acc_token, correct_prediction = self._compute_accuracy(generated, targets)
```

**Three Accuracy Metrics**:
1. `acc_exact`: % of perfectly correct sequences (all target tokens match)
2. `acc_token`: Token-level accuracy (correct_tokens / total_target_tokens)
3. `correct_prediction`: Last token accuracy (accuracy at final prediction position)

### Metrics Persistence
```python
# trainer_base.py:374-403 (validation)
val_metrics_file = os.path.join(self.config.checkpointing.save_dir, "validation_metrics.json")
# Appends JSON for each epoch
```

```python
# trainer_base.py:580-599 (test)
test_metrics_file = os.path.join(self.config.checkpointing.save_dir, "test_metrics.json")
# Overwrites with single JSON object
```

## Metrics Saved to W&B

**Per Validation Epoch**:
- `val/nll`, `val/bpd`, `val/ppl` (aggregated loss metrics)
- `val/{mode}_acc_exact`, `val/{mode}_acc_token`, `val/{mode}_correct_prediction` (per generation mode)
- Conditional generation samples table: `conditioned_generation@global_step{N}`

**At Test Time**:
- `test/nll`, `test/bpd`, `test/ppl` (same as validation)
- `test/{mode}_acc_exact`, etc. (per generation mode)
- Test samples table: `test_conditioned_generation@global_step{N}`

## JSON File Structures

### validation_metrics.json (Cumulative, Append-Only)
```json
[
  {
    "val/nll": 2.345,
    "val/bpd": 3.382,
    "val/ppl": 10.441,
    "val/default_acc_exact": 0.87,
    "val/default_acc_token": 0.92,
    "epoch": 0,
    "global_step": 25
  },
  { ... }  // One entry per validation epoch
]
```

### test_metrics.json (Single Object, Overwrite)
```json
{
  "test/nll": 2.156,
  "test/bpd": 3.112,
  "test/ppl": 8.634,
  "test/default_acc_exact": 0.89,
  "test/default_acc_token": 0.94,
  "epoch": 4,
  "global_step": 1000
}
```

## Batch Information Available

### In validation_step() / test_step()
```python
batch["input_ids"]        # Shape: (batch_size, model.length)
batch["attention_mask"]   # Shape: (batch_size, model.length), 1 for real tokens
batch["do_not_mask"]      # Shape: (batch_size, model.length), boolean mask for prompt region
```

### Computing Sequence Information
```python
# Per-sample sequence lengths (non-padding tokens)
seq_lengths = (batch["attention_mask"] == 1).sum(dim=1)

# Prompt lengths (protected from masking)
prompt_lengths = batch["do_not_mask"].sum(dim=1)

# Target lengths (to be predicted)
target_lengths = (~batch["do_not_mask"] & 
                  (batch["input_ids"] != tokenizer.pad_token_id)).sum(dim=1)
```

## Generation Modes by Algorithm

### AR (Autoregressive)
- Only mode: `default`
- Greedy decoding, token-by-token

### LT (Looping Transformer)
- Only mode: `default`
- Single-pass non-autoregressive generation

### MDLM (Masked Diffusion)
- `random`: Random masking schedule
- `top_k`: Unmask top-k confidence positions
- `one_level`: Unmask one hierarchical level per step
- `all_at_once`: Single-step generation
- `one_at_a_time`: Sequential left-to-right

Default metric for MDLM is mapped from `top_k` mode.

## Important Configuration Values
```yaml
# config.yaml
trainer.val_check_interval: 0.5        # Validate every half epoch
trainer.num_sanity_val_steps: 2        # Sanity check batches
loader.eval_batch_size: 2048          # Validation batch size
eval.top_k: 4                          # Top-k for MDLM
sampling.num_sample_log: 4             # Samples logged per epoch
checkpointing.save_dir: ${cwd:}        # Where metrics are saved
```

## Data Flow Summary

```
Validation Batch
    ↓
_loss() → Loss object (nlls, num_tokens)
    ↓
metrics.update_valid() → Running aggregation
    ↓
_extract_prompts_and_targets() → (prompts, targets)
    ↓
generate_conditioned() → generated tokens
    ↓
_compute_accuracy() → (acc_exact, acc_token, correct_prediction)
    ↓
self.log() → W&B
    ↓
[All batches processed]
    ↓
on_validation_epoch_end()
    ├→ metrics.valid_nlls.compute() → final aggregated metrics
    ├→ self.log() → W&B logging
    └→ Save to validation_metrics.json (append)
```

## Accessing Metrics Programmatically

### During Training
```python
# In validation_step or test_step
current_nlls = self.metrics.valid_nlls["nll"].compute()
current_ppl = self.metrics.valid_nlls["ppl"].compute()
```

### From Saved Files
```python
import json
import pandas as pd

# Load validation metrics
with open("validation_metrics.json") as f:
    val_metrics = json.load(f)
df = pd.DataFrame(val_metrics)

# Load test metrics
with open("test_metrics.json") as f:
    test_metrics = json.load(f)
```

### Using analyze_metrics.py
```bash
# Generate intermediate report during training
python analyze_metrics.py --metrics_dir path/to/output --report_type intermediate

# Generate final report after training
python analyze_metrics.py --metrics_dir path/to/output --report_type final
```

## Missing Features / Gap

### Sequence Length Metrics Not Currently Logged
- No per-length-bin accuracy metrics
- Cannot analyze generalization across sequence lengths
- All metrics aggregated globally across all sequences

To add length-stratified analysis:
1. Compute `target_lengths` in validation_step/test_step
2. Bin sequences by length range
3. Log separate accuracy per bin: `val/acc_token_length_0_64`, etc.
4. Extend JSON to include per-bin statistics

