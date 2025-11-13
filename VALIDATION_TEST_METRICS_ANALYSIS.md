# Validation and Testing Metrics Computation & Logging Analysis

## Overview
This document provides a comprehensive analysis of how validation and testing metrics are computed, aggregated, and logged in the duo research codebase. The system uses PyTorch Lightning as the training framework with Weights & Biases (W&B) for experiment tracking.

---

## 1. Validation/Test Loop Flow

### Entry Points
- **Training Flow**: `main.py` → `_train()` → `trainer.fit(model, train_ds, valid_ds, ckpt_path)`
- **Evaluation Only**: `main.py` → `_eval_ppl()` → `trainer.validate(model, valid_ds)`  
- **Testing**: After training completes, `trainer.test(model, test_ds)` is called

### PyTorch Lightning Lifecycle
The validation and test loops follow this sequence:

**Validation:**
```
on_validation_epoch_start() → validation_step() [per batch] → on_validation_epoch_end()
```

**Testing:**
```
on_test_epoch_start() → test_step() [per batch] → on_test_epoch_end()
```

---

## 2. Metrics Computation Location

### 2.1 Core Metrics Tracking (`trainer_base.py`)

#### Class: `TrainerBase` (extends `L.LightningModule`)

**Metrics Object Initialization** (lines 86-89):
```python
self.metrics = metrics.Metrics(
    gen_ppl_eval_model_name_or_path=self.config.eval.gen_ppl_eval_model_name_or_path,
    eval_ppl_batch_size=self.config.eval.perplexity_batch_size,
)
```

#### Lifecycle Hooks:

**`on_validation_epoch_start()` (line 180-184)**
- Resets all metrics: `self.metrics.reset()`
- Switches to eval mode: `self._eval_mode()`
- Verifies metrics are clean (nll = 0, weight = 0)

**`validation_step()` (line 186-297)**
This is called for each batch in the validation set:

1. **Loss Computation (lines 193-200)**:
   ```python
   losses = self._loss(
       x0=batch["input_ids"],
       valid_tokens=batch["attention_mask"],
       do_not_mask=batch["do_not_mask"],
       train_mode=False,
       ground_truth_masking=self.config.training.ground_truth_masking,
   )
   ```
   - Returns: Loss object with `nlls`, `prior_loss`, `num_tokens`
   
2. **Update Metrics (line 200)**:
   ```python
   self.metrics.update_valid(losses.nlls, losses.prior_loss, losses.num_tokens)
   ```
   - Updates running average of NLL metrics

3. **Accuracy Computation (lines 202-297)**:
   - **Extract Prompts & Targets** (lines 204-206):
     ```python
     prompts, targets = self._extract_prompts_and_targets(
         batch["input_ids"], batch["do_not_mask"]
     )
     ```
     Source: `_extract_prompts_and_targets()` (lines 299-315)
     - Splits sequences using `do_not_mask` boolean mask
     - Prompts: original tokens where `do_not_mask=True`, masked elsewhere
     - Targets: original tokens where `do_not_mask=False`, padded elsewhere
   
   - **Conditional Generation** (lines 217-221):
     ```python
     generated = self.generate_conditioned(
         prompts, targets, mode=gen_mode, top_k=top_k
     )
     ```
     - Different modes per algorithm:
       - **AR (autoregressive)**: Greedy token-by-token generation (algo.py lines 9-58)
       - **LT (looping transformer)**: Single-pass masked token prediction (algo.py lines 125-150)
       - **MDLM (masked diffusion)**: Multiple unmasking strategies
         - `random`: Random masking strategy
         - `top_k`: Top-k confidence positions
         - `one_at_a_time`: Sequential left-to-right
         - `one_level`: Hierarchical level-by-level (uses `|` delimiters)
         - `all_at_once`: Single-step generation
   
   - **Accuracy Metrics Calculation** (lines 224-226):
     ```python
     acc_exact, acc_token, correct_prediction = self._compute_accuracy(
         generated, targets
     )
     ```
     Source: `_compute_accuracy()` (lines 317-357)
     
     Three accuracy metrics:
     1. **`acc_exact`**: Percentage of sequences perfectly correct
        - Compares all tokens in target_mask
        - All target tokens must match
     2. **`acc_token`**: Token-level accuracy
        - Count: (correctly predicted tokens) / (total target tokens)
        - Applies `target_mask` to count only prediction targets
     3. **`correct_prediction`**: Last token accuracy
        - Checks prediction at the last non-padding token
        - Used to evaluate final answer correctness

4. **Logging Metrics (lines 227-262)**:
   - Logs per generation mode:
     ```python
     self.log(f"val/{gen_mode}_acc_exact", acc_exact, on_step=False, on_epoch=True, sync_dist=True)
     self.log(f"val/{gen_mode}_acc_token", acc_token, on_step=False, on_epoch=True, sync_dist=True)
     self.log(f"val/{gen_mode}_correct_prediction", correct_prediction, on_step=False, on_epoch=True, sync_dist=True)
     ```
   - Special handling for MDLM (lines 250-253): Maps `top_k` results to `default` metrics
   - Summary metrics for standard methods (lines 255-262)

5. **Sample Logging (lines 265-296)**:
   - Logs top N samples to W&B table: `conditioned_generation@global_step{}`
   - Only runs on rank 0 and if logger has `log_table()` method

**`on_validation_epoch_end()` (lines 367-448)**

1. **Compute and Log Aggregated Metrics** (lines 368-371):
   ```python
   for k, v in self.metrics.valid_nlls.items():
       self.log(name=k, value=v.compute(), on_step=False, on_epoch=True, sync_dist=True)
   ```
   - Computes final aggregated values for:
     - `val/nll`: Negative log-likelihood
     - `val/bpd`: Bits per dimension
     - `val/ppl`: Perplexity

2. **Save to JSON File** (lines 373-403):
   ```python
   val_metrics_file = os.path.join(
       self.config.checkpointing.save_dir, "validation_metrics.json"
   )
   current_metrics = {
       k: v.compute().item() for k, v in self.metrics.valid_nlls.items()
   }
   current_metrics["epoch"] = self.current_epoch
   current_metrics["global_step"] = self.global_step
   ```
   - Appends to cumulative JSON file (creates new or appends to existing)
   - Includes epoch and global step for tracking progress
   - Also saves callback metrics (accuracy, generation mode metrics, etc.)

3. **Resume Training** (line 448):
   ```python
   self._train_mode()
   ```

#### Test Step (lines 456-567)

**`test_step()` (lines 456-567)**
- **Nearly identical to `validation_step()`** with key differences:
  - Uses same loss computation and accuracy metrics
  - Logs to `test/*` namespace instead of `val/*`
  - No intermediate resumption to training mode

**`on_test_epoch_end()` (lines 569-601)**

1. **Compute and Log Test Metrics** (lines 570-577):
   ```python
   for k, v in self.metrics.valid_nlls.items():
       self.log(
           name="test/" + k,
           value=v.compute(),
           on_step=False,
           on_epoch=True,
           sync_dist=True,
       )
   ```

2. **Save Test Metrics to JSON** (lines 579-599):
   ```python
   test_metrics_file = os.path.join(
       self.config.checkpointing.save_dir, "test_metrics.json"
   )
   current_metrics = {
       "test/" + k: v.compute().item() for k, v in self.metrics.valid_nlls.items()
   }
   ```
   - Saves to single JSON file (overwrites, not append)
   - Includes epoch and global_step
   - Also saves callback metrics

3. **Resume Training** (line 601):
   ```python
   self._train_mode()
   ```

---

## 3. Metrics Object & Aggregation (`metrics.py`)

### Class: `Metrics`

**Initialization** (lines 68-85):
```python
self.valid_nlls = metrics.clone(prefix='val/')
# Contains:
# - 'val/nll': NLL metric
# - 'val/bpd': Bits per dimension  
# - 'val/ppl': Perplexity
```

**Metric Classes:**

1. **`NLL` (extends `MeanMetric`)** (lines 13-44)
   - Tracks mean value and weight for proper averaging
   - `update(value, weight)`: Adds weighted values
   - Computes: `mean_value / weight`

2. **`BPD`** (lines 47-54)
   - Computes bits per dimension: `nll / log(2)`
   - For evaluating compression efficiency

3. **`Perplexity`** (lines 57-64)
   - Computes: `exp(nll / num_tokens)`
   - Standard language modeling metric

**Update Flow**:
```python
losses = self._loss(...)  # Returns Loss object
self.metrics.update_valid(losses.nlls, losses.prior_loss, losses.num_tokens)
```

In `update_valid()` (lines 107-109):
```python
self.valid_nlls.update(nll, num_tokens)  # Weight by token count
```

---

## 4. Available Information About Sequence Lengths

### Data Flow

**Batch Structure** (from `masked_formal_collator.py`):
```python
batch = {
    "input_ids": torch.Tensor,           # Shape: (batch_size, max_length)
    "attention_mask": torch.Tensor,      # Shape: (batch_size, max_length)
    "do_not_mask": torch.Tensor,         # Boolean mask for prompt region
}
```

**Sequence Length Availability** (lines 59-64 of `masked_formal_collator.py`):
```python
is_not_padding = attention_mask.bool()
seq_lengths = is_not_padding.sum(dim=1)  # Shape: (batch_size,)
# seq_lengths[i] = number of non-padding tokens in sequence i
```

### Where Sequence Information Could Be Used

1. **In `validation_step()`/`test_step()`**:
   - Access via: `batch["attention_mask"].sum(dim=1)` → per-sample sequence lengths
   - Or: `batch["input_ids"]` shape = `(batch_size, fixed_seq_length)`
   - Fixed sequence length from config: `self.config.model.length` = 256 (typically)

2. **In `_extract_prompts_and_targets()`** (lines 299-315):
   - `do_not_mask` provides exact prompt/target split per sample
   - Target region length = `(~do_not_mask & (input_ids != pad_token_id)).sum(dim=1)`

3. **In `_compute_accuracy()`** (lines 317-357):
   - `target_mask` is computed: `targets != self.tokenizer.pad_token_id`
   - Contains per-token information (could extract per-sequence stats)

### Current Limitations
- **Sequence length metrics are NOT currently logged or saved**
- No per-sample sequence length tracking in validation/test
- Metrics are aggregated across all samples without stratification by length
- This prevents analysis of length-dependent generalization

---

## 5. Logging to Weights & Biases (W&B)

### Configuration
From `config.yaml` (lines 75-85):
```yaml
wandb:
  project: mdm-expressivity
  notes: Computational expressivity of MDMs
  group: null
  job_type: null
  name: ${now:%Y%m%d_%H%M%S}
  id: ${.name}_${seed}
  tags:
    - ${noise.type}
    - ${data.language}
    - ${algo.name}
```

### Logging Mechanism

**Step Logging** (lines 165-171 in `trainer_base.py`):
```python
self.log(
    name="trainer/loss",
    value=losses.loss.item(),
    on_step=True,      # Log every step
    on_epoch=False,
    sync_dist=True,
)
```

**Epoch Logging** (lines 227, 241, etc.):
```python
self.log(
    f"val/{gen_mode}_acc_exact",
    acc_exact,
    on_step=False,     # Don't log individual steps
    on_epoch=True,     # Aggregate per epoch
    sync_dist=True,    # Synchronize across distributed devices
)
```

**Table Logging** (lines 289-296):
```python
self.trainer.logger.log_table(
    key=f"conditioned_generation@global_step{self.global_step}",
    columns=[f"Generated {_gen_mode}" for _gen_mode in all_generated_samples] + ["Target"],
    data=[s + [t] for s, t in zip(_all_generated_samples, target_samples)],
)
```
- Logs sample tables with generation mode comparison
- Only on rank 0 (distributed training safe)
- Key includes global step for tracking progression

### W&B Logged Metrics

**Training Metrics**:
- `trainer/loss`: Loss per step
- `trainer/lr`: Learning rate

**Validation Metrics** (logged per epoch):
- `val/nll`: Negative log-likelihood
- `val/bpd`: Bits per dimension
- `val/ppl`: Perplexity
- `val/{gen_mode}_acc_exact`: Exact match accuracy per generation mode
- `val/{gen_mode}_acc_token`: Token-level accuracy per generation mode
- `val/{gen_mode}_correct_prediction`: Final token accuracy per generation mode
- `val/default_acc_exact`, etc.: Default generation metrics (mapped from top_k for MDLM)

**Test Metrics** (logged on test epoch end):
- `test/nll`, `test/bpd`, `test/ppl`: Same as validation
- `test/{gen_mode}_acc_exact`, etc.: Same as validation

**Sample Tables**:
- `conditioned_generation@global_step{N}`: Prompt → Generated vs Target comparison
- `test_conditioned_generation@global_step{N}`: Same for test set

---

## 6. File-Based Metrics Persistence

### JSON Files Structure

#### `validation_metrics.json` (lines 373-403)
**Location**: `${checkpointing.save_dir}/validation_metrics.json`

**Format**: Array of epoch objects
```json
[
  {
    "val/nll": 2.345,
    "val/bpd": 3.382,
    "val/ppl": 10.441,
    "val/default_acc_exact": 0.87,
    "val/default_acc_token": 0.92,
    "val/default_correct_prediction": 0.95,
    "val/random_acc_exact": 0.85,
    "val/random_acc_token": 0.90,
    "val/top_k_acc_exact": 0.88,
    "val/top_k_acc_token": 0.93,
    "val/one_at_a_time_acc_exact": 0.84,
    "val/one_at_a_time_acc_token": 0.89,
    "val/one_level_acc_exact": 0.86,
    "val/one_level_acc_token": 0.91,
    "val/all_at_once_acc_exact": 0.80,
    "val/all_at_once_acc_token": 0.88,
    "epoch": 0,
    "global_step": 25
  },
  {
    "val/nll": 2.123,
    ...
    "epoch": 1,
    "global_step": 50
  }
]
```

**Appending Logic** (lines 394-403):
```python
if os.path.exists(val_metrics_file):
    with open(val_metrics_file, "r") as f:
        all_metrics = json.load(f)
else:
    all_metrics = []
all_metrics.append(current_metrics)
with open(val_metrics_file, "w") as f:
    json.dump(all_metrics, f, indent=4)
```

#### `test_metrics.json` (lines 579-599)
**Location**: `${checkpointing.save_dir}/test_metrics.json`

**Format**: Single object (overwrites each test run)
```json
{
  "test/nll": 2.156,
  "test/bpd": 3.112,
  "test/ppl": 8.634,
  "test/default_acc_exact": 0.89,
  "test/default_acc_token": 0.94,
  "test/default_correct_prediction": 0.97,
  "test/random_acc_exact": 0.87,
  "test/random_acc_token": 0.92,
  "test/top_k_acc_exact": 0.90,
  "test/top_k_acc_token": 0.95,
  "test/one_at_a_time_acc_exact": 0.86,
  "test/one_at_a_time_acc_token": 0.91,
  "test/one_level_acc_exact": 0.88,
  "test/one_level_acc_token": 0.93,
  "test/all_at_once_acc_exact": 0.82,
  "test/all_at_once_acc_token": 0.90,
  "epoch": 4,
  "global_step": 1000
}
```

---

## 7. Analysis & Reporting (`analyze_metrics.py`)

### Functions for Processing Saved Metrics

**`load_validation_metrics()`** (lines 30-40):
- Loads `validation_metrics.json`
- Returns pandas DataFrame with columns: `[val/nll, val/bpd, val/ppl, ..., epoch, global_step]`
- One row per epoch

**`load_test_metrics()`** (lines 43-53):
- Loads `test_metrics.json`
- Returns dictionary

**`plot_validation_metrics()`** (lines 56-102):
- Creates subplots for each metric column
- X-axis: epoch, Y-axis: metric value
- Saves to: `validation_metrics_plot.png`

**`create_validation_summary_table()`** (lines 105-135):
- Computes summary statistics (min, max, mean, std)
- Identifies best values per metric
- Saves to: `validation_summary.csv`

**`plot_test_metrics_comparison()`** (lines 138-189):
- Bar chart comparing test metrics
- Saves to: `test_metrics_plot.png`

**`create_publication_table()`** (lines 192-253):
- Formats test results for publication
- Generates LaTeX table: `test_results_table.tex`
- Outputs formatted console table

### Report Generation Workflow

**Intermediate Report** (lines 256-281):
```bash
python analyze_metrics.py --metrics_dir path/to/output --report_type intermediate
```
- Used during training to track progress
- Generates plots and summaries

**Final Report** (lines 284-310):
```bash
python analyze_metrics.py --metrics_dir path/to/output --report_type final
```
- Used after training completes
- Generates publication-ready results

---

## 8. Key Data Flow Diagram

```
Validation/Test Data Batch
    ↓
[input_ids, attention_mask, do_not_mask]
    ↓
trainer_base._loss()
    ↓ Returns: Loss(loss, nlls, prior_loss, num_tokens)
    ↓
metrics.update_valid(nlls, prior_loss, num_tokens)
    ├→ metrics.valid_nlls → NLL/BPD/Perplexity updated
    └→ Running sums accumulated
    ↓
_extract_prompts_and_targets() using do_not_mask
    ↓ Returns: (prompts, targets)
    ↓
generate_conditioned(prompts, targets, mode=...)
    ├→ AR: Autoregressive greedy generation
    ├→ LT: Single-pass looping transformer
    └→ MDLM: Multiple unmasking strategies
    ↓ Returns: generated tokens
    ↓
_compute_accuracy(generated, targets)
    ├→ acc_exact: % of perfect sequences
    ├→ acc_token: token-level accuracy
    └→ correct_prediction: final token accuracy
    ↓
self.log(..., on_step=False, on_epoch=True)
    ├→ W&B logging (accumulated per epoch)
    └→ PyTorch Lightning aggregation
    ↓
[End of all validation batches]
    ↓
on_validation_epoch_end()
    ├→ Compute aggregated metrics: metrics.valid_nlls.compute()
    ├→ self.log(val/*, value) → W&B
    ├→ Save to validation_metrics.json (append)
    ├→ Callback metrics also saved
    └→ self._train_mode()
```

---

## 9. Sequence Length Information Available

### Where to Access:

1. **Per-Sample Sequence Lengths**:
   ```python
   seq_lengths = (batch["attention_mask"] == 1).sum(dim=1)  # Shape: (batch_size,)
   ```

2. **Prompt Length Per Sample**:
   ```python
   prompt_lengths = batch["do_not_mask"].sum(dim=1)  # Shape: (batch_size,)
   ```

3. **Target Length Per Sample**:
   ```python
   target_lengths = (~batch["do_not_mask"] & (batch["input_ids"] != pad_token_id)).sum(dim=1)
   ```

4. **Fixed Model Length**:
   ```python
   model_length = self.config.model.length  # e.g., 256
   ```

### Current Gap:
- **NOT currently being logged or saved**
- Metrics are aggregated without stratification by length
- Cannot analyze generalization curves (accuracy vs. sequence length)

### To Enable Length-Stratified Analysis:
1. Modify `validation_step()` / `test_step()` to compute and save sequence length info
2. Create length bins (e.g., "length_0-64", "length_64-128", etc.)
3. Log accuracy metrics per bin: `val/acc_token_length_0_64`, etc.
4. Extend JSON saving to include per-bin statistics

---

## 10. Summary Table: Metrics Computation Pipeline

| Component | Purpose | Location | Output |
|-----------|---------|----------|--------|
| `_loss()` | Compute NLL per batch | trainer_base.py:650-704 | Loss object |
| `update_valid()` | Aggregate metrics | metrics.py:107-109 | Internal state updated |
| `_extract_prompts_and_targets()` | Split prompt/target | trainer_base.py:299-315 | Tensor pair |
| `generate_conditioned()` | Decode completions | algo.py (per algorithm) | Generated tokens |
| `_compute_accuracy()` | Calculate 3 accuracy metrics | trainer_base.py:317-357 | 3 float values |
| `self.log()` | W&B + Lightning logging | trainer_base.py:227+ | W&B dashboard |
| `on_validation_epoch_end()` | Aggregate & save | trainer_base.py:367-448 | JSON + W&B |
| `analyze_metrics.py` | Post-hoc analysis | analyze_metrics.py | Plots + tables |

---

## 11. Configuration Reference

**Key Config Parameters** (`config.yaml`):
- `trainer.val_check_interval`: 0.5 (validate every half epoch)
- `trainer.num_sanity_val_steps`: 2 (sanity check batches)
- `loader.eval_batch_size`: 2048 (validation batch size)
- `eval.top_k`: 4 (for MDLM top-k generation)
- `sampling.num_sample_log`: 4 (samples logged per epoch)
- `checkpointing.save_dir`: Output directory path

