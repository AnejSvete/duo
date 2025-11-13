# Length-Stratified Metrics

This document describes the length-stratified metrics feature that tracks model performance across different sequence lengths during validation and testing.

## Overview

The length-stratified metrics system automatically:
1. **Bins sequences** by length into percentile-based groups (default: 4 bins/quartiles)
2. **Tracks all accuracy metrics** separately for each length bin
3. **Logs to W&B** with detailed per-bin metrics
4. **Saves to JSON** files with complete stratification data

This allows you to analyze how model performance varies with sequence length, which is critical for understanding:
- Generalization to longer sequences
- Length-dependent failure modes
- Performance degradation patterns

## What Gets Tracked

### Metrics Stratified by Length

For each generation mode (e.g., `default`, `top_k`, `random`, etc.), the following metrics are tracked per length bin:

- **`acc_exact`**: Exact sequence match accuracy
- **`acc_token`**: Token-level accuracy
- **`correct_prediction`**: Final answer accuracy

### Additional Information

For each length bin, the system also tracks:
- Number of samples in the bin
- Min/max length boundaries
- Bin label (e.g., `len_20-35`)

## W&B Logging

Length-stratified metrics are automatically logged to W&B with the following naming convention:

```
{prefix}/{mode}/{metric}_{bin_label}
{prefix}/{mode}/{metric}_overall
{prefix}/{mode}/num_samples_{bin_label}
```

### Examples

**Validation metrics for default generation mode:**
```
val/default/acc_exact_len_20-35
val/default/acc_exact_len_35-45
val/default/acc_exact_len_45-55
...
val/default/acc_exact_overall
val/default/num_samples_len_20-35
...
```

**Test metrics for MDLM with top_k generation:**
```
test/top_k/acc_exact_len_20-35
test/top_k/acc_token_len_20-35
test/top_k/correct_prediction_len_20-35
...
```

## JSON Output Format

Length-stratified metrics are saved to `validation_metrics.json` and `test_metrics.json` under the `length_stratified` key:

```json
{
  "epoch": 10,
  "global_step": 5000,
  "val/nll": 0.234,
  "val/default_acc_exact": 0.85,
  "length_stratified": {
    "default": {
      "overall": {
        "acc_exact": 0.85,
        "acc_token": 0.92,
        "correct_prediction": 0.87
      },
      "bins": [
        {
          "label": "len_20-35",
          "min_length": 20.0,
          "max_length": 35.0,
          "num_samples": 45,
          "acc_exact": 0.93,
          "acc_token": 0.96,
          "correct_prediction": 0.94
        },
        {
          "label": "len_35-45",
          "min_length": 35.0,
          "max_length": 45.0,
          "num_samples": 52,
          "acc_exact": 0.88,
          "acc_token": 0.93,
          "correct_prediction": 0.90
        },
        ...
      ],
      "bin_edges": [20.0, 35.0, 45.0, 55.0, ...],
      "num_samples": 500
    },
    "top_k": {
      ...
    }
  }
}
```

## Configuration

### Number of Bins

By default, sequences are divided into **4 bins** (quartiles). To change this, modify the initialization in [trainer_base.py](trainer_base.py):

```python
# Default: 4 bins (quartiles)
self.val_length_metrics = PerGenerationModeMetrics(num_bins=4)
self.test_length_metrics = PerGenerationModeMetrics(num_bins=4)

# Example: 5 bins (quintiles)
self.val_length_metrics = PerGenerationModeMetrics(num_bins=5)
self.test_length_metrics = PerGenerationModeMetrics(num_bins=5)

# Example: 10 bins (deciles)
self.val_length_metrics = PerGenerationModeMetrics(num_bins=10)
self.test_length_metrics = PerGenerationModeMetrics(num_bins=10)
```

### Binning Strategy

The system uses **percentile-based binning** by default, which ensures roughly equal numbers of samples per bin. Bins are computed from the observed length distribution in the validation/test set.

For a dataset with lengths ranging from 20 to 100, with 4 bins (quartiles):
- Bins are created at the 0th, 25th, 50th, 75th, and 100th percentiles
- Each bin contains approximately 25% of the samples
- Bin edges adapt to the actual length distribution

## Implementation Details

### Core Components

1. **[length_stratified_metrics.py](length_stratified_metrics.py)**: Core implementation
   - `LengthStratifiedMetrics`: Tracks metrics for a single generation mode
   - `PerGenerationModeMetrics`: Manages multiple generation modes

2. **[trainer_base.py](trainer_base.py)**: Integration points
   - Initialization: Creates metrics trackers ([line 93-94](trainer_base.py#L93-L94))
   - Validation: Updates metrics in `validation_step()` ([line 244-252](trainer_base.py#L244-L252))
   - Testing: Updates metrics in `test_step()` ([line 587-595](trainer_base.py#L587-L595))
   - Aggregation: Computes and logs in epoch end hooks

### How Length is Computed

Sequence length is defined as the **number of non-padding tokens in the target sequence**:

```python
target_mask = targets != tokenizer.pad_token_id
seq_lengths = target_mask.sum(dim=1)  # (batch_size,)
```

This ensures:
- Only the completion region is counted (not the prompt)
- Padding tokens don't inflate lengths
- Length reflects the actual prediction difficulty

### Per-Sample Metrics

To enable accurate length stratification, metrics are computed **per sample** rather than per batch:

```python
def _compute_accuracy_per_sample(self, generated, targets):
    """Returns tensors of shape (batch_size,) with per-sample metrics."""
    # Exact match: 1.0 or 0.0 per sequence
    acc_exact_per_sample = is_correct_or_ignored.all(dim=1).float()

    # Token accuracy: ratio of correct tokens per sequence
    acc_token_per_sample = num_correct_per_sample / num_target_per_sample

    # Prediction accuracy: 1.0 or 0.0 for final answer
    correct_prediction_per_sample = (preds == targets).float()

    return acc_exact_per_sample, acc_token_per_sample, correct_prediction_per_sample
```

This allows each sequence to be assigned to its appropriate length bin with its individual metric values.

## Usage in Analysis

### Analyzing Results with Python

```python
import json
import matplotlib.pyplot as plt

# Load metrics
with open("outputs/my_experiment/validation_metrics.json") as f:
    metrics = json.load(f)

# Get latest epoch's length-stratified data
latest = metrics[-1]["length_stratified"]["default"]

# Extract bin data
bin_labels = [b["label"] for b in latest["bins"]]
acc_exact = [b["acc_exact"] for b in latest["bins"]]
num_samples = [b["num_samples"] for b in latest["bins"]]

# Plot accuracy vs length
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(range(len(bin_labels)), acc_exact)
plt.xticks(range(len(bin_labels)), bin_labels, rotation=45)
plt.ylabel("Exact Match Accuracy")
plt.title("Accuracy by Sequence Length")

plt.subplot(1, 2, 2)
plt.bar(range(len(bin_labels)), num_samples)
plt.xticks(range(len(bin_labels)), bin_labels, rotation=45)
plt.ylabel("Number of Samples")
plt.title("Length Distribution")

plt.tight_layout()
plt.savefig("length_stratified_analysis.png")
```

### Querying W&B

You can query length-stratified metrics in W&B using the metric names:

```python
import wandb

api = wandb.Api()
run = api.run("username/project/run_id")

# Get all length-stratified metrics
length_metrics = {
    k: v for k, v in run.summary.items()
    if "len_" in k
}

# Get metrics for a specific bin
short_acc = run.summary["val/default/acc_exact_len_20-35"]
long_acc = run.summary["val/default/acc_exact_len_80-100"]

print(f"Short sequences (20-35): {short_acc:.2%}")
print(f"Long sequences (80-100): {long_acc:.2%}")
```

### Analyzing Generalization

Compare performance across length ranges to assess generalization:

```python
# Assuming train lengths were 20-60, test lengths 60-100
train_bins = ["len_20-35", "len_35-45", "len_45-60"]
test_bins = ["len_60-70", "len_70-80", "len_80-100"]

train_acc = [b["acc_exact"] for b in latest["bins"] if b["label"] in train_bins]
test_acc = [b["acc_exact"] for b in latest["bins"] if b["label"] in test_bins]

print(f"In-distribution accuracy: {sum(train_acc)/len(train_acc):.2%}")
print(f"Out-of-distribution accuracy: {sum(test_acc)/len(test_acc):.2%}")
print(f"Generalization gap: {sum(train_acc)/len(train_acc) - sum(test_acc)/len(test_acc):.2%}")
```

## Example Output

When you run validation, you'll see metrics logged like this in W&B:

```
Epoch 10 Metrics:
  val/default_acc_exact: 0.850
  val/default_acc_token: 0.920

  Length-stratified (default):
    val/default/acc_exact_len_20-35: 0.934
    val/default/acc_exact_len_35-45: 0.889
    val/default/acc_exact_len_45-55: 0.856
    val/default/acc_exact_len_55-65: 0.823
    val/default/acc_exact_len_65-75: 0.798
    val/default/acc_exact_len_75-85: 0.761
    val/default/acc_exact_len_85-95: 0.724
    val/default/acc_exact_len_95-105: 0.689
    val/default/acc_exact_len_105-115: 0.651
    val/default/acc_exact_len_115-128: 0.612

    Sample distribution:
    val/default/num_samples_len_20-35: 52
    val/default/num_samples_len_35-45: 48
    ...
```

This shows a clear degradation in performance as sequence length increases, from 93.4% for short sequences (20-35 tokens) down to 61.2% for long sequences (115-128 tokens).

## Testing

A comprehensive test suite is available in [test_length_metrics.py](test_length_metrics.py):

```bash
conda activate duo
python test_length_metrics.py
```

This tests:
1. Basic binning and metric computation
2. Accumulation across multiple batches
3. Tracking multiple generation modes
4. Realistic validation scenarios

All tests should pass with the ✓ indicator.

## Future Enhancements

Possible extensions to this system:

1. **Configurable binning strategies**
   - Uniform bins (equal width)
   - Custom bin edges
   - Logarithmic spacing

2. **Additional stratifications**
   - By tree depth (for BFVP)
   - By operation type (for arithmetic)
   - By prompt length separately from completion length

3. **Automatic alerts**
   - Detect significant performance drops at certain lengths
   - Flag generalization gaps automatically
   - Generate summary statistics in plain text

4. **Interactive visualization**
   - Real-time W&B plots
   - Heatmaps of accuracy vs. length vs. epoch
   - Comparative plots across algorithm variants

## Troubleshooting

### No length-stratified metrics appearing in W&B

**Possible causes:**
1. The dataset doesn't have `do_not_mask` in the batch (required for formal language tasks)
2. All sequences have the same length (can't create meaningful bins)
3. Very small validation set (< num_bins samples)

**Solution:** Check that your task uses the prompt-completion format with `do_not_mask` tensor.

### Bins have very unequal sample counts

**Cause:** Length distribution is highly skewed or discrete.

**Solution:** This is expected with percentile-based binning on skewed data. The bin edges will cluster around common lengths. Consider using fewer bins or custom bin edges if needed.

### Metrics don't match overall values

**Cause:** The per-sample metrics average may differ slightly from batch-averaged metrics due to different aggregation orders.

**Solution:** This is normal. The length-stratified `overall` metrics should be very close to the standard logged metrics. Small differences (< 0.1%) are acceptable due to floating point arithmetic.

## References

- Implementation: [length_stratified_metrics.py](length_stratified_metrics.py)
- Integration: [trainer_base.py](trainer_base.py)
- Tests: [test_length_metrics.py](test_length_metrics.py)
- Main documentation: [CLAUDE.md](CLAUDE.md)
