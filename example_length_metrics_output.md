# Example: Length-Stratified Metrics Output

This document shows what the length-stratified metrics look like in actual output files and W&B logs.

## Example 1: validation_metrics.json

After running validation with the new system, your `validation_metrics.json` will look like this:

```json
[
  {
    "epoch": 0,
    "global_step": 500,
    "val/nll": 1.234,
    "val/bpd": 1.78,
    "val/ppl": 3.435,
    "val/default_acc_exact": 0.850,
    "val/default_acc_token": 0.920,
    "val/default_correct_prediction": 0.875,
    "length_stratified": {
      "default": {
        "overall": {
          "acc_exact": 0.850,
          "acc_token": 0.920,
          "correct_prediction": 0.875
        },
        "bins": [
          {
            "label": "len_20-28",
            "min_length": 20.0,
            "max_length": 28.0,
            "num_samples": 45,
            "acc_exact": 0.933,
            "acc_token": 0.967,
            "correct_prediction": 0.956
          },
          {
            "label": "len_28-35",
            "min_length": 28.0,
            "max_length": 35.0,
            "num_samples": 48,
            "acc_exact": 0.896,
            "acc_token": 0.938,
            "correct_prediction": 0.917
          },
          {
            "label": "len_35-42",
            "min_length": 35.0,
            "max_length": 42.0,
            "num_samples": 52,
            "acc_exact": 0.865,
            "acc_token": 0.923,
            "correct_prediction": 0.885
          },
          {
            "label": "len_42-48",
            "min_length": 42.0,
            "max_length": 48.0,
            "num_samples": 49,
            "acc_exact": 0.857,
            "acc_token": 0.918,
            "correct_prediction": 0.878
          },
          {
            "label": "len_48-55",
            "min_length": 48.0,
            "max_length": 55.0,
            "num_samples": 51,
            "acc_exact": 0.843,
            "acc_token": 0.912,
            "correct_prediction": 0.867
          },
          {
            "label": "len_55-62",
            "min_length": 55.0,
            "max_length": 62.0,
            "num_samples": 47,
            "acc_exact": 0.830,
            "acc_token": 0.905,
            "correct_prediction": 0.851
          },
          {
            "label": "len_62-70",
            "min_length": 62.0,
            "max_length": 70.0,
            "num_samples": 50,
            "acc_exact": 0.820,
            "acc_token": 0.896,
            "correct_prediction": 0.840
          },
          {
            "label": "len_70-78",
            "min_length": 70.0,
            "max_length": 78.0,
            "num_samples": 46,
            "acc_exact": 0.804,
            "acc_token": 0.884,
            "correct_prediction": 0.826
          },
          {
            "label": "len_78-88",
            "min_length": 78.0,
            "max_length": 88.0,
            "num_samples": 53,
            "acc_exact": 0.774,
            "acc_token": 0.867,
            "correct_prediction": 0.811
          },
          {
            "label": "len_88-128",
            "min_length": 88.0,
            "max_length": 128.0,
            "num_samples": 59,
            "acc_exact": 0.729,
            "acc_token": 0.841,
            "correct_prediction": 0.780
          }
        ],
        "bin_edges": [20.0, 28.0, 35.0, 42.0, 48.0, 55.0, 62.0, 70.0, 78.0, 88.0, 128.0],
        "num_samples": 500
      }
    }
  },
  {
    "epoch": 1,
    "global_step": 1000,
    "...": "..."
  }
]
```

## Example 2: MDLM with Multiple Generation Modes

For MDLM models that use multiple generation strategies, you'll see stratified metrics for each mode:

```json
{
  "epoch": 5,
  "global_step": 2500,
  "val/random_acc_exact": 0.721,
  "val/top_k_acc_exact": 0.856,
  "val/one_level_acc_exact": 0.834,
  "val/all_at_once_acc_exact": 0.798,
  "val/one_at_a_time_acc_exact": 0.845,
  "length_stratified": {
    "random": {
      "overall": {
        "acc_exact": 0.721,
        "acc_token": 0.867,
        "correct_prediction": 0.798
      },
      "bins": [
        {"label": "len_20-28", "num_samples": 45, "acc_exact": 0.844, "...": "..."},
        {"label": "len_28-35", "num_samples": 48, "acc_exact": 0.792, "...": "..."},
        "..."
      ]
    },
    "top_k": {
      "overall": {
        "acc_exact": 0.856,
        "acc_token": 0.928,
        "correct_prediction": 0.891
      },
      "bins": [
        {"label": "len_20-28", "num_samples": 45, "acc_exact": 0.956, "...": "..."},
        {"label": "len_28-35", "num_samples": 48, "acc_exact": 0.917, "...": "..."},
        "..."
      ]
    },
    "one_level": {
      "overall": {"acc_exact": 0.834, "...": "..."},
      "bins": ["..."]
    },
    "all_at_once": {
      "overall": {"acc_exact": 0.798, "...": "..."},
      "bins": ["..."]
    },
    "one_at_a_time": {
      "overall": {"acc_exact": 0.845, "...": "..."},
      "bins": ["..."]
    }
  }
}
```

## Example 3: W&B Logged Metrics

In your W&B dashboard, you'll see these metrics logged (example for validation):

### Standard Metrics (already existed)
```
val/nll: 1.234
val/bpd: 1.78
val/ppl: 3.435
val/default_acc_exact: 0.850
val/default_acc_token: 0.920
val/default_correct_prediction: 0.875
```

### New Length-Stratified Metrics

**Overall metrics (for reference):**
```
val/default/acc_exact_overall: 0.850
val/default/acc_token_overall: 0.920
val/default/correct_prediction_overall: 0.875
```

**Per-bin accuracy metrics:**
```
val/default/acc_exact_len_20-28: 0.933
val/default/acc_exact_len_28-35: 0.896
val/default/acc_exact_len_35-42: 0.865
val/default/acc_exact_len_42-48: 0.857
val/default/acc_exact_len_48-55: 0.843
val/default/acc_exact_len_55-62: 0.830
val/default/acc_exact_len_62-70: 0.820
val/default/acc_exact_len_70-78: 0.804
val/default/acc_exact_len_78-88: 0.774
val/default/acc_exact_len_88-128: 0.729

val/default/acc_token_len_20-28: 0.967
val/default/acc_token_len_28-35: 0.938
...

val/default/correct_prediction_len_20-28: 0.956
val/default/correct_prediction_len_28-35: 0.917
...
```

**Per-bin sample counts:**
```
val/default/num_samples_len_20-28: 45.0
val/default/num_samples_len_28-35: 48.0
val/default/num_samples_len_35-42: 52.0
...
```

## Example 4: Creating Visualizations

You can create plots directly from the saved metrics:

```python
import json
import matplotlib.pyplot as plt
import numpy as np

# Load validation metrics
with open("outputs/my_experiment/validation_metrics.json") as f:
    all_metrics = json.load(f)

# Get the latest epoch
latest = all_metrics[-1]
stratified = latest["length_stratified"]["default"]

# Extract data
bins = stratified["bins"]
labels = [b["label"].replace("len_", "") for b in bins]
acc_exact = [b["acc_exact"] for b in bins]
acc_token = [b["acc_token"] for b in bins]
num_samples = [b["num_samples"] for b in bins]

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Accuracy by length
ax = axes[0]
x = np.arange(len(labels))
width = 0.35

bars1 = ax.bar(x - width/2, acc_exact, width, label='Exact Match', alpha=0.8)
bars2 = ax.bar(x + width/2, acc_token, width, label='Token-level', alpha=0.8)

ax.set_xlabel('Sequence Length Bin')
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Across Sequence Lengths')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)

# Plot 2: Sample distribution
ax = axes[1]
bars = ax.bar(x, num_samples, alpha=0.8, color='steelblue')
ax.set_xlabel('Sequence Length Bin')
ax.set_ylabel('Number of Samples')
ax.set_title('Distribution of Sequence Lengths in Validation Set')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('length_stratified_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved plot to length_stratified_analysis.png")
```

## Example 5: Comparing Across Epochs

Track how length-dependent performance changes during training:

```python
import json
import matplotlib.pyplot as plt

with open("outputs/my_experiment/validation_metrics.json") as f:
    all_metrics = json.load(f)

# Focus on two length bins: short (20-35) and long (78-128)
epochs = []
short_acc = []
long_acc = []

for epoch_metrics in all_metrics:
    if "length_stratified" not in epoch_metrics:
        continue

    bins = epoch_metrics["length_stratified"]["default"]["bins"]
    epochs.append(epoch_metrics["epoch"])

    # Find short and long bins
    for b in bins:
        if "20" in b["label"] or "28" in b["label"]:
            short_acc.append(b["acc_exact"])
        elif "78" in b["label"] or "88" in b["label"]:
            long_acc.append(b["acc_exact"])

plt.figure(figsize=(10, 6))
plt.plot(epochs, short_acc, 'o-', label='Short sequences (20-35)', linewidth=2)
plt.plot(epochs, long_acc, 's-', label='Long sequences (78-128)', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Exact Match Accuracy')
plt.title('Training Dynamics: Short vs Long Sequences')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('training_dynamics_by_length.png', dpi=150)
print("✓ Saved training dynamics plot")
```

## Example 6: Test Metrics Format

The `test_metrics.json` file follows the same structure but uses "test/" prefix:

```json
{
  "test/val/nll": 1.456,
  "test/val/bpd": 2.10,
  "test/val/ppl": 4.291,
  "test/default_acc_exact": 0.812,
  "test/default_acc_token": 0.895,
  "test/default_correct_prediction": 0.847,
  "epoch": 10,
  "global_step": 5000,
  "length_stratified": {
    "default": {
      "overall": {
        "acc_exact": 0.812,
        "acc_token": 0.895,
        "correct_prediction": 0.847
      },
      "bins": [
        {
          "label": "len_60-68",
          "min_length": 60.0,
          "max_length": 68.0,
          "num_samples": 42,
          "acc_exact": 0.857,
          "acc_token": 0.921,
          "correct_prediction": 0.881
        },
        {
          "label": "len_68-75",
          "min_length": 68.0,
          "max_length": 75.0,
          "num_samples": 45,
          "acc_exact": 0.844,
          "acc_token": 0.912,
          "correct_prediction": 0.867
        },
        "... (bins for test length range 60-128)"
      ],
      "bin_edges": [60.0, 68.0, 75.0, 82.0, 88.0, 95.0, 101.0, 108.0, 115.0, 122.0, 128.0],
      "num_samples": 500
    }
  }
}
```

## Key Observations from Examples

1. **Gradual Performance Degradation**: Notice how `acc_exact` decreases from 0.933 (len 20-28) to 0.729 (len 88-128) - a clear length-dependent pattern

2. **Consistent Sample Distribution**: With percentile-based binning, each bin has roughly equal samples (45-59 in the examples)

3. **Multiple Metrics Tracked**: All three accuracy types (exact, token, correct_prediction) are stratified by length

4. **Easy to Parse**: The JSON structure is nested but logical: `length_stratified` → `{mode}` → `bins` → individual bin data

5. **W&B Integration**: Every metric gets its own W&B log entry, enabling filtering and custom plots in the dashboard

6. **Generalization Testing**: Different bin edges for validation (20-128) vs test (60-128) allow measuring generalization to longer sequences
