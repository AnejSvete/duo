# MDM Hyperparameter Grid Search Guide

This guide explains how to run hyperparameter grid searches for MDM and analyze the results.

## Quick Start

```bash
# 1. Run recommended grid search on BFVP task
python scripts/grid_search_mdm.py --task bfvp --grid recommended

# 2. Wait for jobs to complete (check with squeue)

# 3. Analyze results
python scripts/analyze_grid_search.py --results_dir grid_search_results
```

## Grid Search Script

### Basic Usage

```bash
# Run on different tasks
python scripts/grid_search_mdm.py --task parity --grid recommended
python scripts/grid_search_mdm.py --task arithmetic --grid recommended

# Use different grid sizes
python scripts/grid_search_mdm.py --task bfvp --grid quick      # Small grid, faster
python scripts/grid_search_mdm.py --task bfvp --grid recommended # Medium grid (default)
python scripts/grid_search_mdm.py --task bfvp --grid full       # Full grid, comprehensive

# Custom output directory
python scripts/grid_search_mdm.py --task bfvp --output_dir my_search_results

# Multiple random seeds per configuration
python scripts/grid_search_mdm.py --task bfvp --num_seeds 5
```

### Grid Sizes

**Quick Grid** (~36 jobs):
- Good for initial exploration or debugging
- Covers basic parameter ranges
- Fast to complete

**Recommended Grid** (~108 jobs):
- **Recommended starting point**
- Focuses on parameter ranges that work well in diffusion literature
- Good balance between coverage and computational cost

**Full Grid** (~864 jobs):
- Comprehensive search over all parameters
- Use when you have plenty of compute
- Best for final tuning

### Hyperparameters Explored

The grid search explores:

1. **Learning Rate** (`optim.lr`)
   - Most critical hyperparameter for diffusion models
   - Range: 1e-3 to 1e-2

2. **Weight Decay** (`optim.weight_decay`)
   - Regularization strength
   - Range: 0.0 to 0.2

3. **Gradient Clipping** (`trainer.gradient_clip_val`)
   - Prevents gradient explosions
   - Range: 1.0 to 20.0

4. **Noise Schedule** (`noise`)
   - How noise is added during training
   - Options: log-linear, linear, cosine

5. **Warmup Steps** (`lr_scheduler.lr_lambda.warmup_steps`)
   - Learning rate warmup period
   - Range: 100 to 1000 steps

6. **Batch Size** (`loader.batch_size`)
   - Affects gradient variance
   - Options: 1024, 2048, 4096

### Advanced Options

```bash
# Dry run (preview commands without submitting)
python scripts/grid_search_mdm.py --task bfvp --dry_run

# Create job scripts without submitting
python scripts/grid_search_mdm.py --task bfvp --create_scripts_only
# Then submit manually:
# for f in grid_search_results/slurm_scripts/*.sh; do sbatch $f; done

# Run jobs sequentially (no SLURM, for local testing)
python scripts/grid_search_mdm.py --task bfvp --grid quick --sequential
```

## Analysis Script

### Basic Usage

```bash
# Analyze results from grid search
python scripts/analyze_grid_search.py --results_dir grid_search_results

# Optimize for different metrics
python scripts/analyze_grid_search.py \
    --results_dir grid_search_results \
    --metric val/acc_token

python scripts/analyze_grid_search.py \
    --results_dir grid_search_results \
    --metric val/loss

# Show more top configurations
python scripts/analyze_grid_search.py \
    --results_dir grid_search_results \
    --top_k 20

# Save plots to custom directory
python scripts/analyze_grid_search.py \
    --results_dir grid_search_results \
    --output_dir my_analysis_results
```

### Output

The analysis script generates:

1. **Console Output**:
   - Hyperparameter importance ranking
   - Top K best configurations
   - Training stability analysis
   - Recommendations for best settings

2. **Plots** (saved to `{results_dir}/analysis/`):
   - `hyperparameter_effects_*.png`: Box plots showing effect of each hyperparameter
   - `learning_curves_top*.png`: Learning curves for best configurations

3. **CSV File**:
   - `grid_search_results.csv`: Complete results table for further analysis

### Interpreting Results

**Hyperparameter Importance**:
- Higher variance score = more important hyperparameter
- Focus tuning efforts on high-importance parameters

**Top Configurations**:
- Shows best hyperparameter combinations
- Use these as starting points for further tuning

**Training Stability**:
- Identifies configurations that led to instabilities
- Helps avoid problematic hyperparameter combinations

**Recommendations**:
- Suggests safe ranges for each hyperparameter
- Warns about settings to avoid

## Checkpoint Recovery Improvements

The auto-recovery callback has been improved to handle training instabilities better:

### What Was Fixed

**Problem**: When the model recovered from a loss explosion, it would reset the optimizer but keep the same high learning rate that caused the explosion.

**Solution**: The recovery callback now:
1. Loads the best checkpoint (model weights)
2. Reduces the learning rate by a configurable factor (default: 0.5x)
3. Resets the optimizer state with the new lower learning rate
4. Continues training from the recovered state

### Configuration

Edit `configs/callbacks/auto_recovery.yaml`:

```yaml
auto_recovery:
  loss_threshold: 5.0  # Lower threshold = more sensitive to explosions
  patience: 3  # Wait 3 bad steps before triggering recovery
  lr_reduction_factor: 0.5  # Reduce LR to 50% after each recovery
  reset_optimizer: true  # Reset optimizer state (recommended)
```

### How It Works

1. **Detection**: Monitors loss after each batch
2. **Patience**: Waits N steps to confirm instability
3. **Recovery**:
   - Loads best checkpoint
   - Reduces LR: `new_lr = old_lr * (0.5 ^ recovery_count)`
   - Resets optimizer with new LR
4. **Continuation**: Training continues from recovered state

### Example

```
Initial LR: 5e-3
First explosion → Recovery 1 → LR: 2.5e-3
Second explosion → Recovery 2 → LR: 1.25e-3
Third explosion → Recovery 3 → LR: 6.25e-4
```

After 3 recoveries, training stops (configurable via `max_recoveries`).

## Tips and Best Practices

### 1. Start Small
```bash
# Test on quick grid first
python scripts/grid_search_mdm.py --task bfvp --grid quick --num_seeds 1
```

### 2. Monitor Jobs
```bash
# Check SLURM queue
squeue -u $USER

# Watch for completions
watch -n 60 'find grid_search_results -name "validation_metrics.json" | wc -l'
```

### 3. Incremental Analysis
```bash
# You can analyze results as jobs complete
# No need to wait for all jobs to finish
python scripts/analyze_grid_search.py --results_dir grid_search_results
```

### 4. Focus Search
Based on initial results, you can create a focused search:
```python
# Edit scripts/grid_search_mdm.py
CUSTOM_GRID = {
    "lr": [3e-3, 4e-3, 5e-3, 6e-3],  # Narrow range around best LR
    "weight_decay": [0.08, 0.10, 0.12],  # Fine-tune weight decay
    "gradient_clip_val": [10.0],  # Fix this parameter
    "noise_schedule": ["log-linear"],  # Fix this parameter
    "warmup_steps": [250, 500],
    "batch_size": [2048],
}
```

### 5. Compare Across Tasks
```bash
# Run same grid on multiple tasks
for task in bfvp parity arithmetic; do
    python scripts/grid_search_mdm.py --task $task --grid recommended
done

# Analyze each
for task in bfvp parity arithmetic; do
    echo "=== $task ==="
    python scripts/analyze_grid_search.py --results_dir grid_search_results_${task}
done
```

## Troubleshooting

### No Results Found
```bash
# Check if jobs ran
ls -la grid_search_results/*/validation_metrics.json

# Check SLURM logs for errors
tail grid_search_results/logs/*.err
```

### Jobs Failing
```bash
# Check individual job logs
less grid_search_results/logs/mdm_bfvp_lr1e-02_*.err

# Run one job locally to debug
bash grid_search_results/slurm_scripts/mdm_bfvp_lr5e-03_*.sh
```

### Analysis Errors
```bash
# Check what metrics are available
python -c "
import json
from pathlib import Path
metrics_file = list(Path('grid_search_results').rglob('validation_metrics.json'))[0]
with open(metrics_file) as f:
    data = json.load(f)
    if isinstance(data, list):
        print(data[-1].keys())
    else:
        print(data.keys())
"
```

## Example Workflow

Complete workflow for finding best hyperparameters:

```bash
# 1. Quick exploration
python scripts/grid_search_mdm.py --task bfvp --grid quick --output_dir search_v1
python scripts/analyze_grid_search.py --results_dir search_v1

# 2. Focused search based on quick results
# (Edit CUSTOM_GRID in grid_search_mdm.py based on recommendations)
python scripts/grid_search_mdm.py --task bfvp --grid recommended --output_dir search_v2
python scripts/analyze_grid_search.py --results_dir search_v2

# 3. Fine-tuning with multiple seeds
# (Use best config from search_v2, vary slightly)
python scripts/grid_search_mdm.py --task bfvp --grid recommended --num_seeds 5 --output_dir search_v3
python scripts/analyze_grid_search.py --results_dir search_v3

# 4. Train final model with best hyperparameters
python main.py data=bfvp algo=mdlm \
    optim.lr=5e-3 \
    optim.weight_decay=0.1 \
    trainer.gradient_clip_val=10.0 \
    noise=log-linear \
    lr_scheduler.lr_lambda.warmup_steps=250
```
