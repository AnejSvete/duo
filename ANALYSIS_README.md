# Model Analysis Guide

This guide explains how to train models and analyze their results using the comparison scripts.

## How Training Outputs Work

### Directory Structure

When you run training (either via `python main.py` or via slurm scripts), Hydra creates an output directory structure:

```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── config_tree.txt           # Full configuration
        ├── validation_metrics.json   # Validation metrics per epoch (automatically saved)
        ├── test_metrics.json         # Test metrics (saved at end, if test set evaluated)
        ├── checkpoints/
        │   └── last.ckpt
        └── .hydra/
            └── config.yaml
```

### What Metrics Are Saved

**Automatically saved during training:**

1. **validation_metrics.json** - List of per-epoch dictionaries containing:
   - `epoch`: Current epoch number
   - `global_step`: Total number of training updates
   - `trainer/nll`: Negative log-likelihood
   - `val/acc_token`: Token-level accuracy (for formal language tasks)
   - `val/acc_exact`: Exact sequence match accuracy
   - **Decoding strategy metrics** (for MDLM models):
     - `val/random_acc_exact`, `val/random_acc_token`, `val/random_correct_prediction`
     - `val/top_k_acc_exact`, `val/top_k_acc_token`, `val/top_k_correct_prediction`
     - `val/one_level_acc_exact`, `val/one_level_acc_token`, `val/one_level_correct_prediction`
     - `val/all_at_once_acc_exact`, `val/all_at_once_acc_token`, `val/all_at_once_correct_prediction`
     - `val/one_at_a_time_acc_exact`, `val/one_at_a_time_acc_token`, `val/one_at_a_time_correct_prediction`
   - **For AR/LT models**: `val/default_acc_*` metrics

2. **test_metrics.json** - Single dictionary with final test results (same metrics as validation, prefixed with `test/`)

### How to Populate Output Directories

#### Option 1: Run Training Locally

```bash
# Train an AR model with chain-of-thought
python main.py data=bfvp algo=ar model=nano data.properties.format=trace

# Train a looping transformer
python main.py data=parity algo=lt model=nano algo.looping_type=log

# Train MDLM
python main.py data=bfvp algo=mdlm model=nano
```

Each run creates a timestamped directory in `outputs/YYYY-MM-DD/HH-MM-SS/`.

#### Option 2: Run Training via Slurm (Cluster)

```bash
# Create watch folder first
mkdir -p watch_folder

# Submit jobs
sbatch scripts/train_cot.sh bfvp 128        # AR with CoT
sbatch scripts/train_looping.sh parity 128  # LT with log depth
sbatch scripts/train_mdm.sh bfvp 128        # MDLM
```

Outputs go to `outputs/YYYY-MM-DD/HH-MM-SS/` on the compute node.

#### Option 3: Use Existing Runs

If you have existing runs in different locations, just point the analysis script to those directories.

### Metrics JSON Format

**validation_metrics.json:**
```json
[
  {
    "trainer/nll": 2.3456,
    "val/random_acc_exact": 0.45,
    "val/random_acc_token": 0.78,
    "val/top_k_acc_exact": 0.52,
    "val/top_k_acc_token": 0.82,
    "epoch": 0,
    "global_step": 100
  },
  {
    "trainer/nll": 1.8234,
    "val/random_acc_exact": 0.65,
    "val/random_acc_token": 0.88,
    "val/top_k_acc_exact": 0.71,
    "val/top_k_acc_token": 0.91,
    "epoch": 1,
    "global_step": 200
  }
]
```

**test_metrics.json:**
```json
{
  "test/trainer/nll": 1.5678,
  "test/val/random_acc_exact": 0.72,
  "test/val/random_acc_token": 0.90,
  "test/val/top_k_acc_exact": 0.78,
  "test/val/top_k_acc_token": 0.93,
  "epoch": 50,
  "global_step": 10000
}
```

## Using the Analysis Scripts

### analyze_metrics.py (Single Run Analysis)

Analyzes a single training run:

```bash
# Analyze validation and test metrics for one run
python analyze_metrics.py --metrics_dir outputs/2025-01-15/14-30-22/

# Specify output directory
python analyze_metrics.py \
    --metrics_dir outputs/2025-01-15/14-30-22/ \
    --output_dir analysis/run1/

# Generate intermediate report during training
python analyze_metrics.py \
    --metrics_dir outputs/2025-01-15/14-30-22/ \
    --report_type intermediate
```

**Outputs:**
- `validation_metrics_plot.png` - Training curves
- `test_metrics_plot.png` - Bar chart of final results
- `validation_summary.csv` - Statistical summary
- `test_results_table.tex` - LaTeX table for publication

### compare_models.py (Multi-Run Comparison)

Compares multiple runs across all metrics. **Automatically groups by language/task** and creates both per-language and cross-language analyses:

```bash
# Compare specific runs
python compare_models.py \
    --run_dirs outputs/2025-01-15/10-00-00/ \
              outputs/2025-01-15/11-00-00/ \
              outputs/2025-01-15/12-00-00/

# Use glob patterns to compare all runs from a day
python compare_models.py --run_dirs outputs/2025-01-15/*/

# Compare all experiments (multiple languages)
python compare_models.py --run_dirs outputs/*/*/

# Specify output directory
python compare_models.py \
    --run_dirs outputs/2025-01-15/*/ \
    --output_dir analysis/comparison_jan15/
```

**Output Directory Structure:**

When multiple languages are detected, the script automatically organizes results:

```
analysis/
├── by_language/           # Per-language analyses
│   ├── bfvp/
│   │   ├── validation_trends_*.png
│   │   ├── test_metrics_comparison.csv
│   │   ├── test_performance_heatmap.png
│   │   └── ...
│   ├── parity/
│   │   └── ...
│   └── arithmetic/
│       └── ...
├── combined/              # All runs together (cross-language)
│   ├── validation_trends_*.png
│   ├── test_metrics_comparison.csv
│   └── ...
├── cross_language_nll.png              # Heatmaps: Languages × Algorithms
├── cross_language_acc_exact.png        # One for each metric
├── cross_language_acc_token.png
└── cross_language_summary.csv          # Summary table
```

**Generated Files:**

*Per-language directories* (`by_language/{lang}/`):
- `validation_trends_*.png` - Validation curves for this language only
  - Plotted vs both `global_step` and `epoch`
  - Compares AR, LT, MDLM for this specific task
- `decoding_strategies_*.png` - Strategy comparison plots (MDLM only)
  - Compares random, top_k, one_level, all_at_once, one_at_a_time
- `test_metrics_comparison.csv` - Test results for this language
- `test_metrics_comparison.tex` - LaTeX table for publication
- `test_comparison_*.png` - Bar charts per metric
- `test_performance_heatmap.png` - Runs × Metrics heatmap

*Cross-language summary* (root of output_dir):
- `cross_language_summary.csv` - Summary table (Language × Algorithm × Metrics)
- `cross_language_*.png` - Heatmap for each metric showing:
  - Which algorithm works best on which language
  - Rows = Languages, Columns = Algorithms
  - One heatmap per metric (nll, acc_exact, etc.)

*Combined analysis* (`combined/`):
- All runs analyzed together regardless of language
- Useful for seeing overall trends across all experiments

## Example Workflow

### 1. Train Multiple Models on Multiple Languages

```bash
# Train all algorithms on BFVP
python main.py data=bfvp algo=ar model=nano wandb.name=bfvp-ar
python main.py data=bfvp algo=lt algo.looping_type=log model=nano wandb.name=bfvp-lt
python main.py data=bfvp algo=mdlm model=nano wandb.name=bfvp-mdlm

# Train all algorithms on Parity
python main.py data=parity algo=ar model=nano wandb.name=parity-ar
python main.py data=parity algo=lt algo.looping_type=log model=nano wandb.name=parity-lt
python main.py data=parity algo=mdlm model=nano wandb.name=parity-mdlm

# Train all algorithms on Arithmetic
python main.py data=arithmetic algo=ar model=nano wandb.name=arithmetic-ar
python main.py data=arithmetic algo=lt algo.looping_type=log model=nano wandb.name=arithmetic-lt
python main.py data=arithmetic algo=mdlm model=nano wandb.name=arithmetic-mdlm
```

This creates 9 runs total (3 languages × 3 algorithms).

### 2. Check Individual Runs During Training

```bash
# Monitor progress of a specific run
python analyze_metrics.py \
    --metrics_dir outputs/2025-01-15/14-30-22/ \
    --report_type intermediate
```

### 3. Compare All Runs After Training

```bash
# Compare ALL experiments (automatic per-language + cross-language analysis)
python compare_models.py \
    --run_dirs outputs/2025-01-15/*/ \
    --output_dir analysis/full_comparison/

# This will:
# 1. Group runs by language (bfvp, parity, arithmetic)
# 2. Create per-language comparisons in by_language/*/
# 3. Create cross-language heatmaps showing which algo works best where
# 4. Create combined analysis with all runs together
```

Or compare just one language:

```bash
# Compare only BFVP experiments
python compare_models.py \
    --run_dirs outputs/2025-01-15/bfvp-*/ \
    --output_dir analysis/bfvp_only/
```

### 4. Review Generated Plots

For **multi-language experiments**, you'll get:

1. **Per-language comparisons** (`by_language/bfvp/`, `by_language/parity/`, etc.):
   - `validation_trends_trainer_nll_global_step.png` - Learning curves
   - `test_comparison_acc_exact.png` - Which algorithm is best for this task
   - `test_performance_heatmap.png` - Overview of all metrics

2. **Cross-language summary** (root directory):
   - `cross_language_nll.png` - Heatmap showing NLL for each (language, algorithm) pair
   - `cross_language_acc_exact.png` - Accuracy comparison across tasks
   - `cross_language_summary.csv` - Full table with all metrics

   These heatmaps answer: **"Which algorithm is best for which type of problem?"**

3. **Decoding strategies** (MDLM only, per language):
   - `decoding_strategies_*_acc_exact.png` - Compare random, top_k, one_level, etc.

For **single-language experiments**, you'll get just the combined analysis.

## Tips

1. **Naming runs**: Use descriptive `wandb.name` to make plots readable:
   ```bash
   python main.py ... wandb.name=bfvp-ar-depth5
   ```

2. **Organizing outputs**: You can manually organize outputs:
   ```bash
   mkdir -p experiments/bfvp/
   cp -r outputs/2025-01-15/14-30-22/ experiments/bfvp/ar-cot/
   python compare_models.py --run_dirs experiments/bfvp/*/
   ```

3. **Strategy comparison**: MDLM models automatically evaluate 5 decoding strategies:
   - `random`: Randomly unmask k positions
   - `top_k`: Unmask highest-confidence positions
   - `one_level`: Unmask one hierarchical level (uses `|` delimiters)
   - `all_at_once`: Single-step generation
   - `one_at_a_time`: Sequential left-to-right

4. **Global step vs epoch**:
   - `global_step` = total number of gradient updates (more precise)
   - `epoch` = number of passes through training data
   - Both are saved and can be used for x-axis in plots

## Troubleshooting

**Q: My validation_metrics.json is missing decoding strategy metrics**

A: Make sure you're using the updated `trainer_base.py` which saves all callback metrics. The update captures all logged metrics, not just `valid_nlls`.

**Q: The script says "No validation metrics found"**

A: Check that:
1. Training ran long enough to complete at least one validation epoch
2. The path to `--run_dirs` is correct
3. The files are in `{run_dir}/validation_metrics.json`

**Q: How do I know which algorithm each run used?**

A: The script tries to infer from:
1. Directory name (looks for keywords: ar, lt, mdlm, looping, cot)
2. Config file if available
3. You can manually organize runs into descriptive directories

**Q: Can I compare runs from different tasks?**

A: Yes! The script handles this - it will extract task names and show them in legends. Just make sure the metrics are comparable.

**Q: Plots are too crowded with many runs**

A: Consider:
1. Filtering runs first: `--run_dirs outputs/*/bfvp-*ar*/`
2. Organizing into separate comparisons
3. The heatmap view works well for many runs

## Configuration

Metrics are saved to `config.checkpointing.save_dir` (default: current working directory from Hydra).

To change where metrics are saved:
```bash
python main.py ... checkpointing.save_dir=/path/to/custom/dir/
```

All analysis scripts respect the directory structure and will find metrics automatically.
