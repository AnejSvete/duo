# MDM Training Fixes Summary

## Problems Identified and Fixed

### 1. Checkpoint Recovery Bug (CRITICAL) ⚠️

**Problem**: When training recovered from a loss explosion by loading a previous checkpoint, it would:
- Load the model weights correctly ✓
- Reset the optimizer state ✓
- **BUT**: Keep the same high learning rate that caused the explosion ✗

This meant the model would just explode again immediately after recovery!

**Root Cause**: The `AutoRecoveryCallback` in [auto_recovery_callback.py](auto_recovery_callback.py) created fresh optimizers but didn't reduce the learning rate.

**Fix Applied**:
1. Added `lr_reduction_factor` parameter (default: 0.5)
2. After each recovery, learning rate is reduced: `new_lr = old_lr * (0.5 ^ recovery_count)`
3. Added `reshuffle_data` parameter (default: True) to randomize batch order
4. Recovery process now:
   - First recovery: LR × 0.5, data reshuffled
   - Second recovery: LR × 0.25, data reshuffled again
   - Third recovery: LR × 0.125, data reshuffled again (then stops)

**Files Modified**:
- [auto_recovery_callback.py](auto_recovery_callback.py) - Lines 47, 58, 160-301
- [configs/callbacks/auto_recovery.yaml](configs/callbacks/auto_recovery.yaml) - Lines 10-11

**Impact**: Training should now successfully recover from loss explosions without repeating them! The combination of lower LR and different batch order prevents the same failure pattern.

### 2. No Systematic Hyperparameter Search

**Problem**: Finding good hyperparameters for MDM required manual trial-and-error, which is:
- Time-consuming
- Not systematic
- Easy to miss good configurations
- Hard to understand which hyperparameters matter most

**Solution**: Created comprehensive grid search infrastructure

**New Tools**:
1. **[scripts/grid_search_mdm.py](scripts/grid_search_mdm.py)**: Automated hyperparameter grid search
2. **[scripts/analyze_grid_search.py](scripts/analyze_grid_search.py)**: Analysis and visualization of results
3. **[GRID_SEARCH.md](GRID_SEARCH.md)**: Complete usage guide

## How to Use

### Quick Start

```bash
# 1. Run grid search (recommended starting point)
python scripts/grid_search_mdm.py --task bfvp --grid recommended

# 2. Wait for jobs to complete, then analyze
python scripts/analyze_grid_search.py --results_dir grid_search_results
```

### Grid Sizes

- **Quick** (~48 configs): Fast initial exploration
- **Recommended** (~108 configs): **Start here** - good balance
- **Full** (~864 configs): Comprehensive but expensive

### Hyperparameters Searched

1. **Learning Rate**: 1e-3 to 1e-2 (most important!)
2. **Weight Decay**: 0.0 to 0.2
3. **Gradient Clipping**: 1.0 to 20.0
4. **Noise Schedule**: log-linear, linear, cosine
5. **Warmup Steps**: 100 to 1000
6. **Batch Size**: 1024, 2048, 4096

### Analysis Output

The analysis script provides:
- **Hyperparameter importance ranking** - which params matter most
- **Top K best configurations** - ranked by performance
- **Training stability analysis** - which configs exploded
- **Recommendations** - suggested safe ranges
- **Visualizations** - box plots and learning curves
- **CSV export** - for further analysis

## Testing the Fixes

### Test 1: Verify Auto-Recovery Works

```bash
# Train with intentionally high LR to trigger recovery
python main.py data=bfvp algo=mdlm optim.lr=1e-1 \
    callbacks.auto_recovery.lr_reduction_factor=0.5

# Watch the logs - you should see:
# 1. Loss explosion detected
# 2. "Reducing learning rate from X to Y"
# 3. Recovery from checkpoint
# 4. Training continues with lower LR
```

### Test 2: Run Small Grid Search

```bash
# Quick test on small grid
python scripts/grid_search_mdm.py --task bfvp --grid quick --num_seeds 1

# Analyze results
python scripts/analyze_grid_search.py --results_dir grid_search_results
```

## Configuration Changes

### Auto-Recovery Settings

Edit [configs/callbacks/auto_recovery.yaml](configs/callbacks/auto_recovery.yaml):

```yaml
auto_recovery:
  loss_threshold: 5.0  # Trigger at loss > 5.0 (you set this to 5.0)
  patience: 3  # Wait 3 consecutive bad steps
  lr_reduction_factor: 0.5  # NEW: Reduce LR by 50% after recovery
  reshuffle_data: true  # NEW: Reshuffle training data after recovery
  reset_optimizer: true
```

### Recommended MDM Settings (based on literature)

While you wait for grid search results, these are reasonable defaults:

```bash
python main.py data=bfvp algo=mdlm \
    optim.lr=5e-3 \
    optim.weight_decay=0.1 \
    trainer.gradient_clip_val=10.0 \
    noise=log-linear \
    lr_scheduler.lr_lambda.warmup_steps=250 \
    loader.batch_size=2048
```

## Expected Improvements

### Before Fix
```
Step 1000: loss = 2.5 ✓
Step 1500: loss = 2.1 ✓
Step 2000: loss = 150.0 ✗ EXPLOSION!
Step 2001: [Recovery] Loading checkpoint from step 1500
Step 2001: [Recovery] Resetting optimizer
Step 2002: loss = 145.0 ✗ STILL EXPLODING!
Step 2003: loss = 200.0 ✗
[Training stops - max recoveries reached]
```

### After Fix
```
Step 1000: loss = 2.5 ✓
Step 1500: loss = 2.1 ✓
Step 2000: loss = 150.0 ✗ EXPLOSION!
Step 2001: [Recovery] Loading checkpoint from step 1500
Step 2001: [Recovery] Reducing LR: 5e-3 → 2.5e-3
Step 2001: [Recovery] Resetting optimizer
Step 2001: [Recovery] Reshuffling training data
Step 2002: loss = 2.3 ✓ STABLE! (lower LR + different batch order)
Step 2500: loss = 1.8 ✓
[Training continues successfully]
```

## Next Steps

### Immediate
1. **Test the recovery fix** on your current failing runs
2. **Start a recommended grid search** to find optimal hyperparameters
3. **Monitor the grid search** progress and do incremental analysis

### After Grid Search Completes
1. **Analyze results** to find best configurations
2. **Fine-tune** around the best configs with more seeds
3. **Update default configs** with best hyperparameters
4. **Document** findings for future reference

### For Production Use
1. Use grid search to find task-specific hyperparameters
2. Run multiple seeds for best configs to ensure robustness
3. Consider adding grid search to your experiment launch scripts

## Files Created/Modified

### New Files
- `scripts/grid_search_mdm.py` - Grid search runner
- `scripts/analyze_grid_search.py` - Results analysis
- `GRID_SEARCH.md` - Complete usage guide
- `FIXES_SUMMARY.md` - This document

### Modified Files
- `auto_recovery_callback.py` - Added LR reduction and data reshuffling on recovery
- `configs/callbacks/auto_recovery.yaml` - Added lr_reduction_factor and reshuffle_data parameters

## Additional Notes

### Why Learning Rate Reduction Works

When a model's loss explodes, it's usually because:
1. The learning rate is too high for the current loss landscape
2. The model entered a region with large gradients
3. A single large update destabilized the model

Simply reloading the checkpoint doesn't solve the problem because:
- The same high LR will cause the same explosion
- The loss landscape hasn't changed

Reducing the LR after recovery:
- Makes updates more conservative
- Prevents re-entering unstable regions
- Allows fine-tuning from the recovered state
- Is a standard practice in training deep networks

### Why Data Reshuffling Works

Sometimes loss explosions are caused by specific "bad batches":
1. A particularly difficult batch with unusual examples
2. Unlucky combination of examples that creates large gradients
3. Batch order that leads to unstable optimization trajectory

Simply reloading the checkpoint might hit the same bad batch again because:
- The dataloader uses the same random seed
- The batch order is deterministic within an epoch
- Curriculum learning keeps the same filtered dataset

Reshuffling the data after recovery:
- Changes batch composition and order
- Prevents repeating the exact sequence that caused explosion
- Works with curriculum learning (reshuffles within current bin)
- Adds randomness to break out of unstable patterns
- Complements LR reduction for more robust recovery

### Grid Search Design Choices

**Why these hyperparameters?**
- **Learning rate**: Most impactful for diffusion models
- **Weight decay**: Important for generalization
- **Gradient clipping**: Critical for stability
- **Noise schedule**: Affects training dynamics significantly
- **Warmup**: Stabilizes early training
- **Batch size**: Affects gradient variance and memory

**Why multiple seeds?**
- Neural network training is stochastic
- A config might work well by chance
- Multiple seeds ensure robustness

**Why 3 grid sizes?**
- Quick: Debug and initial exploration
- Recommended: Good default for most use cases
- Full: When you have compute and want thoroughness

## Troubleshooting

### Grid Search Jobs Not Running
```bash
# Check SLURM queue
squeue -u $USER

# Check job logs
less grid_search_results/logs/*.err
```

### Recovery Not Working
```bash
# Check if checkpoint exists
ls -la checkpoints/best.ckpt

# Check recovery logs in training output
grep -i "recovery" <training_log>
```

### Analysis Script Errors
```bash
# Check if validation metrics exist
find grid_search_results -name "validation_metrics.json"

# Verify metrics format
python -c "import json; print(json.load(open('path/to/validation_metrics.json')))"
```
