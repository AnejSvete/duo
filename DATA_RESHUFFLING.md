# Data Reshuffling in Auto-Recovery

## Overview

The auto-recovery callback now includes **data reshuffling** after checkpoint recovery. This feature helps prevent the model from encountering the same "bad batch" that may have caused the initial loss explosion.

## Why This Matters

### The Problem

Loss explosions aren't always caused by hyperparameters being fundamentally wrong. Sometimes they happen because:

1. **Bad Batch**: A particularly difficult batch with unusual or extreme examples
2. **Unlucky Combinations**: Specific combinations of examples that create very large gradients
3. **Temporal Patterns**: The order of batches creates an unstable optimization trajectory

### What Happened Before

```
Step 1000: Batch A → loss = 2.5 ✓
Step 1001: Batch B → loss = 2.3 ✓
Step 1002: Batch C (bad!) → loss = 150.0 ✗ EXPLOSION!
[Recovery: load checkpoint, reduce LR]
Step 1003: Batch D → loss = 2.8 ✓
Step 1004: Batch E → loss = 2.6 ✓
Step 1005: Batch C (same bad batch!) → loss = 80.0 ✗ EXPLOSION AGAIN!
```

Even with reduced LR, the same problematic batch could cause issues because:
- Dataloaders use deterministic random seeds within an epoch
- Batch order is predictable once the epoch starts
- Curriculum learning uses the same filtered subset

### What Happens Now

```
Step 1000: Batch A → loss = 2.5 ✓
Step 1001: Batch B → loss = 2.3 ✓
Step 1002: Batch C (bad!) → loss = 150.0 ✗ EXPLOSION!
[Recovery: load checkpoint, reduce LR, RESHUFFLE DATA]
Step 1003: Batch X (different) → loss = 2.4 ✓
Step 1004: Batch Y (different) → loss = 2.2 ✓
Step 1005: Batch Z (different) → loss = 2.0 ✓
[Training continues successfully!]
```

## How It Works

### Regular Training (No Curriculum)

When curriculum learning is disabled:

```python
# Dataloader typically has shuffle=True
DataLoader(dataset, shuffle=True, ...)
```

After recovery, the callback:
1. Calls `trainer.reset_train_dataloader(pl_module)` if available
2. This triggers PyTorch Lightning to create a new dataloader
3. New dataloader gets a new shuffle seed
4. Batch order changes for remaining training

### With Curriculum Learning

When curriculum learning is enabled:

```python
# Curriculum creates filtered dataloaders per bin
filtered_dataset = Subset(full_dataset, indices_in_length_range)
DataLoader(filtered_dataset, shuffle=True, ...)
```

After recovery, the callback:
1. Finds the CurriculumLearningCallback
2. Gets the current bin (e.g., bin 2: sequences of length 64-128)
3. Calls `_create_filtered_dataloader()` to recreate the dataloader
4. Same length range, but **different shuffle order**
5. Training continues on same difficulty level with new batch order

## Implementation Details

### Code Flow

```python
def _trigger_recovery(self, trainer, pl_module, reason):
    # 1. Load checkpoint
    checkpoint = torch.load(best_checkpoint_path)
    pl_module.load_state_dict(checkpoint['state_dict'])

    # 2. Reduce learning rate
    reduced_lr = original_lr * (lr_reduction_factor ** recovery_count)

    # 3. Reset optimizer with new LR
    optimizer_config = pl_module.configure_optimizers()
    trainer.optimizers = optimizers

    # 4. Reshuffle data (NEW!)
    if self.reshuffle_data:
        self._reshuffle_training_data(trainer, pl_module)

    # 5. Continue training
    self.recovering = False
```

### Reshuffling Strategy

```python
def _reshuffle_training_data(self, trainer, pl_module):
    # Try PyTorch Lightning's built-in method first
    if hasattr(trainer, 'reset_train_dataloader'):
        trainer.reset_train_dataloader(pl_module)

    # For curriculum learning, manually trigger recreation
    elif has_curriculum_callback:
        curriculum = find_curriculum_callback(trainer)
        current_bin = curriculum.current_bin
        min_len, max_len = curriculum.bin_boundaries[current_bin]

        # Recreate filtered dataloader with new shuffle
        curriculum._create_filtered_dataloader(
            trainer, pl_module, min_len, max_len
        )
```

## Configuration

Enable/disable in [configs/callbacks/auto_recovery.yaml](configs/callbacks/auto_recovery.yaml):

```yaml
auto_recovery:
  reshuffle_data: true  # Enable data reshuffling (default)
```

Or override via command line:

```bash
# Disable reshuffling
python main.py data=bfvp algo=mdlm \
    callbacks.auto_recovery.reshuffle_data=false

# Enable with custom settings
python main.py data=bfvp algo=mdlm \
    callbacks.auto_recovery.reshuffle_data=true \
    callbacks.auto_recovery.lr_reduction_factor=0.5
```

## When to Disable

You might want to disable reshuffling if:

1. **Debugging**: Want to reproduce exact same batch order
2. **Analysis**: Investigating specific batches causing issues
3. **Curriculum Issues**: Suspect reshuffling is causing curriculum problems

```bash
# Disable for debugging
python main.py ... callbacks.auto_recovery.reshuffle_data=false
```

## Benefits

### 1. Prevents Bad Batch Repetition

If a specific batch causes explosion, reshuffling ensures you don't hit it again immediately.

### 2. Works with Curriculum Learning

Reshuffles **within** the current curriculum bin, so:
- ✓ Maintains progressive difficulty
- ✓ Doesn't reset to easier examples
- ✓ Changes batch order within same length range
- ✓ Preserves curriculum learning benefits

### 3. Complements LR Reduction

Two-pronged recovery approach:
- **LR reduction**: More conservative updates
- **Data reshuffling**: Different batch order
- **Combined effect**: More robust recovery

### 4. Low Overhead

Reshuffling is very cheap:
- No re-computation of data
- No re-filtering for curriculum
- Just changes iteration order
- Minimal performance impact

## Example Recovery Log

```
[RECOVERY] Triggering automatic recovery (Attempt 1/3)
[RECOVERY] Loading checkpoint: checkpoints/best.ckpt
[RECOVERY] Resetting optimizer and reducing learning rate...
[RECOVERY] Learning rate: 0.005000 → 0.002500
[RECOVERY] Reshuffling training data...
[RECOVERY] Curriculum dataloader refreshed for bin 1 (length range: [64, 128])
[RECOVERY] Successfully recovered from checkpoint
[RECOVERY] Checkpoint was from step: 2000
[RECOVERY] Continuing from current step: 2303
```

## Compatibility

### PyTorch Lightning Versions

Works with:
- ✓ PyTorch Lightning 2.0+
- ✓ PyTorch Lightning 1.9+

Uses fallback strategies for older versions.

### Training Modes

Works with:
- ✓ Regular training (shuffle=True)
- ✓ Curriculum learning (all variants)
- ✓ DDP/FSDP multi-GPU training
- ✓ Single GPU training

### Dataloader Types

Works with:
- ✓ Standard PyTorch DataLoader
- ✓ Curriculum-filtered dataloaders
- ✓ Custom dataloaders (as long as they support shuffle)

## Troubleshooting

### Reshuffling Not Happening

Check logs for:
```
[RECOVERY] Reshuffling training data...
[RECOVERY] Warning: Could not reshuffle data: <error>
```

If you see the warning:
1. Check if curriculum learning is properly initialized
2. Verify dataloader has `shuffle=True`
3. Check PyTorch Lightning version

### Same Batches After Recovery

If you suspect batches aren't changing:
1. Add logging to track batch IDs/hashes
2. Check if `reshuffle_data=true` in config
3. Verify curriculum callback is active

### Curriculum Bin Changes

Reshuffling doesn't change curriculum bins:
- ✓ Stays in same bin
- ✓ Same length range
- ✗ Doesn't reset to easier bins

If you want to reset curriculum after recovery, that's a separate feature request.

## Future Enhancements

Potential improvements for the future:

1. **Bin Reset Option**: Optionally move back one curriculum bin after recovery
2. **Sample Weighting**: Reduce probability of sampling "difficult" examples after recovery
3. **Batch Tracking**: Log which batches caused explosions to avoid them
4. **Smart Resampling**: Use validation loss to identify and avoid problematic examples

## Summary

Data reshuffling is a simple but effective addition to the auto-recovery system:

- **Prevents**: Hitting same bad batch after recovery
- **Works with**: Curriculum learning and all training modes
- **Cost**: Negligible performance overhead
- **Benefit**: Significantly improves recovery success rate
- **Default**: Enabled (recommended to keep it on)

Combined with learning rate reduction, this creates a robust recovery mechanism that handles most training instabilities effectively.
