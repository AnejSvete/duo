# Curriculum Learning: Final Bin Behavior

## How the Last Bin Works

The curriculum learning implementation handles the final (longest) bin specially to ensure it trains until `max_steps` is reached.

## Behavior

### Bins 1 through N-1
- Train for **exactly** `epochs_per_bin` epochs
- Then advance to next bin

### Final Bin (Bin N)
- Trains **until `max_steps` is reached**
- Does NOT stop after `epochs_per_bin` epochs
- Continues training on the full length range until training ends

## Implementation

The key logic is in [curriculum_callback.py](curriculum_callback.py):

```python
# Only advance if NOT in the final bin
if self.epochs_in_current_bin >= self.epochs_per_bin and self.current_bin < self.num_bins - 1:
    self.current_bin += 1
```

The condition `self.current_bin < self.num_bins - 1` prevents advancing beyond the last bin.

## Example Timeline

**Configuration:**
- `curriculum.num_bins = 4`
- `curriculum.epochs_per_bin = 5`
- `trainer.max_steps = 10000`
- Assume ~1000 steps per epoch

**Training progression:**

```
Epochs 0-4:   Bin 1 [5, 79]      (5 epochs = ~5000 steps)
Epochs 5-9:   Bin 2 [79, 153]    (5 epochs = ~5000 steps)
Epochs 10-14: Bin 3 [153, 227]   (5 epochs = ~5000 steps)
Epochs 15+:   Bin 4 [227, 300]   (continues until max_steps reached)
              ↑
              Final bin trains until step 10000
```

## Log Output

### When entering the final bin:

```
================================================================================
📚 CURRICULUM ADVANCEMENT: Bin 3 → Bin 4
   Previous length range: [153, 227]
   New length range:      [227, 300]
   Progress: 4/4 bins
   Note: Final bin will train until max_steps is reached
================================================================================
```

### During final bin training:

```
Epoch 15: Training on length range [227, 300] (bin 4/4, final bin - training until max_steps)
Epoch 16: Training on length range [227, 300] (bin 4/4, final bin - training until max_steps)
Epoch 17: Training on length range [227, 300] (bin 4/4, final bin - training until max_steps)
...
(continues until max_steps)
```

## Why This Design?

### Advantages

1. **Maximum exposure to hardest examples**: The longest sequences (often hardest) get the most training time
2. **Natural curriculum conclusion**: Training naturally ends on the full difficulty range
3. **Flexible total training budget**: Works with any `max_steps` value
4. **No wasted compute**: Uses all available training steps

### Alternative (not implemented)

You could make the final bin also stop after `epochs_per_bin`, but this would mean:
- ❌ Training might end before `max_steps`
- ❌ Need to manually calculate epochs to match `max_steps`
- ❌ Less exposure to hardest examples

## Calculating Bin Durations

If you want roughly equal time per bin:

**Total epochs needed:**
```
total_epochs = (num_bins - 1) * epochs_per_bin + remaining_epochs
```

Where `remaining_epochs` depends on your `max_steps`.

**Example:**
- Want 20 total epochs
- Use 4 bins
- Calculate: `(4-1) * epochs_per_bin + remaining = 20`
- If `epochs_per_bin = 5`: `3 * 5 + remaining = 20` → remaining = 5 epochs
- So bins get: 5, 5, 5, 5 epochs (equal)

**Another example:**
- Want 30 total epochs
- Use 4 bins with `epochs_per_bin = 5`
- Calculate: `3 * 5 + remaining = 30` → remaining = 15 epochs
- So bins get: 5, 5, 5, 15 epochs (final bin gets 3x more)

## Step-Based Alternative (Not Implemented)

If you wanted bins based on steps instead of epochs, you could modify the callback to:
- Track `trainer.global_step` instead of epochs
- Advance when `global_step >= next_bin_step`
- Calculate bin boundaries in step space

This would give more precise control but is more complex. The current epoch-based approach is simpler and works well in practice.

## Configuration Tips

### For equal time per bin
Set `epochs_per_bin` so that `num_bins * epochs_per_bin ≈ total_epochs`:

```yaml
curriculum:
  num_bins: 4
  epochs_per_bin: 5    # 4 * 5 = 20 epochs total
```

### For more time on final bin
Use fewer `epochs_per_bin`:

```yaml
curriculum:
  num_bins: 4
  epochs_per_bin: 3    # 3+3+3+remaining epochs
                       # If 20 total: 3+3+3+11 = final bin gets ~55% of training
```

### For very gradual curriculum
Use more bins with fewer epochs each:

```yaml
curriculum:
  num_bins: 8
  epochs_per_bin: 2    # 2+2+2+2+2+2+2+remaining
                       # More frequent progression
```

## Summary

- ✅ First N-1 bins: Train for exactly `epochs_per_bin` epochs
- ✅ Final bin: Trains until `max_steps` is reached
- ✅ Clear logging indicates "final bin - training until max_steps"
- ✅ No compute wasted - uses full training budget
- ✅ Maximum exposure to hardest (longest) examples

This design ensures you get the full benefit of curriculum learning while utilizing your entire training budget!
