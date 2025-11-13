# Curriculum Learning - Changelog

## Recent Updates

### Enhanced Logging (Latest)

**Added prominent logging when curriculum advances to new bins:**
```
================================================================================
📚 CURRICULUM ADVANCEMENT: Bin 1 → Bin 2
   Previous length range: [5, 93]
   New length range:      [93, 181]
   Progress: 2/4 bins
================================================================================
```

**Added validation dataset confirmation:**
- Logs once at first validation: "Validation/Test: Using FULL dataset (curriculum filtering only applies to training)"
- Makes it clear that curriculum only affects training data

**Enhanced completion message:**
```
================================================================================
✅ Curriculum learning completed!
   Trained through all 4 bins
   Final length range: (224, 300)
================================================================================
```

### Bug Fix: Tokenizer Attribute Error

**Issue:** When using distributed training (DDP), the dataloader gets wrapped by Lightning and loses the custom `tokenizer` attribute, causing `AttributeError`.

**Fix:** Store tokenizer from the model during setup phase:
- Added `self.tokenizer = None` in `__init__`
- Store it in `setup()`: `self.tokenizer = pl_module.tokenizer`
- Use stored tokenizer instead of accessing from dataloader

**Result:** Works correctly in both single-GPU and multi-GPU setups.

### Depth-Based Task Support

**Added support for BFVP and arithmetic tasks:**
- Added `min_train_len` and `max_train_len` to configs
- Enhanced error messages for missing length parameters
- Created [scripts/measure_lengths.py](scripts/measure_lengths.py) to empirically measure lengths
- Documented the depth → length correlation

**Key insight:** Curriculum filters by final sequence length, not generation depth. This provides natural difficulty progression since deeper trees produce longer sequences.

## Log Output Examples

### During Training

```
[2025-11-13 16:58:35] Curriculum Learning enabled with 4 bins
[2025-11-13 16:58:35] Length range: [5, 300]
[2025-11-13 16:58:35] Epochs per bin: 2
[2025-11-13 16:58:35] Bin boundaries: [(5, 79), (79, 153), (153, 227), (227, 300)]

[2025-11-13 16:58:35] Epoch 0: Training on length range [5, 79] (bin 1/4, epoch 1/2)
[2025-11-13 17:01:39] Filtered dataset: 124932/499749 examples in length range [5, 79]

[2025-11-13 17:05:12] Validation/Test: Using FULL dataset (curriculum filtering only applies to training)

[2025-11-13 17:10:23] Epoch 2: Training on length range [5, 79] (bin 1/4, epoch 2/2)
[2025-11-13 17:13:45] Filtered dataset: 124932/499749 examples in length range [5, 79]

[2025-11-13 17:18:56]
[2025-11-13 17:18:56] ================================================================================
[2025-11-13 17:18:56] 📚 CURRICULUM ADVANCEMENT: Bin 1 → Bin 2
[2025-11-13 17:18:56]    Previous length range: [5, 79]
[2025-11-13 17:18:56]    New length range:      [79, 153]
[2025-11-13 17:18:56]    Progress: 2/4 bins
[2025-11-13 17:18:56] ================================================================================
[2025-11-13 17:18:56]
[2025-11-13 17:19:02] Filtered dataset: 125031/499749 examples in length range [79, 153]

... (continues through all bins)

[2025-11-13 18:45:23]
[2025-11-13 18:45:23] ================================================================================
[2025-11-13 18:45:23] ✅ Curriculum learning completed!
[2025-11-13 18:45:23]    Trained through all 4 bins
[2025-11-13 18:45:23]    Final length range: (227, 300)
[2025-11-13 18:45:23] ================================================================================
[2025-11-13 18:45:23]
```

## Key Features

### 1. Training-Only Filtering
- ✅ Curriculum applies **only** to training dataloader
- ✅ Validation and test sets use **full dataset**
- ✅ This ensures consistent evaluation across all epochs

### 2. Progressive Difficulty
- ✅ Divides length range into configurable bins
- ✅ Trains N epochs per bin before advancing
- ✅ Optional overlap between bins for smooth transitions

### 3. Automatic Length Detection
- ✅ Counts non-padding tokens via attention_mask or input_ids
- ✅ Works with any tokenization scheme
- ✅ Handles both trace and final_value formats

### 4. State Management
- ✅ Current bin and progress saved in checkpoints
- ✅ Seamless resumption from interrupted training
- ✅ State dict includes bin boundaries for reproducibility

### 5. Comprehensive Logging
- ✅ Clear advancement messages with visual separators
- ✅ Per-epoch progress tracking
- ✅ Dataset size after filtering
- ✅ Explicit confirmation of validation behavior

## Configuration

```yaml
# configs/config.yaml
curriculum:
  enabled: true
  num_bins: 4
  epochs_per_bin: 5
  overlap: 0.2

# configs/data/bfvp.yaml
properties:
  min_depth: 1
  max_depth: 5
  min_train_len: 5      # Empirically measured
  max_train_len: 300    # Empirically measured

# configs/data/arithmetic.yaml
properties:
  min_depth: 1
  max_depth: 4
  min_train_len: 5      # Empirically measured
  max_train_len: 150    # Empirically measured
```

## Validation Behavior

**Question:** Does validation use curriculum?

**Answer:** **NO.** Validation and test sets always use the **full dataset** without any length filtering.

**Why?**
- Consistent evaluation metrics across epochs
- Can measure generalization to all lengths
- Avoid biased accuracy estimates from partial data

**How it works:**
- `on_train_epoch_start`: Only modifies `trainer.train_dataloader`
- `on_validation_epoch_start`: Just logs confirmation, doesn't modify anything
- Validation dataloader remains untouched

## Files Modified

1. **[curriculum_callback.py](curriculum_callback.py)**:
   - Fixed tokenizer access bug
   - Enhanced logging for bin changes
   - Added validation dataset confirmation
   - Improved completion message

2. **[configs/data/bfvp.yaml](configs/data/bfvp.yaml)**:
   - Added `min_train_len: 5`
   - Added `max_train_len: 300`

3. **[configs/data/arithmetic.yaml](configs/data/arithmetic.yaml)**:
   - Added `min_train_len: 5`
   - Added `max_train_len: 150`

## Tools

**[scripts/measure_lengths.py](scripts/measure_lengths.py)**: Empirical length measurement
```bash
python scripts/measure_lengths.py --task bfvp --min_depth 1 --max_depth 5 --format trace
```

Shows per-depth statistics and recommends `min_train_len` / `max_train_len` values.

## Documentation

- **[CURRICULUM_LEARNING.md](CURRICULUM_LEARNING.md)**: Complete user guide
- **[CURRICULUM_DEPTH_SUPPORT.md](CURRICULUM_DEPTH_SUPPORT.md)**: Depth-based tasks guide
- **[HOW_TO_SET_LENGTH_RANGES.md](HOW_TO_SET_LENGTH_RANGES.md)**: How to determine length ranges
- **[CURRICULUM_SUMMARY.md](CURRICULUM_SUMMARY.md)**: Implementation summary

## Future Enhancements

Possible improvements (not implemented):
- [ ] Dynamic `epochs_per_bin` based on validation metrics
- [ ] Curriculum scheduling (e.g., cosine annealing of difficulty)
- [ ] Multi-dimensional curriculum (length + other properties)
- [ ] Reverse curriculum option
- [ ] Web dashboard for curriculum visualization
