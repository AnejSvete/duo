# Curriculum Learning Support for Depth-Based Tasks

This document explains how curriculum learning works with depth-based tasks (BFVP and arithmetic).

## Overview

While BFVP and arithmetic generate examples using **tree depth** parameters (`min_depth`, `max_depth`), curriculum learning operates on the **final sequence length** (number of tokens) after the tree is converted to text.

## Key Concepts

### Generation vs. Filtering

1. **Generation Phase** (BFVP/Arithmetic):
   - Randomly selects depth from `[min_depth, max_depth]`
   - Generates tree of that depth
   - Converts tree to token sequence
   - Different depths produce varying sequence lengths

2. **Curriculum Filtering Phase**:
   - Counts tokens in each generated example
   - Filters examples by `[min_train_len, max_train_len]` for current bin
   - Generally: deeper trees → longer sequences (with variation)

### Why This Works

Depth and sequence length are naturally correlated:
- **Shallow trees (depth 1-2)** → Short sequences (5-50 tokens)
- **Medium trees (depth 3-4)** → Medium sequences (50-150 tokens)
- **Deep trees (depth 5+)** → Long sequences (150-300 tokens)

The curriculum automatically learns simpler (shallower) examples first because they tend to produce shorter sequences.

## Configuration

### BFVP Example

```yaml
# configs/data/bfvp.yaml
language: bfvp
properties:
  num_vars: 2
  min_depth: 1          # Generate trees from depth 1-5
  max_depth: 5
  min_train_len: 5      # Curriculum filters lengths 5-300
  max_train_len: 300
```

**Relationship**:
- Depth 1 typically produces ~5-20 token sequences
- Depth 5 typically produces ~100-280 token sequences
- Curriculum bins progressively cover this range

### Arithmetic Example

```yaml
# configs/data/arithmetic.yaml
language: arithmetic
properties:
  min_val: 1
  max_val: 50
  min_depth: 1          # Generate trees from depth 1-4
  max_depth: 4
  min_train_len: 5      # Curriculum filters lengths 5-150
  max_train_len: 150
```

## How to Set Length Ranges

### Method 1: Use the Measurement Script (Recommended)

We provide a script that measures actual lengths for your exact configuration:

```bash
# For BFVP
python scripts/measure_lengths.py \
  --task bfvp \
  --min_depth 1 \
  --max_depth 5 \
  --num_vars 2 \
  --format trace \
  --samples 500

# For Arithmetic
python scripts/measure_lengths.py \
  --task arithmetic \
  --min_depth 1 \
  --max_depth 4 \
  --min_val 1 \
  --max_val 50 \
  --format trace \
  --samples 500
```

**Example output:**
```
BFVP Length Analysis (depth 1-5, num_vars=2, format=trace):

Per-Depth Statistics:
Depth    Min      Max      Avg      Median
1        5        11       7.2      5
2        13       30       21.0     20
3        29       72       49.3     47
4        75       146      107.2    106
5        194      289      230.0    229

Overall: Min=5, Max=289, Avg=81.0

Recommended Config:
  min_train_len: 3
  max_train_len: 317
```

This shows you **exactly** what lengths your configuration produces!

### Method 2: Conservative Estimates

If you can't run the analysis, use conservative ranges:

| Task | Depth Range | Estimated Length Range |
|------|-------------|------------------------|
| BFVP (num_vars=2) | 1-5 | 5-300 |
| BFVP (num_vars=3) | 1-5 | 5-400 |
| Arithmetic | 1-4 | 5-150 |
| Arithmetic | 1-6 | 5-300 |

Add buffer: Set `max_train_len` 10-20% higher than observed maximum.

## Training Examples

### BFVP with Curriculum

```bash
python main.py \
  data=bfvp \
  algo=ar \
  model=nano \
  data.properties.min_depth=1 \
  data.properties.max_depth=5 \
  curriculum.enabled=true \
  curriculum.num_bins=4 \
  curriculum.epochs_per_bin=5
```

**What happens:**
- Generates examples with depths 1-5 (full range)
- Curriculum bin 1: Filters to shortest ~25% (mostly depth 1-2)
- Curriculum bin 2: Filters to next 25% (mostly depth 2-3)
- Curriculum bin 3: Filters to next 25% (mostly depth 3-4)
- Curriculum bin 4: Filters to longest ~25% (mostly depth 4-5)

### Arithmetic with Curriculum

```bash
python main.py \
  data=arithmetic \
  algo=mdlm \
  model=nano \
  data.properties.min_depth=1 \
  data.properties.max_depth=4 \
  curriculum.enabled=true \
  curriculum.num_bins=3 \
  curriculum.epochs_per_bin=8
```

## Monitoring

During training, the callback logs which examples are included:

```
Curriculum Learning enabled with 4 bins
Length range: [5, 300]
Epochs per bin: 5
Bin boundaries: [(5, 78), (78, 151), (151, 224), (224, 300)]

Epoch 0: Training on length range [5, 78] (bin 1/4, epoch 1/5)
Filtered dataset: 12453/50000 examples in length range [5, 78]
```

This shows:
- 12,453 of 50,000 examples fall in length range [5, 78]
- These correspond mostly to depth 1-2 trees
- Model learns from these simpler examples first

## Benefits for Depth-Based Tasks

1. **Natural difficulty progression**: Shallower trees (easier) before deeper trees (harder)
2. **Structural understanding**: Learn basic operations before complex nesting
3. **Better gradient flow**: Start with shorter sequences where gradients propagate more easily
4. **Improved generalization**: Build understanding incrementally

## Comparison: With vs. Without Curriculum

### Without Curriculum (Standard Training)
```
Every epoch: Mix of all depths 1-5
- Some very simple (depth 1) examples
- Some very complex (depth 5) examples
- Model tries to learn everything simultaneously
```

### With Curriculum
```
Epochs 0-4:   Focus on depth 1-2 (via length filter)
Epochs 5-9:   Add depth 2-3
Epochs 10-14: Add depth 3-4
Epochs 15-19: Full range (depth 1-5)
- Model builds complexity understanding gradually
- Earlier epochs establish foundational patterns
```

## Troubleshooting

### Problem: "No examples found in length range [X, Y]"

**Cause**: Length range doesn't match your depth settings.

**Solution**:
1. Check your `min_depth` and `max_depth` settings
2. Verify `min_train_len` and `max_train_len` cover the produced range
3. Run empirical analysis (see Method 1 above)
4. Update config with correct ranges

### Problem: All examples in one bin

**Cause**: Length range too narrow or bins too wide.

**Solution**:
- Increase `max_depth` to generate more diverse lengths
- Increase `num_bins` to create finer-grained curriculum
- Check that `max_train_len` is set high enough

### Problem: Uneven bin distribution

**Cause**: Depth-length relationship is non-linear.

**Expected behavior**: This is normal! Bins are based on length quartiles, not depth quartiles.
- If depth 1-2 produce similar lengths, they'll be in the same bin
- If depth 4-5 have high variance, they'll spread across multiple bins
- This is actually beneficial - the curriculum adapts to actual complexity

## Implementation Details

The curriculum callback:
1. Reads `min_train_len` and `max_train_len` from config
2. Computes bin boundaries based on these ranges
3. For each example, counts non-padding tokens
4. Includes example if token count falls in current bin range
5. Never looks at the original depth - only final length matters

This design ensures:
- Works with any generation method (depth-based, length-based, etc.)
- Filters based on actual model input
- Handles format variations (trace vs. final_value) automatically

## Files Modified

1. **[configs/data/bfvp.yaml](configs/data/bfvp.yaml)**: Added `min_train_len` and `max_train_len`
2. **[configs/data/arithmetic.yaml](configs/data/arithmetic.yaml)**: Added `min_train_len` and `max_train_len`
3. **[curriculum_callback.py](curriculum_callback.py)**: Enhanced error messages for missing parameters
4. **[CURRICULUM_LEARNING.md](CURRICULUM_LEARNING.md)**: Added depth-based task documentation
5. **[examples/curriculum_example.sh](examples/curriculum_example.sh)**: Updated examples with depth parameters

## Summary

Curriculum learning works seamlessly with depth-based tasks by:
- Generating examples across full depth range
- Filtering by final sequence length (natural proxy for complexity)
- Leveraging depth-length correlation for progressive learning
- Requiring only `min_train_len` and `max_train_len` in config

No changes needed to generation code - the curriculum operates purely at the filtering stage!
