# How to Set Length Ranges for Curriculum Learning

## The Problem

For tasks like BFVP and arithmetic, examples are generated using **tree depth** parameters (`min_depth`, `max_depth`), but curriculum learning filters by **sequence length** (number of tokens).

**The relationship is NOT deterministic:**
- Same depth can produce different lengths
- Depends on: format (trace vs final_value), operators, branching, negations, etc.

## The Solution: Measure, Don't Guess

### Step 1: Run the Measurement Script

We provide [scripts/measure_lengths.py](scripts/measure_lengths.py) to measure actual lengths:

```bash
# Example: Measure BFVP with your exact settings
python scripts/measure_lengths.py \
  --task bfvp \
  --min_depth 1 \
  --max_depth 5 \
  --num_vars 2 \
  --format trace \
  --samples 500
```

### Step 2: Interpret the Output

The script shows:

```
Per-Depth Statistics:
Depth    Count    Min      Max      Avg      Median
1        92       5        11       7.2      5
2        102      13       30       21.0     20
3        109      29       72       49.3     47
4        106      75       146      107.2    106
5        91       194      289      230.0    229

Recommended Config:
  min_train_len: 3
  max_train_len: 317
```

**Key observations:**
- Depth 1 produces 5-11 tokens (very short)
- Depth 5 produces 194-289 tokens (very long)
- There's significant **variation** within each depth
- This is why curriculum based on actual length works well!

### Step 3: Update Your Config

Add the recommended values to your data config:

```yaml
# configs/data/bfvp.yaml
properties:
  min_depth: 1
  max_depth: 5
  min_train_len: 3    # From measurement
  max_train_len: 317  # From measurement
```

## Why This Approach Works

### Correlation Between Depth and Length

From the measurements above, we see a strong correlation:

| Depth | Typical Length Range | Curriculum Bin |
|-------|---------------------|----------------|
| 1 | 5-11 tokens | Bin 1 (shortest) |
| 2 | 13-30 tokens | Bin 1-2 |
| 3 | 29-72 tokens | Bin 2-3 |
| 4 | 75-146 tokens | Bin 3-4 |
| 5 | 194-289 tokens | Bin 4 (longest) |

**With 4 curriculum bins on range [3, 317]:**
- Bin 1 [3-81]: Mostly depth 1-2 trees
- Bin 2 [81-159]: Mostly depth 3-4 trees
- Bin 3 [159-237]: Mostly depth 4-5 trees
- Bin 4 [237-317]: Mostly depth 5 trees

The curriculum naturally progresses from shallow to deep!

### Why Length Is Better Than Depth

Filtering by depth would be simpler, but filtering by length is better because:

1. **Length = actual difficulty**: Longer sequences are harder for models to process
2. **Gradient propagation**: Shorter sequences have better gradient flow
3. **Memory efficiency**: Start with smaller batches (short sequences)
4. **Format-agnostic**: Works for both `trace` and `final_value` formats
5. **Handles variance**: A complex depth-3 tree (long) comes before a simple depth-4 tree (short)

## Common Scenarios

### Scenario 1: Standard BFVP Setup

```bash
python scripts/measure_lengths.py --task bfvp --min_depth 1 --max_depth 5 --num_vars 2 --format trace

# Result: min_train_len=3, max_train_len=317
```

### Scenario 2: Deeper BFVP Trees

```bash
python scripts/measure_lengths.py --task bfvp --min_depth 1 --max_depth 7 --num_vars 2 --format trace

# Expected: Much larger max_train_len (potentially 500+)
```

### Scenario 3: More Variables

```bash
python scripts/measure_lengths.py --task bfvp --min_depth 1 --max_depth 5 --num_vars 3 --format trace

# Expected: Larger lengths due to more variables in expressions
```

### Scenario 4: Final Value Format

```bash
python scripts/measure_lengths.py --task bfvp --min_depth 1 --max_depth 5 --num_vars 2 --format final_value

# Expected: MUCH shorter! No intermediate steps shown
```

### Scenario 5: Arithmetic

```bash
python scripts/measure_lengths.py --task arithmetic --min_depth 1 --max_depth 4 --min_val 1 --max_val 50 --format trace

# Slower to run (constraint-based generation)
```

## What If I Can't Run the Script?

If you can't run the measurement script (e.g., no conda environment), use these conservative estimates:

| Task | Depth | Format | Estimated Range |
|------|-------|--------|-----------------|
| BFVP | 1-5 | trace | 5-350 |
| BFVP | 1-5 | final_value | 3-50 |
| BFVP | 1-7 | trace | 5-600 |
| Arithmetic | 1-4 | trace | 5-150 |
| Arithmetic | 1-6 | trace | 5-350 |

**Important**: These are rough estimates! Actual ranges depend on:
- `num_vars` (BFVP)
- `min_val`, `max_val` (arithmetic)
- Random generation variance

It's **much better** to measure!

## Troubleshooting

### Issue: Script says arithmetic is slow

**This is normal!** Arithmetic uses constraint-based generation:
- Must ensure all operations produce valid results
- May retry generation multiple times
- 500 samples can take 30-60 seconds

**Solution**: Just wait, or reduce `--samples 100` for faster results.

### Issue: Want to measure after changing config

**Always re-measure** when you change:
- `min_depth` or `max_depth`
- `num_vars` (BFVP)
- `min_val` or `max_val` (arithmetic)
- `format` (trace vs final_value)

These all affect sequence length!

### Issue: Too many examples filtered out

If curriculum logs show:
```
Filtered dataset: 250/50000 examples in length range [X, Y]
```

This means your bins might be too narrow. Solutions:
- Reduce `curriculum.num_bins` (fewer, wider bins)
- Increase `curriculum.overlap` (more examples per bin)
- Re-run measurement script to verify your length ranges

## Example: Complete Workflow

```bash
# 1. Decide on your task parameters
TASK=bfvp
MIN_DEPTH=1
MAX_DEPTH=5
NUM_VARS=2
FORMAT=trace

# 2. Measure actual lengths
python scripts/measure_lengths.py \
  --task $TASK \
  --min_depth $MIN_DEPTH \
  --max_depth $MAX_DEPTH \
  --num_vars $NUM_VARS \
  --format $FORMAT \
  --samples 500

# 3. Note the recommended values (e.g., min=3, max=317)

# 4. Update config
cat >> configs/data/bfvp.yaml << EOF
  min_train_len: 3    # From measurement script
  max_train_len: 317  # From measurement script
EOF

# 5. Train with curriculum
python main.py \
  data=bfvp \
  algo=ar \
  model=nano \
  curriculum.enabled=true \
  curriculum.num_bins=4 \
  curriculum.epochs_per_bin=5

# 6. Monitor logs for filtering stats
# Look for: "Filtered dataset: X/Y examples in length range [A, B]"
```

## Summary

**Don't guess length ranges!** The measurement script:
- ✅ Measures your exact configuration
- ✅ Shows per-depth statistics
- ✅ Provides recommended ranges
- ✅ Reveals the depth-length correlation
- ✅ Takes < 1 minute to run

**Always measure when:**
- Setting up curriculum for the first time
- Changing task parameters
- Switching between `trace` and `final_value` formats
- Debugging curriculum filtering issues

This ensures your curriculum learning works optimally!
