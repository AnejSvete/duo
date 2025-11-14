# Ground Truth Masking Implementation

## Overview

This document describes the level-based ground truth masking implementation in the `AbsorbingState` diffusion class. This feature is designed for hierarchical structured tasks like Boolean Formula Value Problems (BFVP) and arithmetic expression evaluation, where computation proceeds in discrete levels.

## Motivation

In structured formal language tasks, sequences are organized hierarchically with intermediate computation steps separated by `|` delimiters. For example:

```
b b b # 32 32 31 24 | 48 16 23 44 | 0 7 24 55 | 7 51 | 18
```

Here:
- The input is before the `#` delimiter
- After `#`, the computation proceeds in levels separated by `|`
- Each level represents a stage of the hierarchical computation

The standard continuous-time diffusion masking (based on noise schedule α(t)) doesn't respect this structure. Ground truth masking addresses this by:
1. Masking entire levels at a time (respecting the hierarchical structure)
2. Using discrete timesteps based on the number of levels
3. Masking from right to left (reverse order of computation)

## Configuration

Enable ground truth masking in your config:

```yaml
training:
  ground_truth_masking: true
```

You can set this in:
- `configs/config.yaml` (global default)
- Command line: `python main.py training.ground_truth_masking=true`

## Implementation Details

### Level-Based Masking

For a sequence like:
```
# A | B | C | D
```

There are 3 pipes (`|`), representing 3 levels to be unmasked progressively:

**Timestep 1 (mask 1 level):**
```
# A | B | C | [MASK D]
```

**Timestep 2 (mask 2 levels):**
```
# A | B | [MASK C] [MASK D]
```

**Timestep 3 (mask 3 levels):**
```
# A | [MASK B] [MASK C] [MASK D]
```

Note: The pipe delimiter is included in the masking.

### Discrete Timestep Sampling

Unlike standard diffusion which samples timesteps continuously from [0, 1], ground truth masking:

1. **Counts the number of levels** (num_levels = number of pipes in the completion region)
2. **Samples uniformly** from {1, 2, ..., num_levels}
3. **Masks k levels from the right** where k is the sampled value

This gives **at most ceil(log(N)) discrete timesteps** for a sequence of length N in hierarchical tasks.

### Loss Weighting

The loss coefficient `dalpha_t` is adjusted for discrete sampling:

```python
# Standard continuous sampling: dalpha_t from noise schedule
# Ground truth masking: scale by level weights
level_weights = 1.0 / num_levels  # Uniform probability over discrete levels
dalpha_t = dalpha_t * level_weights
```

This ensures proper normalization when integrating over discrete timesteps instead of continuous time.

### Timestep to Noise Schedule Mapping

After sampling the discrete level, we map it back to the continuous noise schedule:

```python
# Calculate mask ratio from masked tokens
mask_ratio = (num_masked_tokens / num_maskable_tokens)

# Map to timestep t ∈ [0, 1]
t = mask_ratio.clamp(min=1.0 / T if T > 0 else 1e-6)

# Get noise schedule parameters
dalpha_t, alpha_t = self.noise(t)
```

This allows the model to still use continuous noise schedules (log-linear, cosine, etc.) while respecting the discrete level structure.

## Code Changes

### `diffusion.py` - `AbsorbingState.q_xt()`

The main masking logic:

```python
def q_xt(self, x, alpha_t, do_not_mask, ground_truth_masking):
    if not ground_truth_masking:
        # Standard probabilistic masking
        ...
    else:
        # Level-based masking
        for i in range(batch_size):
            # Find all pipe positions
            pipe_indices = (x[i] == pipe_token_id).nonzero(as_tuple=True)[0]
            valid_pipe_indices = pipe_indices[~do_not_mask[i][pipe_indices]]

            num_levels = len(valid_pipe_indices)

            # Sample discrete timestep
            levels_to_mask = torch.randint(1, num_levels + 1, (1,)).item()

            # Mask from the appropriate pipe position to the end
            ...
```

### `diffusion.py` - `Diffusion.nll()`

Updated to handle discrete timesteps:

```python
def nll(self, x0, output_tokens, do_not_mask, ...):
    if ground_truth_masking:
        # Create masked sequence (determines discrete level)
        xt, num_levels = self.q_xt(...)

        # Derive continuous t from mask ratio
        mask_ratio = (mask_counts / num_maskable_tokens)
        t = mask_ratio.clamp(...)

        # Get noise schedule parameters
        dalpha_t, alpha_t = self.noise(t)

        # Adjust for discrete sampling
        level_weights = 1.0 / num_levels.float()
        dalpha_t = dalpha_t * level_weights
```

## Usage Example

### Training with Ground Truth Masking

```bash
# BFVP with trace format
python main.py \
    data=bfvp \
    algo=mdlm \
    data.properties.format=trace \
    training.ground_truth_masking=true

# Arithmetic with trace format
python main.py \
    data=arithmetic \
    algo=mdlm \
    data.properties.format=trace \
    training.ground_truth_masking=true
```

### Monitoring

Ground truth masking adds these W&B metrics:

- `diffusion/ground_truth_masking`: Flag (1.0 when enabled)
- `diffusion/num_levels_mean`: Average number of levels per sequence
- `diffusion/num_levels_std`: Standard deviation of levels
- `diffusion/mask_ratio`: Overall masking ratio
- `diffusion/t_mean`: Mean timestep value (should reflect discrete levels)

## Testing

A test script is provided to verify the implementation:

```bash
python test_ground_truth_masking.py
```

This will show example sequences and the masking patterns at each discrete timestep.

## Expected Behavior

### Sequence Structure Requirements

Ground truth masking expects sequences in the format:

```
[input tokens] # [level_1] | [level_2] | ... | [level_k]
```

Where:
- `#` separates input from completion
- `|` separates hierarchical levels
- `do_not_mask` protects the input region (before and including `#`)

### When to Use

**Use ground truth masking when:**
- Task has hierarchical structure (BFVP, arithmetic with traces)
- Computation proceeds in discrete levels
- You want the model to learn level-by-level generation

**Don't use ground truth masking when:**
- Task has no hierarchical structure
- Using `final_value` format (no intermediate steps)
- Sequence has no `|` delimiters

### Compatibility

- **Compatible with:** All diffusion-based algorithms (MDLM, D3PM, SEDD)
- **Data formats:** `trace` format (with `|` delimiters)
- **Noise schedules:** All noise schedules (log-linear, cosine, etc.)
- **Not compatible with:** Autoregressive (AR) or Looping Transformer (LT) algorithms

## Performance Considerations

### Advantages

1. **Respects hierarchical structure** - Model learns to generate level-by-level
2. **Fewer effective timesteps** - O(log N) instead of O(N) for continuous masking
3. **Cleaner training signal** - Each timestep corresponds to a meaningful computation level

### Limitations

1. **Requires structured data** - Only works with `|` delimited sequences
2. **Different batch sequences may have different numbers of levels** - Handled by per-sequence level counting
3. **Loss weighting needs careful tuning** - The `level_weights` adjustment may need refinement

## Future Work

Potential improvements:

1. **Adaptive weighting** - Weight levels by their complexity (length, depth in tree)
2. **Partial level masking** - Allow masking partial levels for finer-grained control
3. **Bidirectional masking** - Support masking from left-to-right or both directions
4. **Multi-task learning** - Mix ground truth and standard masking in the same batch

## References

- CLAUDE.md - Project overview and setup instructions
- configs/config.yaml - Configuration options
- diffusion.py - Implementation details
- test_ground_truth_masking.py - Test script
