# Curriculum Learning Implementation Summary

## Files Added

### Core Implementation
1. **[curriculum_callback.py](curriculum_callback.py)** (225 lines)
   - Main implementation of `CurriculumLearningCallback`
   - Handles bin computation, dataset filtering, and state management
   - Supports checkpointing and resumption

2. **[configs/callbacks/curriculum_learning.yaml](configs/callbacks/curriculum_learning.yaml)** (7 lines)
   - Hydra configuration for the callback
   - Integrates with main config system

### Documentation
3. **[CURRICULUM_LEARNING.md](CURRICULUM_LEARNING.md)** (400+ lines)
   - Comprehensive user guide
   - Usage examples and best practices
   - Troubleshooting guide
   - Visual diagrams

4. **[examples/curriculum_example.sh](examples/curriculum_example.sh)** (60 lines)
   - Runnable examples for different tasks
   - Demonstrates various curriculum configurations

### Testing
5. **[test_curriculum.py](test_curriculum.py)** (150+ lines)
   - Unit tests for bin computation
   - State dict save/load tests
   - Configuration parsing tests

## Files Modified

1. **[configs/config.yaml](configs/config.yaml)**
   - Added `curriculum` section with 4 hyperparameters
   - Added callback to default callbacks list

2. **[README.md](README.md)**
   - Added "Curriculum Learning" section
   - Quick reference with link to detailed docs

3. **[CLAUDE.md](CLAUDE.md)**
   - Added curriculum learning section
   - Examples and parameter descriptions

## Configuration Parameters

```yaml
curriculum:
  enabled: false          # Enable/disable
  num_bins: 4            # Number of length bins
  epochs_per_bin: 5      # Epochs per bin
  overlap: 0.2           # Overlap ratio (0-1)
```

## Key Features

### 1. Automatic Bin Computation
- Divides length range `[min_train_len, max_train_len]` into `num_bins` bins
- Supports configurable overlap between consecutive bins
- Ensures last bin always reaches `max_train_len`

### 2. Dynamic Dataset Filtering
- Filters training data based on current bin's length range
- Counts non-padding tokens to determine sequence length
- Logs statistics about filtered dataset size

### 3. Stateful Training
- Tracks current bin and epochs within bin
- State saved/restored with model checkpoints
- Seamless resumption from interrupted training

### 4. Integration with PyTorch Lightning
- Implements Lightning's `Callback` interface
- Hooks into `on_train_epoch_start` for bin advancement
- Compatible with all existing training features

### 5. Flexible Configuration
- Hydra-based configuration system
- Override via command line or config files
- Easy to enable/disable without code changes

## Usage Examples

### Basic Usage
```bash
python main.py data=bfvp algo=ar \
  curriculum.enabled=true \
  curriculum.num_bins=4 \
  curriculum.epochs_per_bin=5
```

### Custom Settings
```bash
python main.py data=arithmetic algo=mdlm \
  curriculum.enabled=true \
  curriculum.num_bins=6 \
  curriculum.epochs_per_bin=3 \
  curriculum.overlap=0.4
```

### Disable
```bash
python main.py curriculum.enabled=false
```

## How It Works

1. **Setup Phase** (`setup` method):
   - Reads length range from config
   - Computes bin boundaries with overlap
   - Logs curriculum plan

2. **Training Loop** (`on_train_epoch_start` method):
   - Checks if time to advance to next bin
   - Filters dataset to current bin's length range
   - Creates new dataloader with filtered data
   - Logs progress

3. **State Management** (`state_dict`/`load_state_dict`):
   - Saves current bin, epochs count, and boundaries
   - Restores state on checkpoint load

## Technical Details

### Bin Boundary Algorithm
```python
total_range = max_train_len - min_train_len
base_bin_size = total_range / num_bins
overlap_size = base_bin_size * overlap

for i in range(num_bins):
    bin_start = min_train_len + i * base_bin_size
    bin_end = bin_start + base_bin_size + overlap_size
    # Clamp to valid range
```

### Length Filtering
```python
# Check attention mask
if 'attention_mask' in example:
    seq_len = example['attention_mask'].sum().item()
# Fallback to counting non-pad tokens
elif 'input_ids' in example:
    seq_len = (input_ids != pad_token_id).sum().item()

# Include if in range
if min_len <= seq_len <= max_len:
    filtered_indices.append(idx)
```

## Testing

Run test suite:
```bash
python test_curriculum.py
```

Tests verify:
- Bin boundaries computed correctly
- Overlap between consecutive bins
- Single bin covers full range
- State dict save/load works
- Configuration parsing works

## Performance Considerations

### Overhead
- Minimal: bin computation done once at setup
- Dataset filtering: O(n) per epoch advancement
- Dataloader recreation: negligible for most datasets

### Memory
- No additional memory overhead
- Uses `torch.utils.data.Subset` for filtering (views, not copies)

### Disk I/O
- No additional disk I/O
- Works with existing cached datasets

## Future Enhancements

Possible extensions (not implemented):

1. **Dynamic epochs_per_bin**: Adjust based on validation metrics
2. **Non-linear bin sizes**: E.g., logarithmic spacing
3. **Reverse curriculum**: Start with hard examples
4. **Multi-dimensional curriculum**: Length + other properties
5. **Warm restart**: Reduce LR when advancing bins
6. **Mixed sampling**: Gradually blend in longer sequences

## Comparison with Standard Training

| Aspect | Standard | Curriculum |
|--------|----------|------------|
| Examples per epoch | All lengths | Current bin only |
| Training time | Baseline | ~Same (fewer examples per epoch, but more epochs) |
| Convergence | May be slower | Often faster on long sequences |
| Generalization | Good | Often better |
| Hyperparameters | Fewer | 4 additional |
| Complexity | Simple | Moderate |

## Citation

If you use this implementation, consider citing:

```
Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009).
Curriculum learning. In Proceedings of the 26th annual international
conference on machine learning (pp. 41-48).
```

## Support

For issues or questions:
1. Check [CURRICULUM_LEARNING.md](CURRICULUM_LEARNING.md) documentation
2. Review [examples/curriculum_example.sh](examples/curriculum_example.sh)
3. Run [test_curriculum.py](test_curriculum.py) to verify setup
4. Check logs for filtering statistics
