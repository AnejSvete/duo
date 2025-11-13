# Curriculum Learning

This document explains how to use the curriculum learning feature for progressive length-based training.

## Overview

Curriculum learning progressively trains models on increasingly difficult examples. In this implementation, we define "difficulty" as sequence length - the model first trains on shorter sequences and gradually moves to longer ones.

The training data is divided into **bins** based on sequence length, and the model trains on each bin for a specified number of epochs before advancing to the next bin. Bins can optionally **overlap** to ensure smooth transitions between difficulty levels.

### Visual Example

Here's how curriculum learning divides the training:

```
Without Curriculum (Standard Training):
├────────────────────────────────────────┤
│   All lengths [16-64] every epoch      │
└────────────────────────────────────────┘

With Curriculum (4 bins, 20% overlap):
Epoch 0-4:   ├──────┤
             │ 16-28│
             └──────┘
Epoch 5-9:           ├──────┤
                     │ 28-40│
                     └──────┘
Epoch 10-14:                 ├──────┤
                             │ 40-52│
                             └──────┘
Epoch 15-19:                         ├──────┤
                                     │ 52-64│
                                     └──────┘
             ├────────────────────────────┤
             16                          64
             (Sequence Length)
```

Note the overlap between consecutive bins, providing smooth transitions.

## Benefits

- **Faster convergence**: Starting with simpler (shorter) examples helps the model learn basic patterns more quickly
- **Better generalization**: Progressive exposure to longer sequences helps the model scale its learned patterns
- **More stable training**: Gradual difficulty increase can reduce training instability
- **Improved sample efficiency**: The model learns from easier examples first, building a foundation for harder ones

## Configuration

Curriculum learning is controlled by the `curriculum` section in [configs/config.yaml](configs/config.yaml):

```yaml
curriculum:
  enabled: false  # Enable curriculum learning
  num_bins: 4  # Number of length bins to divide training into
  epochs_per_bin: 5  # Number of epochs to train on each bin
  overlap: 0.2  # Overlap ratio between consecutive bins (0-1)
```

### Parameters

- **`enabled`** (bool): Whether to enable curriculum learning. Default: `false`
- **`num_bins`** (int): Number of bins to divide the length range into. Default: `4`
- **`epochs_per_bin`** (int): How many epochs to train on each bin before advancing. Default: `5`
- **`overlap`** (float): Overlap ratio between consecutive bins (0.0 to 1.0). Default: `0.2`
  - 0.0 = no overlap (adjacent bins)
  - 0.2 = 20% overlap between consecutive bins
  - Higher values create smoother transitions

The length range is determined by the data configuration:
- `data.properties.min_train_len`: Minimum sequence length (in tokens)
- `data.properties.max_train_len`: Maximum sequence length (in tokens)

### Important Note on Depth-Based Tasks

For tasks like **BFVP** and **arithmetic** that generate examples based on tree depth (not directly by length):
- The curriculum still operates on **sequence length** (number of tokens), not depth
- `min_train_len` and `max_train_len` should cover the range of lengths produced by your depth settings
- Example: BFVP with `min_depth=1, max_depth=5` produces sequences from ~5 to ~280 tokens
- The curriculum will filter examples by their actual tokenized length
- This means shallower trees (which tend to produce shorter sequences) are learned first

**Config examples:**
- **BFVP**: `min_train_len: 5, max_train_len: 300` (for depth 1-5)
- **Arithmetic**: `min_train_len: 5, max_train_len: 150` (for depth 1-4)
- **FSA tasks**: Use the explicit length ranges from data generation

## Usage Examples

### Example 1: Basic Curriculum Learning

Train on BFVP with 4 bins, 5 epochs per bin, 20% overlap:

```bash
python main.py \
  data=bfvp \
  algo=ar \
  model=nano \
  curriculum.enabled=true \
  curriculum.num_bins=4 \
  curriculum.epochs_per_bin=5 \
  curriculum.overlap=0.2
```

**What happens:**
- Length range: [16, 24] (from data config)
- Bin 1: [16, 18] - trains for 5 epochs
- Bin 2: [18, 20] - trains for 5 epochs
- Bin 3: [20, 22] - trains for 5 epochs
- Bin 4: [22, 24] - trains for 5 epochs
- Total: 20 epochs

### Example 2: Aggressive Curriculum (No Overlap)

For discrete difficulty levels without overlap:

```bash
python main.py \
  data=parity \
  algo=lt \
  algo.looping_type=log \
  model=nano \
  curriculum.enabled=true \
  curriculum.num_bins=3 \
  curriculum.epochs_per_bin=10 \
  curriculum.overlap=0.0
```

### Example 3: Smooth Curriculum (High Overlap)

For very gradual transitions with high overlap:

```bash
python main.py \
  data=arithmetic \
  algo=mdlm \
  model=nano \
  curriculum.enabled=true \
  curriculum.num_bins=5 \
  curriculum.epochs_per_bin=4 \
  curriculum.overlap=0.5
```

### Example 4: Quick Curriculum

For fast experimentation with fewer bins:

```bash
python main.py \
  data=bfvp \
  algo=ar \
  model=nano \
  curriculum.enabled=true \
  curriculum.num_bins=2 \
  curriculum.epochs_per_bin=3 \
  curriculum.overlap=0.1
```

## How It Works

### Bin Computation

Given a length range `[min_len, max_len]` and `num_bins`, the callback:

1. Calculates base bin size: `(max_len - min_len) / num_bins`
2. Computes overlap size: `base_bin_size * overlap`
3. Creates bins with computed boundaries

**Example:** Range [16, 64], 4 bins, 20% overlap:
- Base bin size: (64 - 16) / 4 = 12
- Overlap size: 12 * 0.2 = 2.4
- Bin 1: [16, 28] (size 12 + 2.4 = 14.4)
- Bin 2: [28, 40]
- Bin 3: [40, 52]
- Bin 4: [52, 64]

### Training Flow

At the start of each epoch:
1. The callback checks if it's time to advance to the next bin
2. If yes, increments `current_bin` and resets `epochs_in_current_bin`
3. Filters the training dataset to only include examples in the current bin's length range
4. Creates a new dataloader with the filtered dataset
5. Training proceeds normally on the filtered data

### Length Calculation

The callback determines sequence length by:
1. First checking the `attention_mask` (counts non-padding tokens)
2. Fallback to counting `input_ids` that aren't padding tokens
3. Includes the example if length falls within `[min_len, max_len]` for current bin

**For depth-based tasks (BFVP, Arithmetic):**
- Examples are generated with random depths from `[min_depth, max_depth]`
- Each tree of depth D is converted to a token sequence
- The curriculum filters based on final token count, not original depth
- Generally: deeper trees → longer sequences (but with variation)
- This provides a natural correlation between depth and curriculum progression

## Monitoring

The callback logs detailed information during training:

```
Curriculum Learning enabled with 4 bins
Length range: [16, 64]
Epochs per bin: 5
Overlap ratio: 0.2
Bin boundaries: [(16, 28), (28, 40), (40, 52), (52, 64)]

Epoch 0: Training on length range [16, 28] (bin 1/4, epoch 1/5)
Filtered dataset: 12453/50000 examples in length range [16, 28]

Epoch 5: Training on length range [28, 40] (bin 2/4, epoch 1/5)
Filtered dataset: 13821/50000 examples in length range [28, 40]
...
```

## Checkpointing

The callback state (current bin, epochs in bin, bin boundaries) is automatically saved with model checkpoints. If training is resumed, it will continue from the same curriculum position.

## Best Practices

### Choosing `num_bins`

- **Fewer bins (2-3)**: Faster training, larger jumps in difficulty
- **More bins (4-6)**: Smoother progression, more gradual learning
- Consider your total training budget and `epochs_per_bin`

### Choosing `epochs_per_bin`

- Should be enough for the model to adapt to each difficulty level
- Monitor validation accuracy to see if the model has plateaued before advancing
- Typical values: 3-10 epochs per bin

### Choosing `overlap`

- **No overlap (0.0)**: Clear difficulty boundaries, faster training
- **Light overlap (0.1-0.3)**: Smooth transitions, recommended default
- **Heavy overlap (0.4-0.6)**: Very gradual transitions, may be redundant

### Tuning for Your Task

Different tasks may benefit from different curriculum settings:

- **BFVP**: Moderate curriculum (4 bins, 5 epochs, 0.2 overlap)
- **Arithmetic**: Aggressive curriculum (5+ bins, shorter expressions are much easier)
- **FSA tasks**: Light curriculum (2-3 bins, patterns scale more uniformly)

### Combining with Other Techniques

Curriculum learning works well with:
- **Chain-of-thought (trace format)**: Learn reasoning on simple examples first
- **Looping Transformers**: Progressive depth adjustment alongside length curriculum
- **Learning rate schedules**: Consider warmup for each new bin

## Implementation Details

The curriculum learning feature consists of:

1. **[curriculum_callback.py](curriculum_callback.py)**: Main implementation
   - `CurriculumLearningCallback`: PyTorch Lightning callback
   - Handles bin computation, dataset filtering, and state management

2. **[configs/callbacks/curriculum_learning.yaml](configs/callbacks/curriculum_learning.yaml)**: Callback configuration
   - Integrates with Hydra config system
   - Automatically included in default callbacks

3. **[configs/config.yaml](configs/config.yaml)**: Global configuration
   - `curriculum` section with all parameters
   - Callback included in default callback list

## Troubleshooting

### No examples in current bin

If you see warnings like:
```
No examples found in length range [X, Y]. Keeping full dataset.
```

**Cause**: The bin range doesn't match any examples in your dataset.

**Solution**:
- Check your `min_train_len` and `max_train_len` settings
- For BFVP/arithmetic: Ensure length range covers sequences produced by your depth settings
- Reduce `num_bins` to create wider bins
- Increase `overlap` to ensure bins capture more examples

**To determine correct length ranges for BFVP/arithmetic:**
```bash
# Quick analysis of actual sequence lengths
python -c "
import random, bfvp
lengths = []
for _ in range(100):
    tree = bfvp.generate_formula_tree(random.randint(1, 5), 2)
    vars = bfvp.get_variables_from_tree(tree)
    assigns = {v: random.choice([True, False]) for v in vars}
    sub = bfvp.substitute_vars_in_tree(tree, assigns)
    text = bfvp._generate_text_from_tree(sub, tree, assigns, 'trace')
    lengths.append(len(text.split()))
print(f'Min: {min(lengths)}, Max: {max(lengths)}')
"
```

### Training seems slow

**Cause**: Filtering and recreating dataloaders has some overhead.

**Solution**:
- Use fewer bins (2-3 instead of 5-6)
- Increase `epochs_per_bin` to reduce the number of dataloader recreations

### Model forgets earlier bins

**Cause**: Catastrophic forgetting - model only sees later bins for many epochs.

**Solution**:
- Increase `overlap` to maintain exposure to shorter sequences
- Reduce `epochs_per_bin` to progress through curriculum faster
- Consider a "review" phase after curriculum (train on full range)

## Disabling Curriculum Learning

To disable curriculum learning, simply set:

```yaml
curriculum:
  enabled: false
```

Or via command line:
```bash
python main.py curriculum.enabled=false
```

The callback will be inactive and training will proceed normally on the full dataset.

## Testing

Run the test suite to verify the implementation:

```bash
python test_curriculum.py
```

This validates:
- Bin boundary computation
- State saving/loading
- Configuration parsing
