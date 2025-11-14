# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for studying computational expressivity of different sequence modeling approaches on formal languages and structured tasks. Compares:

- **Autoregressive (AR)** models with chain-of-thought reasoning
- **Looping Transformers (LT)** with variable depth functions (constant/linear/log)
- **Masked Diffusion Language Models (MDLM)**
- **D3PM** and **SEDD** (discrete diffusion baselines)

Focus areas: Boolean Formula Value Problems (BFVP), arithmetic expression evaluation, finite state automata (FSAs), and parity.

## Quick Start

**Launch an experiment (recommended):**
```bash
./scripts/launch_smart.sh my_experiment bfvp parity arithmetic
```

This automatically:
1. Prepares all data formats in parallel
2. Launches all 8 algorithm variants per language
3. Creates analysis scripts

**See [QUICK_START.md](QUICK_START.md) for complete workflow.**

## Setup Commands

```bash
# Create environment
conda create -n duo python=3.12
conda activate duo
conda install nvidia/label/cuda-12.4.0::cuda-toolkit
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1
```

## Common Development Commands

### Training

All training uses Hydra configuration system via [main.py](main.py):

```bash
# Basic training command
python main.py data=<task> algo=<algorithm> model=<size> [overrides]

# Example: Train AR model on BFVP with chain-of-thought
python main.py data=bfvp algo=ar model=nano data.properties.format=trace

# Example: Train looping transformer with log depth
python main.py data=parity algo=lt model=nano algo.looping_type=log

# Prepare data only (no training)
python main.py mode=prepare_data data=bfvp model.length=256 data.properties.format=trace
```

**Via Slurm (recommended):**
```bash
mkdir watch_folder  # Create log directory first

# Launch full experiment (data prep + all algorithms)
./scripts/launch_smart.sh my_experiment bfvp parity arithmetic
```

See [QUICK_START.md](QUICK_START.md) for complete workflow.

**Available modes:**
- `mode=train` (default) - Full training
- `mode=prepare_data` - Only prepare and cache data
- `mode=ppl_eval` - Perplexity evaluation
- `mode=sample_eval` - Generate samples

Available scripts in [scripts/](scripts/):
- `train_classifier.sh` - LT with constant depth (single-pass classifier)
- `train_cot.sh` - AR with chain-of-thought (trace format)
- `train_looping.sh` - LT with logarithmic depth
- `train_padded_looping.sh` - LT with padding
- `train_mdm.sh` - MDLM baseline

### Evaluation

```bash
# Perplexity evaluation on validation set
python main.py mode=ppl_eval eval.checkpoint_path=path/to/checkpoint.ckpt

# Generate samples and compute metrics
python main.py mode=sample_eval eval.checkpoint_path=path/to/checkpoint.ckpt
```

### Curriculum Learning

Enable progressive training on longer sequences:

```bash
# Train with curriculum learning (4 bins, 5 epochs each)
python main.py data=bfvp algo=ar curriculum.enabled=true curriculum.num_bins=4 curriculum.epochs_per_bin=5

# Customize curriculum settings
python main.py data=parity algo=lt curriculum.enabled=true curriculum.num_bins=3 curriculum.epochs_per_bin=8 curriculum.overlap=0.3
```

Key parameters:
- `curriculum.enabled`: Enable/disable curriculum learning
- `curriculum.num_bins`: Number of length bins (default: 4)
- `curriculum.epochs_per_bin`: Epochs per bin (default: 5)
- `curriculum.overlap`: Overlap ratio between bins 0-1 (default: 0.2)
- `curriculum.min_examples_per_bin`: Minimum examples per bin (default: 512)
  - Bins with fewer examples are automatically skipped
- `curriculum.min_batches_per_epoch`: Minimum batches per epoch (default: 100)
  - If a bin has too few batches (but enough examples), the range is automatically expanded

### Analysis

Use [analyze_metrics.py](analyze_metrics.py) to generate plots from saved metrics:
```bash
python analyze_metrics.py --metrics_dir path/to/output/dir
```

## High-Level Architecture

### Core Components

**[main.py](main.py)** - Entry point with three modes:
- `train`: Training loop with PyTorch Lightning
- `ppl_eval`: Compute perplexity on validation/test set
- `sample_eval`: Generate samples and evaluate

**[algo.py](algo.py)** - Algorithm implementations:
- `AR`: Autoregressive model (greedy decoding)
- `LT`: Looping Transformer with configurable loop depth
- `MDLM`: Masked Diffusion Language Model
- `D3PMAbsorb`, `SEDDAbsorb`: Discrete diffusion baselines

Each algorithm implements:
- `nll()`: Negative log-likelihood computation
- `generate_conditioned()`: Conditional generation from prompts
- `_process_model_input()`: Input preprocessing for specific training objective

**[trainer_base.py](trainer_base.py)** - PyTorch Lightning module hierarchy:
- `TrainerBase`: Base class with common training logic
- `Diffusion`: Adds diffusion-specific sampling and timestep handling
- `AbsorbingState`: Mask-based diffusion (used by MDLM, D3PM, SEDD)

**[dataloader.py](dataloader.py)** - Data loading and generation:
- Generates formal language examples on-the-fly
- `MaskedFormalCollator`: Handles prompt-completion splits
- Uses `do_not_mask` tensor to protect prompt tokens during training

**Task generators**:
- [bfvp.py](bfvp.py): Boolean Formula Value Problems (AND/OR trees with NOT)
- [arithmetic.py](arithmetic.py): Arithmetic expression evaluation with constraints
- FSA tasks: Pattern recognition (contains_a, ab_star, mod_3, etc.)

**[models/](models/)** - Backbone architectures:
- [dit.py](models/dit.py): Diffusion Transformer (for diffusion models)
- [lt.py](models/lt.py): Looping Transformer with depth functions

### Configuration System

Hydra-based hierarchical configuration in [configs/](configs/):

- **[config.yaml](configs/config.yaml)**: Main config with training hyperparameters
- **configs/algo/**: Algorithm configs (ar, lt, mdlm, sedd, d3pm)
- **configs/data/**: Task configs (bfvp, arithmetic, parity, FSA variants)
- **configs/model/**: Model size configs (nano, small, etc.)

Key config parameters:
```yaml
# Algorithm selection
algo.name: ar / lt / mdlm / d3pm / sedd
algo.looping_type: constant / linear / log  # For LT only

# Data format
data.properties.format: trace / final_value
  # trace: Show intermediate computation (CoT)
  # final_value: Only input and final answer

# Length ranges for generalization testing
data.properties.min_train_len, max_train_len
data.properties.min_val_len, max_val_len
data.properties.min_test_len, max_test_len
```

Override via command line:
```bash
python main.py data=bfvp algo=lt algo.looping_type=log model.length=128
```

### Key Design Patterns

**Prompt-Completion Format**: Training sequences use delimiters:
- `#` separates prompt from completion region
- `|` marks hierarchical levels (for structured tasks)
- `do_not_mask` tensor protects prompt tokens

Example: `x1 x2 x3 # y1 | y2 | y3 | final`

**Format Types**:
1. **trace**: Full computation trace (chain-of-thought)
   - AR models trained on this format
   - Shows intermediate steps: `((x1 AND x2) | (partial_result) | final)`

2. **final_value**: Direct input-output mapping
   - LT models trained on this format
   - Only shows: `x1 x2 x3 # final_answer`

**Looping Transformer Depth Functions** ([algo.py:80-88](algo.py#L80-88)):
- `constant`: Single pass (depth=1) - acts as classifier
- `linear`: Depth proportional to sequence length
- `log`: Logarithmic depth - efficient iterative refinement

**Conditional Generation for Evaluation** ([trainer_base.py:287-383](trainer_base.py#L287-383)):
When `do_not_mask` is present in batch (formal language tasks):
- Automatically extracts prompts and targets
- Generates completions with different strategies
- Computes accuracy metrics:
  - `acc_exact`: Exact sequence match
  - `acc_token`: Token-level accuracy
  - `correct_prediction`: Accuracy on final answer token

**Generation Modes** (for diffusion models):
- `random`: Randomly unmask k positions per step
- `top_k`: Unmask highest-confidence positions
- `one_at_a_time`: Sequential left-to-right
- `one_level`: Unmask one hierarchical level (uses `|` delimiters)
- `all_at_once`: Single-step generation

**Masking Strategies** ([trainer_base.py:1056-1132](trainer_base.py#L1056-1132)):
1. **Standard**: Probabilistic masking based on noise schedule
2. **Ground truth**: Mask specific segment between `|` delimiters
   - Set `training.ground_truth_masking=true`
   - Useful for structured hierarchical tasks

### Task-Specific Details

**BFVP** (Boolean Formula Value Problems):
- Configurable via `num_vars`, `max_depth`, `fan_in`
- Generates AND/OR trees with random negations
- Variables: x1, x2, ..., xN
- Evaluates with random assignments

**Arithmetic**:
- Operations: +, -, *, /
- Constraint-driven generation ensures valid intermediate values
- Configurable via `max_depth`, `min_val`, `max_val`

**FSA Tasks**:
- Pattern recognition over alphabets
- Examples: parity, contains_a, ab_star, mod_3
- Configurable alphabet size and sequence length

**Format Impact**:
- AR + trace: Learns step-by-step reasoning
- LT + final_value: Learns direct mapping with iterative refinement
- MDLM: Uses masking, works with both formats

### Metrics and Logging

**Automatic metrics** (saved to JSON):
- `validation_metrics.json`: Per-epoch validation results
- `test_metrics.json`: Final test results
- Includes NLL, accuracy metrics, and training metadata

**Logged to W&B**:
- Loss curves, perplexity, accuracy metrics
- Generated samples (tables showing prompt → generated → target)
- Learning rate, gradient norms

**Analysis** via [analyze_metrics.py](analyze_metrics.py):
- Plots accuracy vs. sequence length (generalization)
- Training curves with smoothing
- Comparative analysis across runs

## Important Implementation Notes

**Gradient Accumulation**: Automatically computed by Lightning:
```python
accumulate_grad_batches = global_batch_size / (devices * batch_size * num_nodes)
```

**Checkpointing**:
- Saves to `{hydra_output_dir}/checkpoints/last.ckpt`
- Auto-resume if `checkpointing.resume_from_ckpt=true`
- Validation metrics saved alongside checkpoints

**Formal Language Tokenizer**: Special tokenizer includes:
- Logical operators: AND, OR, NOT
- Arithmetic: +, -, *, /
- Structural: #, |, (, )
- Variables and constants

**Length Handling**:
- `model.length`: Fixed sequence length for model
- Sequences padded/truncated to this length
- Train/val/test can have different length ranges for generalization testing
