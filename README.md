# Computational Expressivity of Sequence Models

Research codebase for studying the computational expressivity of different sequence modeling approaches on formal languages and structured reasoning tasks.

## Overview

This repository implements and compares several sequence modeling paradigms on tasks requiring compositional reasoning:

### Models
* **Autoregressive (AR)** with chain-of-thought reasoning
* **Looping Transformers (LT)** with configurable depth functions
* **Masked Diffusion Language Models (MDLM)**
* **Discrete Diffusion baselines** (D3PM, SEDD)

### Tasks
* **Boolean Formula Value Problems (BFVP)**: Evaluation of boolean formulas with AND/OR/NOT
* **Arithmetic**: Expression evaluation with constraints
* **Finite State Automata**: Pattern recognition (parity, contains_a, mod_3, etc.)

## Getting Started

Create a conda environment with the required dependencies:

```bash
conda create -n duo python=3.12
conda activate duo
conda install nvidia/label/cuda-12.4.0::cuda-toolkit
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1
```

## Training

### Basic Training Command

```bash
python main.py data=<task> algo=<algorithm> model=<size> [overrides]
```

**Examples:**
```bash
# Train AR model with chain-of-thought on BFVP
python main.py data=bfvp algo=ar model=nano data.properties.format=trace

# Train looping transformer with logarithmic depth on parity
python main.py data=parity algo=lt model=ltnano algo.looping_type=log

# Train MDLM on arithmetic
python main.py data=arithmetic algo=mdlm model=nano
```

### Training via Slurm

Create log directory first:
```bash
mkdir watch_folder
```

Submit jobs with task and length parameters:
```bash
sbatch scripts/train_looping.sh <task> <min_train> <max_train> <min_val> <max_val> <min_test> <max_test> <model_length>
```

**Available scripts:**
- `train_classifier.sh` - LT with constant depth (single-pass classifier)
- `train_cot.sh` - AR with chain-of-thought reasoning
- `train_looping.sh` - LT with logarithmic depth
- `train_padded_looping.sh` - LT with padding strategies
- `train_mdm.sh` - MDLM baseline

**Example:**
```bash
# Train looping transformer on BFVP with increasing complexity
sbatch scripts/train_looping.sh bfvp 16 32 32 64 64 128 256
```

## Evaluation

```bash
# Compute perplexity on validation set
python main.py mode=ppl_eval eval.checkpoint_path=path/to/checkpoint.ckpt

# Generate samples and evaluate
python main.py mode=sample_eval eval.checkpoint_path=path/to/checkpoint.ckpt
```

## Configuration

The codebase uses Hydra for configuration management. Key configuration options:

### Algorithm Selection
- `algo=ar` - Autoregressive model
- `algo=lt` - Looping Transformer
  - `algo.looping_type=constant` - Single pass (classifier)
  - `algo.looping_type=linear` - Linear depth
  - `algo.looping_type=log` - Logarithmic depth
- `algo=mdlm` - Masked Diffusion Language Model
- `algo=d3pm` - D3PM discrete diffusion
- `algo=sedd` - SEDD discrete diffusion

### Data Format
- `data.properties.format=trace` - Chain-of-thought format (shows intermediate steps)
- `data.properties.format=final_value` - Direct input-output mapping

### Task Configuration

**BFVP:**
```bash
python main.py data=bfvp \
  data.properties.num_vars=5 \
  data.properties.max_depth=3 \
  data.properties.fan_in=2
```

**Arithmetic:**
```bash
python main.py data=arithmetic \
  data.properties.max_depth=3 \
  data.properties.min_val=0 \
  data.properties.max_val=100
```

**Length Ranges** (for testing generalization):
```bash
python main.py data=parity \
  data.properties.min_train_len=16 \
  data.properties.max_train_len=32 \
  data.properties.min_val_len=32 \
  data.properties.max_val_len=64
```

## Analysis

Generate plots from saved metrics:

```bash
python analyze_metrics.py --metrics_dir path/to/output/directory
```

This creates visualizations for:
- Accuracy vs. sequence length (generalization curves)
- Training curves with smoothing
- Comparative analysis across different runs

## Code Organization

- `main.py` - Entry point for training and evaluation
- `algo.py` - Algorithm implementations (AR, LT, MDLM, D3PM, SEDD)
- `trainer_base.py` - PyTorch Lightning training loop
- `dataloader.py` - Data loading and on-the-fly generation
- `models/` - Backbone architectures (DiT, Looping Transformer)
- `configs/` - Hydra configuration files
- `scripts/` - Slurm training scripts
- `bfvp.py` - Boolean formula generation
- `arithmetic.py` - Arithmetic expression generation
- `parity.py` - Parity task generation
- `analyze_metrics.py` - Metrics analysis and plotting

## Acknowledgements

This repository was built on top of [MDLM's Github repository](https://github.com/kuleshov-group/mdlm) and incorporates components from [The Diffusion Duality](https://arxiv.org/abs/2506.10892v1).
