#!/bin/bash
# Examples of using curriculum learning for different tasks

# Example 1: BFVP with chain-of-thought and curriculum learning
# Trains on 4 bins of increasing length, 5 epochs per bin
# Note: BFVP uses tree depth for generation, but curriculum filters by sequence length
echo "Example 1: BFVP with curriculum learning"
python main.py \
  data=bfvp \
  algo=ar \
  model=nano \
  data.properties.format=trace \
  data.properties.min_depth=1 \
  data.properties.max_depth=5 \
  curriculum.enabled=true \
  curriculum.num_bins=4 \
  curriculum.epochs_per_bin=5 \
  curriculum.overlap=0.2 \
  trainer.max_steps=10000

# Example 2: Looping Transformer on parity with aggressive curriculum
# No overlap between bins for clear difficulty levels
echo "Example 2: Looping Transformer with aggressive curriculum"
python main.py \
  data=parity \
  algo=lt \
  algo.looping_type=log \
  model=nano \
  data.properties.min_train_len=8 \
  data.properties.max_train_len=32 \
  curriculum.enabled=true \
  curriculum.num_bins=3 \
  curriculum.epochs_per_bin=8 \
  curriculum.overlap=0.0 \
  trainer.max_steps=8000

# Example 3: MDLM on arithmetic with smooth curriculum
# High overlap for very gradual transitions
# Note: Arithmetic uses tree depth for generation, curriculum filters by sequence length
echo "Example 3: MDLM with smooth curriculum"
python main.py \
  data=arithmetic \
  algo=mdlm \
  model=nano \
  data.properties.min_depth=1 \
  data.properties.max_depth=4 \
  curriculum.enabled=true \
  curriculum.num_bins=5 \
  curriculum.epochs_per_bin=4 \
  curriculum.overlap=0.4 \
  trainer.max_steps=12000

# Example 4: Quick curriculum for fast experimentation
# Just 2 bins with short training per bin
echo "Example 4: Quick curriculum for experimentation"
python main.py \
  data=bfvp \
  algo=ar \
  model=nano \
  data.properties.min_depth=1 \
  data.properties.max_depth=3 \
  curriculum.enabled=true \
  curriculum.num_bins=2 \
  curriculum.epochs_per_bin=3 \
  curriculum.overlap=0.1 \
  trainer.max_steps=3000

# Example 5: Disable curriculum (baseline comparison)
echo "Example 5: Baseline without curriculum"
python main.py \
  data=bfvp \
  algo=ar \
  model=nano \
  data.properties.min_depth=1 \
  data.properties.max_depth=5 \
  curriculum.enabled=false \
  trainer.max_steps=10000
