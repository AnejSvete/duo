#!/bin/bash
#SBATCH -J padded_looping                  # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --mem-per-cpu=32000                   # server memory requested (per node)
#SBATCH -t 24:00:00                  # Time limit (hh:mm:ss)
#SBATCH --gpus=rtx_3090:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

module load stack/2024-06 python/3.12.8 eth_proxy
source /cluster/home/asvete/duo/bin/activate

TASK=$1
MIN_TRAIN_LENGTH=$2
MAX_TRAIN_LENGTH=$3
MIN_VAL_LENGTH=$4
MAX_VAL_LENGTH=$5
MIN_TEST_LENGTH=$6
MAX_TEST_LENGTH=$7
MODEL_LENGTH=$8

srun python -u -m main \
  wandb.name="$TASK-padded-looping-$(date +%Y%m%d-%H%M%S)" \
  data=$TASK \
  model=ltnano \
  algo=lt \
  algo.looping_type=log \
  model.length=$MODEL_LENGTH \
  data.properties.format=trace \
  data.properties.min_train_len=$MIN_TRAIN_LENGTH \
  data.properties.max_train_len=$MAX_TRAIN_LENGTH \
  data.properties.min_val_len=$MIN_VAL_LENGTH \
  data.properties.max_val_len=$MAX_VAL_LENGTH \
  data.properties.min_test_len=$MIN_TEST_LENGTH \
  data.properties.max_test_len=$MAX_TEST_LENGTH
