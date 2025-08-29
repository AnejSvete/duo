#!/bin/bash
#SBATCH -J empty_padded_looping                  # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --mem-per-cpu=32000                   # server memory requested (per node)
#SBATCH -t 24:00:00                  # Time limit (hh:mm:ss)
#SBATCH --gpus=rtx_3090:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

module load stack/2024-06 python/3.12.8 eth_proxy
source /cluster/home/asvete/duo/bin/activate

TASK=$1

LENGTH=${2:-96}

srun python -u -m main \
  wandb.name="$TASK-empty-padded-looping-$(date +%Y%m%d-%H%M%S)" \
  data=$TASK \
  model=ltnano \
  algo=lt \
  algo.looping_type=log \
  model.length=$LENGTH \
  data.properties.format=empty_trace
