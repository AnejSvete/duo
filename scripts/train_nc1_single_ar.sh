#!/bin/bash
#SBATCH -J nc1_ar_single                  # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --mem-per-cpu=32000                   # server memory requested (per node)
#SBATCH -t 24:00:00                  # Time limit (hh:mm:ss)
#SBATCH --gpus=rtx_3090:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

module load stack/2024-06 python/3.12.8 eth_proxy
source /cluster/home/asvete/duo/bin/activate

srun python -u -m main \
  wandb.name="ar-nc1-single-$(date +%Y%m%d-%H%M%S)" \
  data=formal \
  model=nano \
  algo=ar \
  data.formal.format=final_value 
