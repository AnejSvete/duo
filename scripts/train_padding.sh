#!/bin/bash
#SBATCH -J padding                 # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --mem-per-cpu=32000                   # server memory requested (per node)
#SBATCH -t 04:00:00                  # Time limit (hh:mm:ss)
#SBATCH --gpus=rtx_3090:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

module load stack/2024-06 python/3.12.8 eth_proxy
source /cluster/home/asvete/duo/bin/activate

# Accept params from env vars (set by launch_smart.sh) or command line args
LANGUAGE=${LANGUAGE:-$1}
MODEL_LENGTH=${MODEL_LENGTH:-$2}

if [ "$LANGUAGE" = "bfvp" ]; then
  DATA_CLASS="bfvp"
elif [ "$LANGUAGE" = "arithmetic" ]; then
  DATA_CLASS="arithmetic"
else
  DATA_CLASS="regular"
fi

# Build Hydra args
HYDRA_ARGS=""
if [ -n "$OUTPUT_DIR" ]; then
  HYDRA_ARGS="hydra.run.dir=$OUTPUT_DIR"
fi

srun python -u -m main \
  wandb.name="$LANGUAGE-padding-$(date +%Y%m%d-%H%M%S)" \
  data=$DATA_CLASS \
  data.language=$LANGUAGE \
  model=nano \
  algo=lt \
  algo.looping_type=constant \
  model.length=$MODEL_LENGTH \
  data.properties.format=trace \
  $HYDRA_ARGS