#!/bin/bash
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --mem-per-cpu=32000           # server memory requested (per node)
#SBATCH -t 04:00:00                   # Time limit (hh:mm:ss)
#SBATCH --gpus=rtx_3090:1             # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# Generic single job runner with custom output directory support
#
# Usage:
#   sbatch --export=ALL,TASK=bfvp,ALGO=ar,MODEL_LENGTH=128,OUTPUT_DIR=path scripts/run_single.sh
#
# Environment variables:
#   TASK          - Task name (bfvp, parity, arithmetic, etc.)
#   ALGO          - Algorithm (ar, lt, mdlm)
#   MODEL_LENGTH  - Model sequence length
#   OUTPUT_DIR    - Custom output directory (optional)
#   MAX_STEPS     - Max training steps (optional, default: 20000)

module load stack/2024-06 python/3.12.8 eth_proxy
source /cluster/home/asvete/duo/bin/activate

# Get parameters from environment or defaults
TASK=${TASK:-bfvp}
ALGO=${ALGO:-ar}
MODEL_LENGTH=${MODEL_LENGTH:-128}
MAX_STEPS=${MAX_STEPS:-20000}

# Determine data class
if [ "$TASK" = "bfvp" ]; then
  DATA_CLASS="bfvp"
elif [ "$TASK" = "arithmetic" ]; then
  DATA_CLASS="arithmetic"
else
  DATA_CLASS="regular"
fi

# Determine algorithm settings
case $ALGO in
  ar|cot)
    ALGO_NAME="ar"
    FORMAT="trace"
    ;;
  lt|looping)
    ALGO_NAME="lt"
    FORMAT="final_value"
    EXTRA_ARGS="algo.looping_type=log"
    ;;
  mdlm|mdm)
    ALGO_NAME="mdlm"
    FORMAT="trace"
    ;;
  *)
    echo "Error: Unknown algorithm: $ALGO"
    exit 1
    ;;
esac

# Build command
CMD="python -u -m main \
  wandb.name=\"$TASK-$ALGO-$(date +%Y%m%d-%H%M%S)\" \
  data=$DATA_CLASS \
  data.language=$TASK \
  model=nano \
  algo=$ALGO_NAME \
  model.length=$MODEL_LENGTH \
  data.properties.format=$FORMAT \
  trainer.max_steps=$MAX_STEPS"

# Add extra args if defined
if [ -n "$EXTRA_ARGS" ]; then
  CMD="$CMD $EXTRA_ARGS"
fi

# Add custom output directory if specified
if [ -n "$OUTPUT_DIR" ]; then
  CMD="$CMD checkpointing.save_dir=$OUTPUT_DIR"
fi

echo "========================================"
echo "Running: $TASK × $ALGO"
echo "Model Length: $MODEL_LENGTH"
echo "Max Steps: $MAX_STEPS"
echo "Output Dir: ${OUTPUT_DIR:-<default>}"
echo "========================================"
echo ""

# Run the command
srun $CMD
