#!/bin/bash
#SBATCH -J prepare_data            # Job name
#SBATCH -o watch_folder/%x_%j.out  # output file (%j expands to jobID)
#SBATCH --mem-per-cpu=16000        # Less memory needed for data prep
#SBATCH -t 00:30:00                # 30 min should be enough
#SBATCH --cpus-per-task=4          # CPU only, no GPU needed!
#SBATCH --open-mode=append         # Do not overwrite logs
#SBATCH --requeue                  # Requeue upon pre-emption

# Prepare data for a specific language and format
# This script just generates and caches data without any training
#
# Usage: sbatch --export=ALL,LANGUAGE=bfvp,FORMAT=trace,MODEL_LENGTH=128 scripts/prepare_data.sh

module load stack/2024-06 python/3.12.8 eth_proxy
source /cluster/home/asvete/duo/bin/activate

LANGUAGE=${LANGUAGE:-bfvp}
FORMAT=${FORMAT:-trace}
MODEL_LENGTH=${MODEL_LENGTH:-128}

# Determine data class
if [ "$LANGUAGE" = "bfvp" ]; then
  DATA_CLASS="bfvp"
elif [ "$LANGUAGE" = "arithmetic" ]; then
  DATA_CLASS="arithmetic"
else
  DATA_CLASS="regular"
fi

echo "========================================"
echo "Preparing Data"
echo "========================================"
echo "Language: $LANGUAGE"
echo "Format: $FORMAT"
echo "Model Length: $MODEL_LENGTH"
echo "Data Class: $DATA_CLASS"
echo "========================================"
echo ""

# Use the new prepare_data mode - no training, just data generation
srun python -u -m main \
  mode=prepare_data \
  data=$DATA_CLASS \
  data.language=$LANGUAGE \
  model.length=$MODEL_LENGTH \
  data.properties.format=$FORMAT

echo ""
echo "✓ Data prepared and cached for: $LANGUAGE ($FORMAT format, length $MODEL_LENGTH)"
