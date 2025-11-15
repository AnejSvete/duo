#!/bin/bash
# Quick helper script to launch grid searches for MDM hyperparameter tuning
#
# Usage:
#   ./scripts/quick_grid_search.sh bfvp           # Use recommended grid
#   ./scripts/quick_grid_search.sh parity quick   # Use quick grid
#   ./scripts/quick_grid_search.sh arithmetic full # Use full grid

set -e

TASK=${1:-bfvp}
GRID=${2:-recommended}
NUM_SEEDS=${3:-3}

# Create timestamp for this search
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="grid_search_${TASK}_${GRID}_${TIMESTAMP}"

echo "================================================="
echo "MDM Hyperparameter Grid Search"
echo "================================================="
echo "Task: $TASK"
echo "Grid: $GRID"
echo "Seeds: $NUM_SEEDS"
echo "Output: $OUTPUT_DIR"
echo "================================================="
echo ""

# Run grid search
python scripts/grid_search_mdm.py \
    --task "$TASK" \
    --grid "$GRID" \
    --num_seeds "$NUM_SEEDS" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "================================================="
echo "Grid search jobs submitted!"
echo "================================================="
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check progress with:"
echo "  watch -n 60 'find $OUTPUT_DIR -name \"validation_metrics.json\" | wc -l'"
echo ""
echo "Analyze results with:"
echo "  python scripts/analyze_grid_search.py --results_dir $OUTPUT_DIR"
echo ""
echo "================================================="
