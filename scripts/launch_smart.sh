#!/bin/bash
# Smart experiment launcher with automatic data preparation
#
# This script:
# 1. First prepares all required data formats in parallel
# 2. Then launches all training jobs with dependencies (wait for data prep)
#
# Usage:
#   ./scripts/launch_smart.sh EXPERIMENT_NAME LANGUAGE1 [LANGUAGE2 ...]
#
# Examples:
#   ./scripts/launch_smart.sh my_exp bfvp
#   ./scripts/launch_smart.sh full_study bfvp parity arithmetic

set -e

if [ $# -lt 2 ]; then
    cat <<EOF
Usage: $0 EXPERIMENT_NAME LANGUAGE1 [LANGUAGE2 ...]

Smart launcher that:
  1. Prepares all data formats first (parallel)
  2. Launches training jobs with dependencies

Examples:
  $0 test bfvp
  $0 full_study bfvp parity arithmetic

This launches all algorithm variants for each language.

EOF
    exit 1
fi

EXPERIMENT_NAME=$1
shift
LANGUAGES=("$@")

# Configuration
SCRATCH_DIR="/cluster/scratch/asvete/duo/outputs"
EXPERIMENT_DIR="experiments/${EXPERIMENT_NAME}"  # For helper scripts only
mkdir -p "$EXPERIMENT_DIR"
mkdir -p watch_folder

echo "========================================"
echo "Smart Launch: $EXPERIMENT_NAME"
echo "========================================"
echo "Languages: ${LANGUAGES[*]}"
echo "Output: $EXPERIMENT_DIR"
echo ""
echo "This will:"
echo "  1. Prepare all data formats (3 formats per language)"
echo "  2. Launch all training jobs when data is ready"
echo ""

# Determine model lengths based on language
declare -A SHORT_LENGTHS
declare -A LONG_LENGTHS

for lang in "${LANGUAGES[@]}"; do
    if [ "$lang" = "arithmetic" ] || [ "$lang" = "bfvp" ]; then
        SHORT_LENGTHS[$lang]=45
        LONG_LENGTHS[$lang]=90
    else
        SHORT_LENGTHS[$lang]=45
        LONG_LENGTHS[$lang]=90
    fi
done

# Save config
cat > "$EXPERIMENT_DIR/config.txt" <<EOF
Experiment: $EXPERIMENT_NAME
Launched: $(date)
Languages: ${LANGUAGES[*]}
Pipeline: Data Prep → Training (with dependencies)
EOF

# ============================================================================
# STAGE 1: Prepare all data formats
# ============================================================================

echo "STAGE 1: Preparing data..."
echo ""

DATA_PREP_JOBS=()

for lang in "${LANGUAGES[@]}"; do
    SHORT=${SHORT_LENGTHS[$lang]}
    LONG=${LONG_LENGTHS[$lang]}

    # Three data formats needed:
    # 1. trace (for CoT, MDLM, padding variants)
    # 2. final_value (for classifier, looping)
    # 3. empty_trace (for empty_padding variants)

    for format_spec in "trace:$LONG" "final_value:$SHORT" "empty_trace:$LONG"; do
        IFS=':' read -r format length <<< "$format_spec"

        echo "  Submitting data prep: $lang ($format, length=$length)"

        JOB_ID=$(sbatch \
            --job-name="dataprep-${lang}-${format}" \
            --export=ALL,LANGUAGE=$lang,FORMAT=$format,MODEL_LENGTH=$length \
            scripts/prepare_data.sh | grep -oP '\d+$')

        DATA_PREP_JOBS+=($JOB_ID)

        # Store job ID for this specific format (for dependencies)
        eval "PREP_${lang}_${format}=${JOB_ID}"

        echo "    → Job $JOB_ID"
    done
done

echo ""
echo "Submitted ${#DATA_PREP_JOBS[@]} data prep jobs: ${DATA_PREP_JOBS[*]}"
echo ""

# Create dependency string (all data prep jobs)
PREP_DEPENDENCY=$(IFS=:; echo "${DATA_PREP_JOBS[*]}")

# ============================================================================
# STAGE 2: Launch training jobs (depend on relevant data prep)
# ============================================================================

echo "STAGE 2: Scheduling training jobs (will start after data prep)..."
echo ""

TRAINING_JOBS=()

for lang in "${LANGUAGES[@]}"; do
    SHORT=${SHORT_LENGTHS[$lang]}
    LONG=${LONG_LENGTHS[$lang]}

    # Get data prep job IDs for this language
    eval "PREP_TRACE=\$PREP_${lang}_trace"
    eval "PREP_FINAL=\$PREP_${lang}_final_value"
    eval "PREP_EMPTY=\$PREP_${lang}_empty_trace"

    # Define all training configurations
    # Format: "script:algo:length:depends_on_prep_job"
    declare -a CONFIGS=(
        # Uses trace format (depends on trace prep)
        "train_cot.sh:cot:$LONG:$PREP_TRACE"
        "train_mdm.sh:mdm:$LONG:$PREP_TRACE"
        "train_padding.sh:padding:$LONG:$PREP_TRACE"
        "train_padded_looping.sh:padded_looping:$LONG:$PREP_TRACE"

        # Uses final_value format (depends on final_value prep)
        "train_classifier.sh:classifier:$SHORT:$PREP_FINAL"
        "train_looping.sh:looping:$SHORT:$PREP_FINAL"

        # Uses empty_trace format (depends on empty_trace prep)
        "train_empty_padding.sh:empty_padding:$LONG:$PREP_EMPTY"
        "train_empty_padded_looping.sh:empty_padded_looping:$LONG:$PREP_EMPTY"
    )

    for config in "${CONFIGS[@]}"; do
        IFS=':' read -r script algo length dep_job <<< "$config"

        echo "  Scheduling: $lang × $algo (depends on data prep job $dep_job)"

        # Output to scratch (has space), not home
        OUTPUT_DIR="${SCRATCH_DIR}/${EXPERIMENT_NAME}/${lang}/${algo}"

        # Submit with dependency on specific data prep job
        JOB_ID=$(sbatch \
            --job-name="${EXPERIMENT_NAME}-${lang}-${algo}" \
            --dependency=afterok:$dep_job \
            --export=ALL,OUTPUT_DIR=$OUTPUT_DIR,LANGUAGE=$lang,MODEL_LENGTH=$length \
            scripts/$script | grep -oP '\d+$')

        TRAINING_JOBS+=($JOB_ID)

        echo "    → Job $JOB_ID (waits for $dep_job)"
    done
done

echo ""
echo "Scheduled ${#TRAINING_JOBS[@]} training jobs: ${TRAINING_JOBS[*]}"
echo ""

# ============================================================================
# Create helper scripts
# ============================================================================

# Analysis script
cat > "$EXPERIMENT_DIR/analyze.sh" <<'ANALYSIS_EOF'
#!/bin/bash
ANALYSIS_EOF

cat >> "$EXPERIMENT_DIR/analyze.sh" <<EOF
SCRATCH_DIR="$SCRATCH_DIR"
EXPERIMENT_NAME="$EXPERIMENT_NAME"
SCRATCH_EXPERIMENT_DIR="\$SCRATCH_DIR/\$EXPERIMENT_NAME"

echo "Analyzing: $EXPERIMENT_NAME"
NUM_RUNS=\$(find "\$SCRATCH_EXPERIMENT_DIR" -name "validation_metrics.json" 2>/dev/null | wc -l)
EXPECTED=${#TRAINING_JOBS[@]}

echo "Progress: \$NUM_RUNS / \$EXPECTED runs completed"
echo ""

if [ \$NUM_RUNS -eq 0 ]; then
    echo "No completed runs. Check status: ./status.sh"
    exit 1
fi

echo "Running analysis..."
python compare_models.py \\
    --run_dirs "\$SCRATCH_EXPERIMENT_DIR"/*/* \\
    --output_dir "$EXPERIMENT_DIR"/analysis

echo ""
echo "✓ Results in: $EXPERIMENT_DIR/analysis/"
EOF

chmod +x "$EXPERIMENT_DIR/analyze.sh"

# Status script
cat > "$EXPERIMENT_DIR/status.sh" <<'STATUS_EOF'
#!/bin/bash
STATUS_EOF

cat >> "$EXPERIMENT_DIR/status.sh" <<EOF
SCRATCH_DIR="$SCRATCH_DIR"
EXPERIMENT_NAME="$EXPERIMENT_NAME"
SCRATCH_EXPERIMENT_DIR="\$SCRATCH_DIR/\$EXPERIMENT_NAME"

echo "========================================"
echo "Status: $EXPERIMENT_NAME"
echo "========================================"
echo ""

echo "Data Preparation Jobs:"
squeue -j $(IFS=,; echo "${DATA_PREP_JOBS[*]}") --format="%.18i %.40j %.8T %.10M" 2>/dev/null || echo "  All data prep complete"

echo ""
echo "Training Jobs:"
squeue -u \$USER | grep "${EXPERIMENT_NAME}-" 2>/dev/null || echo "  No training jobs in queue"

echo ""
NUM_COMPLETED=\$(find "\$SCRATCH_EXPERIMENT_DIR" -name "validation_metrics.json" 2>/dev/null | wc -l)
echo "Completed: \$NUM_COMPLETED / ${#TRAINING_JOBS[@]} training runs"

echo ""
echo "Data location: \$SCRATCH_EXPERIMENT_DIR"
echo "To analyze: $EXPERIMENT_DIR/analyze.sh"
EOF

chmod +x "$EXPERIMENT_DIR/status.sh"

# ============================================================================
# Print summary
# ============================================================================

cat <<EOF

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Smart Launch Complete: $EXPERIMENT_NAME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pipeline:
  1. Data Prep:  ${#DATA_PREP_JOBS[@]} jobs (running now)
  2. Training:   ${#TRAINING_JOBS[@]} jobs (will auto-start when data ready)

Languages: ${LANGUAGES[*]}
Outputs: $SCRATCH_DIR/$EXPERIMENT_NAME/
Scripts: $EXPERIMENT_DIR/

Data Prep Jobs: ${DATA_PREP_JOBS[*]}
Training Jobs: ${TRAINING_JOBS[*]}

Commands:

  # Check status (shows both stages)
  $EXPERIMENT_DIR/status.sh

  # Check queue
  squeue -u \$USER

  # View logs
  tail -f watch_folder/dataprep-*        # Data prep
  tail -f watch_folder/${EXPERIMENT_NAME}-*  # Training

  # When complete, analyze
  $EXPERIMENT_DIR/analyze.sh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Workflow:
  ⏳ Data prep jobs running now (~5-30 min)
  ⏸️  Training jobs queued (will auto-start after data prep)
  🎯 Analysis ready when all jobs complete

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EOF
