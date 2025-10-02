#!/bin/bash
# Example workflow for training and comparing models across multiple languages

# ============================================================================
# STEP 1: Train multiple models on multiple languages
# ============================================================================

echo "========================================="
echo "Training on BFVP (Boolean Formulas)"
echo "========================================="

python main.py \
    data=bfvp \
    algo=ar \
    model=nano \
    data.properties.format=trace \
    wandb.name=bfvp-ar \
    trainer.max_steps=5000

python main.py \
    data=bfvp \
    algo=lt \
    algo.looping_type=log \
    model=nano \
    data.properties.format=final_value \
    wandb.name=bfvp-lt \
    trainer.max_steps=5000

python main.py \
    data=bfvp \
    algo=mdlm \
    model=nano \
    data.properties.format=trace \
    wandb.name=bfvp-mdlm \
    trainer.max_steps=5000

echo "========================================="
echo "Training on Parity"
echo "========================================="

python main.py \
    data=parity \
    algo=ar \
    model=nano \
    data.properties.format=trace \
    wandb.name=parity-ar \
    trainer.max_steps=5000

python main.py \
    data=parity \
    algo=lt \
    algo.looping_type=log \
    model=nano \
    data.properties.format=final_value \
    wandb.name=parity-lt \
    trainer.max_steps=5000

python main.py \
    data=parity \
    algo=mdlm \
    model=nano \
    data.properties.format=trace \
    wandb.name=parity-mdlm \
    trainer.max_steps=5000

echo "========================================="
echo "Training on Arithmetic"
echo "========================================="

python main.py \
    data=arithmetic \
    algo=ar \
    model=nano \
    data.properties.format=trace \
    wandb.name=arithmetic-ar \
    trainer.max_steps=5000

python main.py \
    data=arithmetic \
    algo=lt \
    algo.looping_type=log \
    model=nano \
    data.properties.format=final_value \
    wandb.name=arithmetic-lt \
    trainer.max_steps=5000

python main.py \
    data=arithmetic \
    algo=mdlm \
    model=nano \
    data.properties.format=trace \
    wandb.name=arithmetic-mdlm \
    trainer.max_steps=5000

# ============================================================================
# STEP 2: Monitor a single run (during or after training)
# ============================================================================

# Find the latest run directory
LATEST_RUN=$(ls -td outputs/*/* | head -1)

echo "Analyzing latest run: $LATEST_RUN"
python analyze_metrics.py \
    --metrics_dir "$LATEST_RUN" \
    --report_type final

# ============================================================================
# STEP 3: Compare all runs (AUTOMATIC PER-LANGUAGE + CROSS-LANGUAGE ANALYSIS)
# ============================================================================

# Get today's date
TODAY=$(date +%Y-%m-%d)

echo "========================================="
echo "Comparing all runs from $TODAY..."
echo "This will:"
echo "  1. Group by language (bfvp, parity, arithmetic)"
echo "  2. Create per-language comparisons"
echo "  3. Create cross-language heatmaps"
echo "  4. Create combined analysis"
echo "========================================="

python compare_models.py \
    --run_dirs outputs/$TODAY/*/ \
    --output_dir analysis/full_comparison_$TODAY/

# ============================================================================
# STEP 4: Compare just one language
# ============================================================================

echo "Comparing only BFVP runs..."
python compare_models.py \
    --run_dirs outputs/$TODAY/bfvp-*/ \
    --output_dir analysis/bfvp_only/

# ============================================================================
# STEP 5: Compare specific algorithm across tasks
# ============================================================================

# Example: Compare all MDLM runs across different tasks
echo "Comparing MDLM across all tasks..."
python compare_models.py \
    --run_dirs outputs/$TODAY/*-mdlm-*/ \
    --output_dir analysis/mdlm_comparison/

# This will still do per-language grouping, showing how MDLM performs
# on bfvp vs parity vs arithmetic

# ============================================================================
# STEP 6: Review results
# ============================================================================

echo ""
echo "========================================="
echo "Analysis complete!"
echo "========================================="
echo ""
echo "Check these directories for results:"
echo ""
echo "Full comparison (all languages):"
echo "  - analysis/full_comparison_$TODAY/by_language/bfvp/"
echo "  - analysis/full_comparison_$TODAY/by_language/parity/"
echo "  - analysis/full_comparison_$TODAY/by_language/arithmetic/"
echo "  - analysis/full_comparison_$TODAY/cross_language_*.png"
echo "  - analysis/full_comparison_$TODAY/cross_language_summary.csv"
echo ""
echo "Single language:"
echo "  - analysis/bfvp_only/combined/"
echo ""
echo "Single algorithm:"
echo "  - analysis/mdlm_comparison/by_language/*/"
echo ""
echo "Key files to check:"
echo "  - cross_language_acc_exact.png (which algo for which task?)"
echo "  - cross_language_nll.png (loss comparison)"
echo "  - by_language/*/test_performance_heatmap.png (per-task overview)"
echo "  - by_language/*/validation_trends_*.png (learning curves)"
echo ""
