#!/usr/bin/env python3
"""
Analyze results from MDM hyperparameter grid search.

This script:
1. Collects metrics from all grid search runs
2. Ranks configurations by performance
3. Analyzes hyperparameter importance
4. Generates visualization plots
5. Identifies best configurations and problematic hyperparameters

Usage:
    python scripts/analyze_grid_search.py --results_dir grid_search_results
    python scripts/analyze_grid_search.py --results_dir grid_search_results --metric val/acc_exact
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def extract_hyperparams_from_name(job_name: str) -> Dict[str, Any]:
    """Extract hyperparameters from job name."""
    params = {}

    # Parse job name pattern: mdm_{task}_lr{lr}_wd{wd}_gc{gc}_ns{noise}_ws{ws}_bs{bs}_r{seed}
    patterns = {
        "lr": r"lr([\d.e-]+)",
        "weight_decay": r"wd([\d.]+)",
        "gradient_clip_val": r"gc([\d.]+)",
        "noise_schedule": r"ns(\w+)",
        "warmup_steps": r"ws(\d+)",
        "batch_size": r"bs(\d+)",
        "seed": r"r(\d+)",
    }

    for param, pattern in patterns.items():
        match = re.search(pattern, job_name)
        if match:
            value = match.group(1)
            # Convert to appropriate type
            if param in ["lr", "weight_decay", "gradient_clip_val"]:
                params[param] = float(value)
            elif param in ["warmup_steps", "batch_size", "seed"]:
                params[param] = int(value)
            else:
                params[param] = value

    return params


def load_metrics_from_run(run_dir: Path) -> Dict[str, Any]:
    """Load metrics from a single run directory."""
    metrics_file = run_dir / "validation_metrics.json"

    if not metrics_file.exists():
        return None

    try:
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        print(f"Error loading {metrics_file}: {e}")
        return None


def collect_all_results(results_dir: Path) -> pd.DataFrame:
    """Collect results from all runs into a DataFrame."""
    all_results = []

    # Find all run directories
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue

        # Extract hyperparameters from directory name
        params = extract_hyperparams_from_name(run_dir.name)
        if not params:
            continue

        # Load metrics
        metrics = load_metrics_from_run(run_dir)
        if metrics is None:
            continue

        # Combine parameters and metrics
        result = {**params}

        # Extract key metrics (use final values)
        if isinstance(metrics, list):
            # Take last epoch metrics
            final_metrics = metrics[-1] if metrics else {}
        else:
            final_metrics = metrics

        # Add metrics to result
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                result[key] = value

        # Add training success indicator
        result["completed"] = True
        result["run_dir"] = str(run_dir)

        all_results.append(result)

    if not all_results:
        print("No results found!")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    print(f"Loaded {len(df)} runs")
    return df


def analyze_hyperparameter_importance(df: pd.DataFrame, metric: str = "val/acc_exact"):
    """Analyze which hyperparameters matter most for the metric."""
    print(f"\n{'='*80}")
    print(f"Hyperparameter Importance Analysis for {metric}")
    print(f"{'='*80}\n")

    hyperparams = ["lr", "weight_decay", "gradient_clip_val", "noise_schedule", "warmup_steps", "batch_size"]

    # For each hyperparameter, compute mean metric value for each setting
    importance_scores = {}

    for param in hyperparams:
        if param not in df.columns:
            continue

        # Group by this parameter and compute mean and std
        grouped = df.groupby(param)[metric].agg(["mean", "std", "count"])
        grouped = grouped.sort_values("mean", ascending=False)

        print(f"\n{param}:")
        print(grouped.to_string())

        # Compute importance as variance explained
        # Higher variance = more important
        variance = df.groupby(param)[metric].var().mean()
        importance_scores[param] = variance

    print(f"\n{'='*80}")
    print("Importance Scores (higher = more important):")
    print(f"{'='*80}\n")
    for param, score in sorted(importance_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{param:20s}: {score:.6f}")

    return importance_scores


def find_best_configurations(df: pd.DataFrame, metric: str = "val/acc_exact", top_k: int = 10):
    """Find the best configurations."""
    print(f"\n{'='*80}")
    print(f"Top {top_k} Configurations by {metric}")
    print(f"{'='*80}\n")

    # Sort by metric (descending for accuracy, ascending for loss)
    ascending = "loss" in metric.lower()
    df_sorted = df.sort_values(metric, ascending=ascending)

    # Show top configurations
    display_cols = ["lr", "weight_decay", "gradient_clip_val", "noise_schedule",
                   "warmup_steps", "batch_size", metric]
    display_cols = [c for c in display_cols if c in df.columns]

    print(df_sorted[display_cols].head(top_k).to_string(index=False))

    return df_sorted.head(top_k)


def analyze_training_stability(df: pd.DataFrame):
    """Analyze which configurations led to training instabilities."""
    print(f"\n{'='*80}")
    print("Training Stability Analysis")
    print(f"{'='*80}\n")

    # Check for NaN/Inf in loss
    unstable = df[df["train/loss"].isna() | (df["train/loss"] > 100)]

    if len(unstable) > 0:
        print(f"Found {len(unstable)} unstable runs (loss > 100 or NaN)")
        print("\nUnstable configurations by hyperparameter:")

        hyperparams = ["lr", "weight_decay", "gradient_clip_val", "noise_schedule"]
        for param in hyperparams:
            if param in unstable.columns:
                counts = unstable[param].value_counts()
                print(f"\n{param}:")
                print(counts.to_string())
    else:
        print("All runs were stable!")

    # Check recovery count
    if "recovery_count" in df.columns:
        recovered = df[df["recovery_count"] > 0]
        if len(recovered) > 0:
            print(f"\n{len(recovered)} runs required recovery")
            print(f"Average recovery count: {recovered['recovery_count'].mean():.2f}")


def plot_hyperparameter_effects(df: pd.DataFrame, metric: str = "val/acc_exact", output_dir: Path = None):
    """Create visualization plots for hyperparameter effects."""
    print(f"\n{'='*80}")
    print("Generating plots...")
    print(f"{'='*80}\n")

    hyperparams = ["lr", "weight_decay", "gradient_clip_val", "noise_schedule", "warmup_steps", "batch_size"]
    hyperparams = [h for h in hyperparams if h in df.columns]

    # Set up plot style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, param in enumerate(hyperparams):
        ax = axes[idx]

        if param in ["lr", "weight_decay", "gradient_clip_val", "warmup_steps", "batch_size"]:
            # Numerical parameters: box plot
            df_sorted = df.sort_values(param)
            df_sorted[param] = df_sorted[param].astype(str)  # Convert to string for categorical
            sns.boxplot(data=df_sorted, x=param, y=metric, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            # Categorical parameters: box plot
            sns.boxplot(data=df, x=param, y=metric, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        ax.set_title(f"{metric} vs {param}")
        ax.set_xlabel(param)
        ax.set_ylabel(metric)

    plt.tight_layout()

    if output_dir:
        plot_path = output_dir / f"hyperparameter_effects_{metric.replace('/', '_')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {plot_path}")
    else:
        plt.show()

    plt.close()


def plot_learning_curves(df: pd.DataFrame, metric: str = "val/acc_exact", top_k: int = 5, output_dir: Path = None):
    """Plot learning curves for top configurations."""
    print(f"\n{'='*80}")
    print(f"Generating learning curves for top {top_k} configurations...")
    print(f"{'='*80}\n")

    # Get top configurations
    ascending = "loss" in metric.lower()
    df_sorted = df.sort_values(metric, ascending=ascending).head(top_k)

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (_, row) in enumerate(df_sorted.iterrows()):
        run_dir = Path(row["run_dir"])
        metrics_file = run_dir / "validation_metrics.json"

        if not metrics_file.exists():
            continue

        with open(metrics_file, "r") as f:
            metrics_history = json.load(f)

        if isinstance(metrics_history, list):
            values = [m.get(metric, np.nan) for m in metrics_history]
            epochs = range(len(values))

            label = f"lr={row['lr']:.0e}, wd={row['weight_decay']:.2f}, gc={row['gradient_clip_val']:.0f}"
            ax.plot(epochs, values, marker='o', label=label, alpha=0.7)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"Learning Curves - Top {top_k} Configurations")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        plot_path = output_dir / f"learning_curves_top{top_k}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {plot_path}")
    else:
        plt.show()

    plt.close()


def generate_recommendations(df: pd.DataFrame, metric: str = "val/acc_exact"):
    """Generate recommendations based on analysis."""
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}\n")

    # Find best overall configuration
    ascending = "loss" in metric.lower()
    best = df.sort_values(metric, ascending=ascending).iloc[0]

    print("Best Configuration:")
    print(f"  Learning rate: {best['lr']:.0e}")
    print(f"  Weight decay: {best['weight_decay']:.3f}")
    print(f"  Gradient clip: {best['gradient_clip_val']:.1f}")
    print(f"  Noise schedule: {best['noise_schedule']}")
    print(f"  Warmup steps: {best['warmup_steps']:.0f}")
    print(f"  Batch size: {best['batch_size']:.0f}")
    print(f"  {metric}: {best[metric]:.4f}")

    # Find safe ranges for each hyperparameter
    # (ranges that consistently give good results)
    print("\n\nRecommended Ranges (based on top 25% of runs):")
    threshold = df[metric].quantile(0.75 if not ascending else 0.25)

    if ascending:
        good_runs = df[df[metric] <= threshold]
    else:
        good_runs = df[df[metric] >= threshold]

    for param in ["lr", "weight_decay", "gradient_clip_val", "warmup_steps", "batch_size"]:
        if param in good_runs.columns:
            values = good_runs[param]
            print(f"  {param}: [{values.min()}, {values.max()}] (median: {values.median()})")

    # Identify problematic settings
    print("\n\nSettings to Avoid (frequently in bottom 25%):")
    bad_threshold = df[metric].quantile(0.25 if not ascending else 0.75)

    if ascending:
        bad_runs = df[df[metric] >= bad_threshold]
    else:
        bad_runs = df[df[metric] <= bad_threshold]

    for param in ["lr", "weight_decay", "gradient_clip_val", "noise_schedule"]:
        if param in bad_runs.columns:
            bad_values = bad_runs[param].mode()
            if len(bad_values) > 0:
                print(f"  {param}: {bad_values.values[0]} (appears in {len(bad_runs[bad_runs[param] == bad_values.values[0]])} poor runs)")


def main():
    parser = argparse.ArgumentParser(description="Analyze MDM grid search results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing grid search results")
    parser.add_argument("--metric", type=str, default="val/acc_exact",
                       help="Metric to optimize (default: val/acc_exact)")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of top configurations to show")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save plots (default: results_dir/analysis)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        return

    # Set up output directory for plots
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*80}")
    print(f"MDM Grid Search Analysis")
    print(f"{'='*80}")
    print(f"Results directory: {results_dir}")
    print(f"Optimizing for: {args.metric}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Collect all results
    df = collect_all_results(results_dir)

    if df.empty:
        print("No results found to analyze!")
        return

    # Print available metrics
    print(f"\nAvailable metrics: {[c for c in df.columns if '/' in c]}")

    # Check if requested metric exists
    if args.metric not in df.columns:
        print(f"\nWarning: Metric '{args.metric}' not found in results!")
        print(f"Using first available metric instead")
        metric_cols = [c for c in df.columns if '/' in c]
        if metric_cols:
            args.metric = metric_cols[0]
            print(f"Using: {args.metric}")
        else:
            print("No metrics found!")
            return

    # Run analyses
    analyze_hyperparameter_importance(df, args.metric)
    find_best_configurations(df, args.metric, args.top_k)
    analyze_training_stability(df)
    generate_recommendations(df, args.metric)

    # Generate plots
    plot_hyperparameter_effects(df, args.metric, output_dir)
    plot_learning_curves(df, args.metric, min(args.top_k, 5), output_dir)

    # Save summary to CSV
    summary_file = output_dir / "grid_search_results.csv"
    df.to_csv(summary_file, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to {summary_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
