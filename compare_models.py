"""
Comprehensive model comparison and analysis script.

This script:
1. Compares different models (AR, LT, MDLM) across all metrics
2. Plots validation trends against number of updates (global_step)
3. Compares different decoding strategies for diffusion models
4. Generates publication-ready plots and tables

Usage:
    python compare_models.py --run_dirs outputs/run1 outputs/run2 outputs/run3
    python compare_models.py --run_dirs outputs/run* --output_dir analysis/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for publication-quality plots
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 18,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def extract_run_metadata(run_dir: Path) -> Dict:
    """Extract metadata about the run from directory name and config."""
    metadata = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
    }

    # Try to parse config if available
    config_file = run_dir / "config_tree.txt"
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                content = f.read()

                # Extract algo name
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("name:") and "algo" in content[:content.index(line) if line in content else 0]:
                        metadata["algo"] = line.split("name:")[-1].strip()
                    # Extract language/task
                    if line.startswith("language:"):
                        metadata["language"] = line.split("language:")[-1].strip()
                    # Also check for data.language
                    if "data.language:" in line or (line.startswith("language:") and "data:" in content[:max(0, content.index(line) - 100):content.index(line) if line in content else 0]):
                        lang = line.split(":")[-1].strip()
                        if lang:
                            metadata["language"] = lang
        except Exception as e:
            print(f"Warning: Could not parse config for {run_dir}: {e}")

    # Try to infer from directory name (common pattern: task-algo-timestamp)
    parts = run_dir.name.split("-")
    if len(parts) >= 2:
        # First part is often the task/language
        potential_task = parts[0]
        # Check if it's a known task
        known_tasks = [
            "bfvp", "arithmetic", "parity", "contains_a", "ab_star",
            "mod_3", "even_pairs", "cycle", "dyck"
        ]
        if potential_task in known_tasks:
            metadata["task"] = potential_task
            if "language" not in metadata:
                metadata["language"] = potential_task

        # Look for algo keywords
        for part in parts:
            if part in ["ar", "lt", "mdlm", "d3pm", "sedd", "looping", "cot", "mdm"]:
                if part == "looping":
                    metadata["algo"] = "lt"
                elif part == "cot":
                    metadata["algo"] = "ar"
                elif part == "mdm":
                    metadata["algo"] = "mdlm"
                else:
                    metadata["algo"] = part
                break

    # Set defaults if not found
    if "language" not in metadata:
        metadata["language"] = metadata.get("task", "unknown")
    if "task" not in metadata:
        metadata["task"] = metadata.get("language", "unknown")

    return metadata


def load_all_validation_metrics(run_dirs: List[Path]) -> Dict[str, pd.DataFrame]:
    """Load validation metrics from multiple runs."""
    all_metrics = {}

    for run_dir in run_dirs:
        val_file = run_dir / "validation_metrics.json"
        if val_file.exists():
            try:
                with open(val_file, "r") as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                metadata = extract_run_metadata(run_dir)
                all_metrics[run_dir.name] = {
                    "data": df,
                    "metadata": metadata,
                }
            except Exception as e:
                print(f"Warning: Could not load {val_file}: {e}")
        else:
            print(f"Warning: No validation metrics found in {run_dir}")

    return all_metrics


def load_all_test_metrics(run_dirs: List[Path]) -> Dict[str, Dict]:
    """Load test metrics from multiple runs."""
    all_metrics = {}

    for run_dir in run_dirs:
        test_file = run_dir / "test_metrics.json"
        if test_file.exists():
            try:
                with open(test_file, "r") as f:
                    data = json.load(f)
                metadata = extract_run_metadata(run_dir)
                all_metrics[run_dir.name] = {
                    "data": data,
                    "metadata": metadata,
                }
            except Exception as e:
                print(f"Warning: Could not load {test_file}: {e}")
        else:
            print(f"Warning: No test metrics found in {run_dir}")

    return all_metrics


def plot_validation_trends(
    all_metrics: Dict[str, Dict],
    save_dir: Path,
    metric_name: str = "trainer/nll",
    x_axis: str = "global_step",
    smoothing: int = 1,
):
    """
    Plot validation trends across multiple runs.

    Args:
        all_metrics: Dictionary of validation metrics per run
        save_dir: Directory to save plots
        metric_name: Name of metric to plot
        x_axis: x-axis variable ('epoch' or 'global_step')
        smoothing: Window size for moving average smoothing
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("husl", len(all_metrics))

    for (run_name, run_data), color in zip(all_metrics.items(), colors):
        df = run_data["data"]
        metadata = run_data["metadata"]

        if metric_name not in df.columns:
            print(f"Warning: {metric_name} not found in {run_name}")
            continue

        x = df[x_axis].values
        y = df[metric_name].values

        # Apply smoothing if requested
        if smoothing > 1:
            y = pd.Series(y).rolling(window=smoothing, min_periods=1).mean().values

        # Create label from metadata
        label = metadata.get("algo", run_name)
        if "task" in metadata and metadata["task"] != "unknown":
            label = f"{metadata['task']}-{label}"

        ax.plot(x, y, label=label, linewidth=2, color=color, alpha=0.8)

    ax.set_xlabel(x_axis.replace("_", " ").title())
    ax.set_ylabel(metric_name.split("/")[-1].replace("_", " ").title())
    ax.set_title(f"{metric_name} vs {x_axis}")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    safe_metric_name = metric_name.replace("/", "_")
    save_path = save_dir / f"validation_trends_{safe_metric_name}_{x_axis}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved validation trends plot to {save_path}")
    plt.close()


def plot_all_validation_metrics(
    all_metrics: Dict[str, Dict],
    save_dir: Path,
    x_axis: str = "global_step",
    smoothing: int = 1,
):
    """Plot all available validation metrics."""
    # Get all unique metric names
    all_metric_names = set()
    for run_data in all_metrics.values():
        df = run_data["data"]
        metric_cols = [
            col for col in df.columns if col not in ["epoch", "global_step"]
        ]
        all_metric_names.update(metric_cols)

    print(f"Found {len(all_metric_names)} unique metrics: {sorted(all_metric_names)}")

    for metric_name in sorted(all_metric_names):
        plot_validation_trends(
            all_metrics, save_dir, metric_name, x_axis, smoothing
        )


def compare_decoding_strategies(
    all_metrics: Dict[str, Dict],
    save_dir: Path,
):
    """
    Compare different decoding strategies for diffusion models.
    Analyzes metrics like random_acc_exact, top_k_acc_exact, etc.
    """
    # Extract decoding strategy metrics
    strategy_metrics = {}

    for run_name, run_data in all_metrics.items():
        df = run_data["data"]
        metadata = run_data["metadata"]

        # Look for strategy-specific metrics (val/{strategy}_acc_*)
        strategy_cols = [col for col in df.columns if any(
            strategy in col for strategy in
            ["random", "top_k", "one_level", "all_at_once", "one_at_a_time", "default"]
        )]

        if strategy_cols:
            strategy_metrics[run_name] = {
                "data": df[strategy_cols + ["epoch", "global_step"]],
                "metadata": metadata,
            }

    if not strategy_metrics:
        print("No decoding strategy metrics found")
        return

    # Plot comparison for each run with strategy metrics
    for run_name, run_data in strategy_metrics.items():
        df = run_data["data"]
        metadata = run_data["metadata"]

        # Group by metric type (acc_exact, acc_token, correct_prediction)
        metric_types = set()
        for col in df.columns:
            if col not in ["epoch", "global_step"]:
                # Extract metric type (last part after strategy name)
                parts = col.split("_")
                if len(parts) >= 2:
                    metric_type = "_".join(parts[-2:])  # e.g., "acc_exact"
                    metric_types.add(metric_type)

        for metric_type in sorted(metric_types):
            fig, ax = plt.subplots(figsize=(10, 6))

            strategies = []
            for col in df.columns:
                if metric_type in col and col not in ["epoch", "global_step"]:
                    # Extract strategy name
                    strategy = col.replace(f"val/", "").replace(f"_{metric_type}", "")
                    strategies.append((strategy, col))

            colors = sns.color_palette("husl", len(strategies))

            for (strategy, col), color in zip(strategies, colors):
                ax.plot(
                    df["global_step"],
                    df[col],
                    label=strategy,
                    linewidth=2,
                    color=color,
                    marker="o",
                    markersize=4,
                    alpha=0.8,
                )

            ax.set_xlabel("Global Step")
            ax.set_ylabel(metric_type.replace("_", " ").title())
            algo = metadata.get("algo", "unknown")
            ax.set_title(f"Decoding Strategy Comparison: {metric_type} ({run_name})")
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()

            safe_name = f"{run_name}_{metric_type}".replace("/", "_")
            save_path = save_dir / f"decoding_strategies_{safe_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved decoding strategy plot to {save_path}")
            plt.close()


def create_test_metrics_comparison_table(
    all_test_metrics: Dict[str, Dict],
    save_dir: Path,
):
    """Create a comparison table of test metrics across runs."""
    if not all_test_metrics:
        print("No test metrics to compare")
        return

    # Collect all metrics
    rows = []
    for run_name, run_data in all_test_metrics.items():
        data = run_data["data"]
        metadata = run_data["metadata"]

        row = {
            "Run": run_name,
            "Algorithm": metadata.get("algo", "unknown"),
            "Task": metadata.get("task", "unknown"),
        }

        # Add all numeric metrics
        for k, v in data.items():
            if k not in ["epoch", "global_step"] and isinstance(v, (int, float)):
                # Clean metric name
                clean_name = k.replace("test/", "").replace("trainer/", "")
                row[clean_name] = v

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = save_dir / "test_metrics_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved test metrics comparison to {csv_path}")

    # Print formatted table
    print("\n" + "=" * 80)
    print("TEST METRICS COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False))
    print()

    # Create LaTeX table
    latex_path = save_dir / "test_metrics_comparison.tex"
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\small\n")

        # Generate column specification
        n_cols = len(df.columns)
        col_spec = "|" + "l|" * n_cols
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\hline\n")

        # Header
        f.write(" & ".join(df.columns) + " \\\\\n")
        f.write("\\hline\n")

        # Data rows
        for _, row in df.iterrows():
            formatted_row = []
            for col in df.columns:
                val = row[col]
                if isinstance(val, float):
                    formatted_row.append(f"{val:.4f}")
                else:
                    formatted_row.append(str(val))
            f.write(" & ".join(formatted_row) + " \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Test Metrics Comparison Across Models}\n")
        f.write("\\label{tab:test_comparison}\n")
        f.write("\\end{table}\n")

    print(f"Saved LaTeX table to {latex_path}")


def plot_test_metrics_bar_comparison(
    all_test_metrics: Dict[str, Dict],
    save_dir: Path,
):
    """Create bar plots comparing test metrics across runs."""
    if not all_test_metrics:
        print("No test metrics to compare")
        return

    # Collect all unique metrics
    all_metric_names = set()
    for run_data in all_test_metrics.values():
        data = run_data["data"]
        for k, v in data.items():
            if k not in ["epoch", "global_step"] and isinstance(v, (int, float)):
                clean_name = k.replace("test/", "").replace("trainer/", "")
                all_metric_names.add(clean_name)

    # Create a bar plot for each metric
    for metric_name in sorted(all_metric_names):
        fig, ax = plt.subplots(figsize=(12, 6))

        run_names = []
        values = []
        algos = []

        for run_name, run_data in all_test_metrics.items():
            data = run_data["data"]
            metadata = run_data["metadata"]

            # Look for this metric
            for k, v in data.items():
                clean_name = k.replace("test/", "").replace("trainer/", "")
                if clean_name == metric_name:
                    run_names.append(run_name)
                    values.append(v)
                    algos.append(metadata.get("algo", "unknown"))
                    break

        if not values:
            continue

        # Create bar plot with colors by algorithm
        unique_algos = list(set(algos))
        algo_colors = dict(zip(unique_algos, sns.color_palette("husl", len(unique_algos))))
        colors = [algo_colors[algo] for algo in algos]

        bars = ax.bar(range(len(run_names)), values, color=colors)

        ax.set_xticks(range(len(run_names)))
        ax.set_xticklabels(run_names, rotation=45, ha="right")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title(f"Test Metric Comparison: {metric_name}")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(values) * 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Add legend for algorithms
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=algo_colors[algo], label=algo)
            for algo in unique_algos
        ]
        ax.legend(handles=legend_elements, title="Algorithm", loc="best")

        plt.tight_layout()

        safe_name = metric_name.replace("/", "_")
        save_path = save_dir / f"test_comparison_{safe_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved test comparison plot to {save_path}")
        plt.close()


def plot_final_performance_heatmap(
    all_test_metrics: Dict[str, Dict],
    save_dir: Path,
):
    """Create a heatmap of final performance across runs and metrics."""
    if not all_test_metrics:
        return

    # Create matrix: runs x metrics
    rows = []
    run_names = []

    for run_name, run_data in all_test_metrics.items():
        data = run_data["data"]
        row = {}
        for k, v in data.items():
            if k not in ["epoch", "global_step"] and isinstance(v, (int, float)):
                clean_name = k.replace("test/", "").replace("trainer/", "")
                row[clean_name] = v
        rows.append(row)
        run_names.append(run_name)

    df = pd.DataFrame(rows, index=run_names)

    if df.empty:
        return

    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(12, len(df.columns) * 1.2), max(8, len(df) * 0.8)))

    sns.heatmap(
        df,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        center=df.mean().mean(),
        cbar_kws={"label": "Metric Value"},
        ax=ax,
    )

    ax.set_title("Test Performance Heatmap Across Runs")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Runs")

    plt.tight_layout()

    save_path = save_dir / "test_performance_heatmap.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved performance heatmap to {save_path}")
    plt.close()


def group_runs_by_language(all_val_metrics: Dict, all_test_metrics: Dict) -> Dict[str, Dict]:
    """Group runs by language/task."""
    languages = {}

    # Get all unique languages
    for run_name, run_data in all_val_metrics.items():
        lang = run_data["metadata"].get("language", "unknown")
        if lang not in languages:
            languages[lang] = {"val": {}, "test": {}}
        languages[lang]["val"][run_name] = run_data

    for run_name, run_data in all_test_metrics.items():
        lang = run_data["metadata"].get("language", "unknown")
        if lang not in languages:
            languages[lang] = {"val": {}, "test": {}}
        languages[lang]["test"][run_name] = run_data

    return languages


def create_cross_language_summary(languages: Dict[str, Dict], output_dir: Path):
    """Create summary comparing performance across languages."""
    print("Creating cross-language summary...")

    # Collect best performance for each (language, algorithm) pair
    summary_data = []

    for lang, runs in languages.items():
        test_runs = runs.get("test", {})

        # Group by algorithm
        by_algo = {}
        for run_name, run_data in test_runs.items():
            algo = run_data["metadata"].get("algo", "unknown")
            if algo not in by_algo:
                by_algo[algo] = []
            by_algo[algo].append(run_data["data"])

        # Get best metrics for each algorithm
        for algo, metrics_list in by_algo.items():
            if not metrics_list:
                continue

            # Average across runs with same algo (or take best)
            avg_metrics = {}
            for metrics in metrics_list:
                for k, v in metrics.items():
                    if k not in ["epoch", "global_step"] and isinstance(v, (int, float)):
                        if k not in avg_metrics:
                            avg_metrics[k] = []
                        avg_metrics[k].append(v)

            # Compute averages
            best_metrics = {k: np.mean(v) for k, v in avg_metrics.items()}

            summary_data.append({
                "Language": lang,
                "Algorithm": algo,
                **{k.replace("test/", "").replace("trainer/", ""): v
                   for k, v in best_metrics.items()}
            })

    if not summary_data:
        print("No cross-language data to summarize")
        return

    df = pd.DataFrame(summary_data)

    # Save CSV
    csv_path = output_dir / "cross_language_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved cross-language summary to {csv_path}")

    # Create pivot table for each metric
    metric_cols = [col for col in df.columns if col not in ["Language", "Algorithm"]]

    for metric in metric_cols:
        if metric not in df.columns:
            continue

        # Create pivot: Languages x Algorithms
        pivot = df.pivot(index="Language", columns="Algorithm", values=metric)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 2), max(6, len(pivot) * 1)))

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".4f",
            cmap="RdYlGn",
            center=pivot.values.mean() if not pivot.empty else 0,
            cbar_kws={"label": metric},
            ax=ax,
        )

        ax.set_title(f"Cross-Language Comparison: {metric}")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Language/Task")

        plt.tight_layout()

        safe_name = metric.replace("/", "_").replace(" ", "_")
        save_path = output_dir / f"cross_language_{safe_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved cross-language plot: {save_path}")
        plt.close()

    print()


def generate_comprehensive_report(run_dirs: List[Path], output_dir: Path):
    """Generate comprehensive analysis report."""
    print("=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON REPORT")
    print("=" * 80)
    print(f"Analyzing {len(run_dirs)} runs...")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all metrics
    print("Loading validation metrics...")
    all_val_metrics = load_all_validation_metrics(run_dirs)
    print(f"Loaded validation metrics from {len(all_val_metrics)} runs")
    print()

    print("Loading test metrics...")
    all_test_metrics = load_all_test_metrics(run_dirs)
    print(f"Loaded test metrics from {len(all_test_metrics)} runs")
    print()

    # Group by language
    print("Grouping runs by language/task...")
    languages = group_runs_by_language(all_val_metrics, all_test_metrics)
    print(f"Found {len(languages)} languages: {list(languages.keys())}")
    print()

    # Generate per-language analyses
    if len(languages) > 1:
        print("=" * 80)
        print("PER-LANGUAGE ANALYSIS")
        print("=" * 80)

        for lang, runs in languages.items():
            print(f"\nAnalyzing language: {lang}")
            print("-" * 40)

            lang_dir = output_dir / f"by_language/{lang}"
            lang_dir.mkdir(parents=True, exist_ok=True)

            val_runs = runs.get("val", {})
            test_runs = runs.get("test", {})

            print(f"  - {len(val_runs)} runs with validation metrics")
            print(f"  - {len(test_runs)} runs with test metrics")

            # Generate validation trend plots for this language
            if val_runs:
                plot_all_validation_metrics(
                    val_runs, lang_dir, x_axis="global_step", smoothing=1
                )
                plot_all_validation_metrics(
                    val_runs, lang_dir, x_axis="epoch", smoothing=1
                )
                compare_decoding_strategies(val_runs, lang_dir)

            # Generate test metric comparisons for this language
            if test_runs:
                create_test_metrics_comparison_table(test_runs, lang_dir)
                plot_test_metrics_bar_comparison(test_runs, lang_dir)
                plot_final_performance_heatmap(test_runs, lang_dir)

        print()
        print("=" * 80)
        print("CROSS-LANGUAGE SUMMARY")
        print("=" * 80)
        create_cross_language_summary(languages, output_dir)

    # Also generate combined analysis across all runs
    print("=" * 80)
    print("COMBINED ANALYSIS (ALL LANGUAGES)")
    print("=" * 80)

    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Generate validation trend plots
    if all_val_metrics:
        print("Generating combined validation trend plots...")
        plot_all_validation_metrics(
            all_val_metrics, combined_dir, x_axis="global_step", smoothing=1
        )
        plot_all_validation_metrics(
            all_val_metrics, combined_dir, x_axis="epoch", smoothing=1
        )
        compare_decoding_strategies(all_val_metrics, combined_dir)
        print()

    # Generate test metric comparisons
    if all_test_metrics:
        print("Generating combined test metric comparisons...")
        create_test_metrics_comparison_table(all_test_metrics, combined_dir)
        plot_test_metrics_bar_comparison(all_test_metrics, combined_dir)
        plot_final_performance_heatmap(all_test_metrics, combined_dir)
        print()

    print("=" * 80)
    print("REPORT GENERATION COMPLETE")
    print("=" * 80)
    print(f"All outputs saved to: {output_dir}")
    print()
    print("Directory structure:")
    if len(languages) > 1:
        print("  - by_language/{lang}/ (per-language analyses)")
        print("  - combined/ (all runs together)")
        print("  - cross_language_*.png (language comparison heatmaps)")
        print("  - cross_language_summary.csv (summary table)")
    else:
        print("  - combined/ (all analyses)")
    print()
    print("Each directory contains:")
    print("  - validation_trends_*.png (validation curves)")
    print("  - decoding_strategies_*.png (strategy comparisons)")
    print("  - test_metrics_comparison.csv (detailed comparison)")
    print("  - test_metrics_comparison.tex (LaTeX table)")
    print("  - test_comparison_*.png (bar charts)")
    print("  - test_performance_heatmap.png (heatmap)")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive model comparison and analysis"
    )
    parser.add_argument(
        "--run_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Directories containing run outputs (supports glob patterns)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save analysis outputs (default: ./analysis/)",
    )

    args = parser.parse_args()

    # Resolve paths
    run_dirs = []
    for pattern in args.run_dirs:
        path = Path(pattern)
        if path.is_dir():
            run_dirs.append(path)
        else:
            # Try glob pattern
            parent = path.parent
            matched = parent.glob(path.name)
            run_dirs.extend([p for p in matched if p.is_dir()])

    if not run_dirs:
        print("Error: No valid run directories found")
        return

    output_dir = Path(args.output_dir) if args.output_dir else Path("analysis")

    generate_comprehensive_report(run_dirs, output_dir)


if __name__ == "__main__":
    main()
