import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style for publication-quality plots
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")

# Configure matplotlib for high-quality output
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def load_validation_metrics(metrics_dir):
    """Load validation metrics from JSON file."""
    val_file = Path(metrics_dir) / "validation_metrics.json"
    if not val_file.exists():
        print(f"Validation metrics file not found: {val_file}")
        return None

    with open(val_file, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data)


def load_test_metrics(metrics_dir):
    """Load test metrics from JSON file."""
    test_file = Path(metrics_dir) / "test_metrics.json"
    if not test_file.exists():
        print(f"Test metrics file not found: {test_file}")
        return None

    with open(test_file, "r") as f:
        data = json.load(f)

    return data


def plot_validation_metrics(df, save_dir, show_plot=True):
    """Create plots for validation metrics over epochs."""
    if df is None or df.empty:
        print("No validation data to plot")
        return

    # Get metric columns (exclude epoch and global_step)
    metric_cols = [col for col in df.columns if col not in ["epoch", "global_step"]]

    if not metric_cols:
        print("No metric columns found")
        return

    # Create subplots
    n_metrics = len(metric_cols)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, metric in enumerate(metric_cols):
        ax = axes[i]
        ax.plot(df["epoch"], df[metric], marker="o", linewidth=2, markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("trainer/", "").replace("_", " ").title())
        ax.set_title(f"{metric} vs Epoch")
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(len(metric_cols), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Save plot
    save_path = Path(save_dir) / "validation_metrics_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Validation plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def create_validation_summary_table(df, save_dir):
    """Create a summary table for validation metrics."""
    if df is None or df.empty:
        print("No validation data for summary table")
        return

    # Calculate summary statistics
    summary = df.describe()

    # Save as CSV
    csv_path = Path(save_dir) / "validation_summary.csv"
    summary.to_csv(csv_path)
    print(f"Validation summary saved to: {csv_path}")

    # Print formatted table
    print("\n=== VALIDATION METRICS SUMMARY ===")
    print(summary.round(4))

    # Best values
    print("\n=== BEST VALUES ===")
    best_values = {}
    for col in df.columns:
        if col not in ["epoch", "global_step"]:
            if "loss" in col.lower():
                best_values[col] = df[col].min()
            else:
                best_values[col] = df[col].max()

    for metric, value in best_values.items():
        epoch = df.loc[df[metric] == value, "epoch"].iloc[0]
        print(f"{metric}: {value:.4f} (epoch {epoch})")


def plot_test_metrics_comparison(test_metrics, save_dir, show_plot=True):
    """Create a bar plot comparing test metrics."""
    if test_metrics is None:
        print("No test data to plot")
        return

    # Filter out non-metric keys
    metrics = {
        k: v
        for k, v in test_metrics.items()
        if k not in ["epoch", "global_step"] and isinstance(v, (int, float))
    }

    if not metrics:
        print("No numeric metrics found in test data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    names = [k.replace("test/", "").replace("_", " ").title() for k in metrics.keys()]
    values = list(metrics.values())

    bars = ax.bar(names, values, color=sns.color_palette("husl", len(metrics)))

    ax.set_ylabel("Value")
    ax.set_title("Test Metrics Comparison")
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
            fontsize=10,
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save plot
    save_path = Path(save_dir) / "test_metrics_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Test metrics plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def create_publication_table(test_metrics, save_dir):
    """Create a publication-ready table for test metrics."""
    if test_metrics is None:
        print("No test data for publication table")
        return

    # Create a nice formatted table
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    # Group metrics by category
    loss_metrics = {}
    other_metrics = {}

    for k, v in test_metrics.items():
        if k in ["epoch", "global_step"]:
            continue
        if "loss" in k.lower():
            loss_metrics[k] = v
        else:
            other_metrics[k] = v

    # Print losses
    if loss_metrics:
        print("\nLoss Metrics:")
        for metric, value in loss_metrics.items():
            clean_name = metric.replace("test/", "").replace("_", " ").title()
            print(f"  {clean_name:25s}: {value:.4f}")

    # Print other metrics
    if other_metrics:
        print("\nOther Metrics:")
        for metric, value in other_metrics.items():
            clean_name = metric.replace("test/", "").replace("_", " ").title()
            print(f"  {clean_name:25s}: {value:.4f}")

    print(f"\nTraining completed at epoch {test_metrics.get('epoch', 'N/A')}")
    print(f"Global step: {test_metrics.get('global_step', 'N/A')}")

    # Save as LaTeX table for publication
    latex_path = Path(save_dir) / "test_results_table.tex"
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|l|c|}\n")
        f.write("\\hline\n")
        f.write("Metric & Value \\\\\n")
        f.write("\\hline\n")

        all_metrics = {**loss_metrics, **other_metrics}
        for metric, value in all_metrics.items():
            clean_name = metric.replace("test/", "").replace("_", " ").title()
            f.write(f"{clean_name} & {value:.4f} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Final Test Results}\n")
        f.write("\\label{tab:test_results}\n")
        f.write("\\end{table}\n")

    print(f"Publication table saved to: {latex_path}")


def generate_intermediate_report(metrics_dir, save_dir=None):
    """Generate intermediate tracking report during training."""
    if save_dir is None:
        save_dir = metrics_dir

    print("Generating Intermediate Training Report...")
    print("=" * 50)

    # Load and analyze validation data
    val_df = load_validation_metrics(metrics_dir)
    if val_df is not None:
        print(f"Loaded {len(val_df)} epochs of validation data")

        # Plot metrics
        plot_validation_metrics(val_df, save_dir, show_plot=False)

        # Create summary table
        create_validation_summary_table(val_df, save_dir)

    # Load test data if available
    test_data = load_test_metrics(metrics_dir)
    if test_data is not None:
        print("\nTest data found - generating test plots...")
        plot_test_metrics_comparison(test_data, save_dir, show_plot=False)

    print(f"\nIntermediate report saved to: {save_dir}")


def generate_final_report(metrics_dir, save_dir=None):
    """Generate final publication-ready report."""
    if save_dir is None:
        save_dir = metrics_dir

    print("Generating Final Publication Report...")
    print("=" * 50)

    # Load validation data for training curves
    val_df = load_validation_metrics(metrics_dir)
    if val_df is not None:
        print("Creating publication-quality training curves...")
        plot_validation_metrics(val_df, save_dir, show_plot=False)

    # Load and present test results
    test_data = load_test_metrics(metrics_dir)
    if test_data is not None:
        print("Creating final results table...")
        create_publication_table(test_data, save_dir)
        plot_test_metrics_comparison(test_data, save_dir, show_plot=False)

    print(f"\nFinal report saved to: {save_dir}")
    print("Files generated:")
    print("  - validation_metrics_plot.png (training curves)")
    print("  - test_metrics_plot.png (test results comparison)")
    print("  - validation_summary.csv (detailed validation stats)")
    print("  - test_results_table.tex (LaTeX table for publication)")


def main():
    """Main function to generate reports."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate training metrics reports")
    parser.add_argument(
        "--metrics_dir",
        type=str,
        required=True,
        help="Directory containing metrics JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save reports (default: same as metrics_dir)",
    )
    parser.add_argument(
        "--report_type",
        choices=["intermediate", "final"],
        default="final",
        help="Type of report to generate",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Show plots instead of just saving them",
    )

    args = parser.parse_args()

    # Update plot display setting
    global show_plot
    show_plot = args.show_plots

    if args.report_type == "intermediate":
        generate_intermediate_report(args.metrics_dir, args.output_dir)
    else:
        generate_final_report(args.metrics_dir, args.output_dir)


if __name__ == "__main__":
    main()
