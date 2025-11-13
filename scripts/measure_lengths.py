#!/usr/bin/env python
"""
Script to measure actual sequence lengths for different task configurations.
Use this to determine correct min_train_len and max_train_len for curriculum learning.

Usage:
    python scripts/measure_lengths.py --task bfvp --min_depth 1 --max_depth 5 --num_vars 2 --format trace
    python scripts/measure_lengths.py --task arithmetic --min_depth 1 --max_depth 4 --min_val 1 --max_val 50
"""

import argparse
import os
import random
import sys
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arithmetic
import bfvp


def measure_bfvp_lengths(min_depth, max_depth, num_vars, format_mode, num_samples=500):
    """Measure sequence lengths for BFVP task."""
    print(f"\n{'='*80}")
    print(f"BFVP Length Analysis")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  min_depth: {min_depth}")
    print(f"  max_depth: {max_depth}")
    print(f"  num_vars: {num_vars}")
    print(f"  format: {format_mode}")
    print(f"  samples: {num_samples}")
    print(f"{'='*80}\n")

    lengths_by_depth = defaultdict(list)
    all_lengths = []

    for _ in range(num_samples):
        depth = random.randint(min_depth, max_depth)
        tree = bfvp.generate_formula_tree(depth, num_vars)
        variables = bfvp.get_variables_from_tree(tree)
        assignments = {var: random.choice([True, False]) for var in variables}
        substituted = bfvp.substitute_vars_in_tree(tree, assignments)

        text = bfvp._generate_text_from_tree(substituted, tree, assignments, format_mode)
        length = len(text.split())

        lengths_by_depth[depth].append(length)
        all_lengths.append(length)

    # Print per-depth statistics
    print("Per-Depth Statistics:")
    print(f"{'Depth':<8} {'Count':<8} {'Min':<8} {'Max':<8} {'Avg':<8} {'Median':<8}")
    print("-" * 56)

    for depth in range(min_depth, max_depth + 1):
        if depth in lengths_by_depth:
            lens = lengths_by_depth[depth]
            print(f"{depth:<8} {len(lens):<8} {min(lens):<8} {max(lens):<8} "
                  f"{sum(lens)/len(lens):<8.1f} {sorted(lens)[len(lens)//2]:<8}")

    # Print overall statistics
    print(f"\n{'='*80}")
    print("Overall Statistics:")
    print(f"  Total samples: {len(all_lengths)}")
    print(f"  Min length: {min(all_lengths)}")
    print(f"  Max length: {max(all_lengths)}")
    print(f"  Avg length: {sum(all_lengths)/len(all_lengths):.1f}")
    print(f"  Median length: {sorted(all_lengths)[len(all_lengths)//2]}")

    # Percentiles
    sorted_lens = sorted(all_lengths)
    print(f"\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        idx = int(len(sorted_lens) * p / 100)
        print(f"  {p:2d}%: {sorted_lens[idx]}")

    # Recommendations
    recommended_min = max(1, min(all_lengths) - 2)
    recommended_max = max(all_lengths) + max(20, int(max(all_lengths) * 0.1))

    print(f"\n{'='*80}")
    print("Recommended Config:")
    print(f"{'='*80}")
    print(f"  min_train_len: {recommended_min}")
    print(f"  max_train_len: {recommended_max}")
    print(f"\nAdd to configs/data/bfvp.yaml:")
    print(f"  min_train_len: {recommended_min}  # Measured from depth {min_depth}-{max_depth}, format={format_mode}")
    print(f"  max_train_len: {recommended_max}  # With 10% buffer")
    print(f"{'='*80}\n")


def measure_arithmetic_lengths(min_depth, max_depth, min_val, max_val, format_mode, num_samples=500):
    """Measure sequence lengths for arithmetic task."""
    print(f"\n{'='*80}")
    print(f"Arithmetic Length Analysis")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  min_depth: {min_depth}")
    print(f"  max_depth: {max_depth}")
    print(f"  min_val: {min_val}")
    print(f"  max_val: {max_val}")
    print(f"  format: {format_mode}")
    print(f"  samples: {num_samples}")
    print(f"{'='*80}\n")

    lengths_by_depth = defaultdict(list)
    all_lengths = []

    for _ in range(num_samples):
        depth = random.randint(min_depth, max_depth)
        tree = arithmetic.generate_expression_tree(depth, min_val, max_val)

        text = arithmetic._generate_arithmetic_text(tree, format_mode)
        length = len(text.split())

        lengths_by_depth[depth].append(length)
        all_lengths.append(length)

    # Print per-depth statistics
    print("Per-Depth Statistics:")
    print(f"{'Depth':<8} {'Count':<8} {'Min':<8} {'Max':<8} {'Avg':<8} {'Median':<8}")
    print("-" * 56)

    for depth in range(min_depth, max_depth + 1):
        if depth in lengths_by_depth:
            lens = lengths_by_depth[depth]
            print(f"{depth:<8} {len(lens):<8} {min(lens):<8} {max(lens):<8} "
                  f"{sum(lens)/len(lens):<8.1f} {sorted(lens)[len(lens)//2]:<8}")

    # Print overall statistics
    print(f"\n{'='*80}")
    print("Overall Statistics:")
    print(f"  Total samples: {len(all_lengths)}")
    print(f"  Min length: {min(all_lengths)}")
    print(f"  Max length: {max(all_lengths)}")
    print(f"  Avg length: {sum(all_lengths)/len(all_lengths):.1f}")
    print(f"  Median length: {sorted(all_lengths)[len(all_lengths)//2]}")

    # Percentiles
    sorted_lens = sorted(all_lengths)
    print(f"\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        idx = int(len(sorted_lens) * p / 100)
        print(f"  {p:2d}%: {sorted_lens[idx]}")

    # Recommendations
    recommended_min = max(1, min(all_lengths) - 2)
    recommended_max = max(all_lengths) + max(20, int(max(all_lengths) * 0.1))

    print(f"\n{'='*80}")
    print("Recommended Config:")
    print(f"{'='*80}")
    print(f"  min_train_len: {recommended_min}")
    print(f"  max_train_len: {recommended_max}")
    print(f"\nAdd to configs/data/arithmetic.yaml:")
    print(f"  min_train_len: {recommended_min}  # Measured from depth {min_depth}-{max_depth}, vals {min_val}-{max_val}, format={format_mode}")
    print(f"  max_train_len: {recommended_max}  # With 10% buffer")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Measure sequence lengths for curriculum learning setup"
    )
    parser.add_argument(
        "--task",
        choices=["bfvp", "arithmetic"],
        required=True,
        help="Which task to analyze"
    )
    parser.add_argument(
        "--min_depth",
        type=int,
        default=1,
        help="Minimum tree depth"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum tree depth"
    )
    parser.add_argument(
        "--num_vars",
        type=int,
        default=2,
        help="Number of variables (BFVP only)"
    )
    parser.add_argument(
        "--min_val",
        type=int,
        default=1,
        help="Minimum value (arithmetic only)"
    )
    parser.add_argument(
        "--max_val",
        type=int,
        default=50,
        help="Maximum value (arithmetic only)"
    )
    parser.add_argument(
        "--format",
        choices=["trace", "final_value"],
        default="trace",
        help="Output format"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    if args.task == "bfvp":
        measure_bfvp_lengths(
            args.min_depth,
            args.max_depth,
            args.num_vars,
            args.format,
            args.samples
        )
    elif args.task == "arithmetic":
        measure_arithmetic_lengths(
            args.min_depth,
            args.max_depth,
            args.min_val,
            args.max_val,
            args.format,
            args.samples
        )


if __name__ == "__main__":
    main()
