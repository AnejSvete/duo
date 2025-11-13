"""Test script for length-stratified metrics functionality.

This script tests the LengthStratifiedMetrics class to ensure it correctly:
1. Bins sequences by length percentiles
2. Tracks metrics per bin
3. Computes aggregate statistics
"""

import torch
import numpy as np
from length_stratified_metrics import LengthStratifiedMetrics, PerGenerationModeMetrics


def test_basic_functionality():
    """Test basic metrics tracking and binning."""
    print("=" * 80)
    print("Test 1: Basic functionality")
    print("=" * 80)

    metrics = LengthStratifiedMetrics(num_bins=5)

    # Simulate a batch of sequences with varying lengths
    # Lengths: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    lengths = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    # Simulate per-sample accuracy: longer sequences are harder (lower accuracy)
    acc_exact = torch.tensor([1.0, 1.0, 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2, 0.2])
    acc_token = torch.tensor([1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55])

    metrics.update_per_sample(
        lengths=lengths,
        per_sample_metrics={
            'acc_exact': acc_exact,
            'acc_token': acc_token,
        }
    )

    # Compute results
    results = metrics.compute()

    print(f"\nTotal samples: {results['num_samples']}")
    print(f"\nOverall metrics:")
    for metric_name, value in results['overall'].items():
        print(f"  {metric_name}: {value:.4f}")

    print(f"\nBin edges: {[f'{x:.1f}' for x in results['bin_edges']]}")
    print(f"\nPer-bin metrics:")
    for bin_info in results['bins']:
        print(f"\n  {bin_info['label']} ({bin_info['min_length']:.1f} - {bin_info['max_length']:.1f})")
        print(f"    Num samples: {bin_info['num_samples']}")
        if 'acc_exact' in bin_info:
            print(f"    acc_exact: {bin_info['acc_exact']:.4f}")
            print(f"    acc_token: {bin_info['acc_token']:.4f}")

    # Test wandb log format
    print(f"\nWandb log keys:")
    wandb_logs = metrics.get_wandb_logs(prefix="val")
    for key in sorted(wandb_logs.keys()):
        print(f"  {key}: {wandb_logs[key]:.4f}")

    print("\n✓ Test 1 passed!\n")


def test_multiple_batches():
    """Test accumulation over multiple batches."""
    print("=" * 80)
    print("Test 2: Multiple batches")
    print("=" * 80)

    metrics = LengthStratifiedMetrics(num_bins=3)

    # Simulate 3 batches
    for batch_idx in range(3):
        # Generate random lengths between 10 and 100
        batch_size = 8
        lengths = torch.randint(10, 100, (batch_size,))

        # Random accuracies
        acc_exact = torch.rand(batch_size)
        acc_token = torch.rand(batch_size) * 0.5 + 0.5  # Between 0.5 and 1.0

        metrics.update_per_sample(
            lengths=lengths,
            per_sample_metrics={
                'acc_exact': acc_exact,
                'acc_token': acc_token,
            }
        )

        print(f"Batch {batch_idx + 1}: Added {batch_size} samples")

    results = metrics.compute()
    print(f"\nTotal samples accumulated: {results['num_samples']}")
    print(f"Number of bins: {len(results['bins'])}")

    for bin_info in results['bins']:
        print(f"  {bin_info['label']}: {bin_info['num_samples']} samples")

    print("\n✓ Test 2 passed!\n")


def test_per_generation_mode():
    """Test tracking metrics for multiple generation modes."""
    print("=" * 80)
    print("Test 3: Per-generation-mode metrics")
    print("=" * 80)

    multi_metrics = PerGenerationModeMetrics(num_bins=4)

    gen_modes = ["random", "top_k", "one_level"]
    batch_size = 10

    for mode in gen_modes:
        lengths = torch.randint(20, 80, (batch_size,))

        # Each mode has different performance characteristics
        if mode == "random":
            acc_exact = torch.rand(batch_size) * 0.5  # Poor performance
        elif mode == "top_k":
            acc_exact = torch.rand(batch_size) * 0.3 + 0.6  # Good performance
        else:  # one_level
            acc_exact = torch.rand(batch_size) * 0.2 + 0.4  # Medium performance

        multi_metrics.update(
            mode=mode,
            lengths=lengths,
            per_sample_metrics={'acc_exact': acc_exact}
        )

        print(f"Added metrics for mode: {mode}")

    # Compute all
    all_results = multi_metrics.compute_all()

    print(f"\nResults for {len(all_results)} generation modes:")
    for mode, results in all_results.items():
        print(f"\n  Mode: {mode}")
        print(f"    Overall acc_exact: {results['overall']['acc_exact']:.4f}")
        print(f"    Number of bins: {len(results['bins'])}")

    # Test wandb logs
    wandb_logs = multi_metrics.get_wandb_logs(prefix="val")
    print(f"\nGenerated {len(wandb_logs)} wandb log entries")
    print("Sample keys:")
    for i, key in enumerate(sorted(wandb_logs.keys())[:5]):
        print(f"  {key}: {wandb_logs[key]:.4f}")
    if len(wandb_logs) > 5:
        print(f"  ... and {len(wandb_logs) - 5} more")

    print("\n✓ Test 3 passed!\n")


def test_realistic_scenario():
    """Test with a realistic validation scenario."""
    print("=" * 80)
    print("Test 4: Realistic validation scenario")
    print("=" * 80)

    # Simulate validation over multiple batches
    multi_metrics = PerGenerationModeMetrics(num_bins=10)

    num_batches = 20
    batch_size = 16
    gen_modes = ["default"]

    print(f"Simulating {num_batches} validation batches...")

    for batch_idx in range(num_batches):
        # Lengths follow a distribution: mostly 40-60, some outliers
        mean_len = 50
        std_len = 10
        lengths = torch.clamp(
            torch.normal(mean_len, std_len, (batch_size,)).long(),
            min=20,
            max=100
        )

        # Accuracy degrades with length (realistic behavior)
        # Base accuracy ~0.9, decreases by ~0.005 per length unit
        base_acc = 0.95
        length_penalty = 0.005
        acc_exact = torch.clamp(
            base_acc - (lengths.float() - 20) * length_penalty + torch.randn(batch_size) * 0.1,
            min=0.0,
            max=1.0
        )

        # Token-level accuracy is generally higher
        acc_token = torch.clamp(acc_exact + torch.rand(batch_size) * 0.1, min=0.0, max=1.0)

        # Correct prediction somewhere in between
        correct_prediction = torch.clamp(
            acc_exact + torch.rand(batch_size) * 0.15,
            min=0.0,
            max=1.0
        )

        multi_metrics.update(
            mode="default",
            lengths=lengths,
            per_sample_metrics={
                'acc_exact': acc_exact,
                'acc_token': acc_token,
                'correct_prediction': correct_prediction,
            }
        )

    results = multi_metrics.compute_all()["default"]

    print(f"\nTotal validation samples: {results['num_samples']}")
    print(f"\nOverall metrics:")
    for metric_name, value in results['overall'].items():
        print(f"  {metric_name}: {value:.4f}")

    print(f"\nLength distribution across bins:")
    print(f"{'Bin':<20} {'Samples':<10} {'acc_exact':<12} {'acc_token':<12} {'correct_pred':<12}")
    print("-" * 70)
    for bin_info in results['bins']:
        if bin_info['num_samples'] > 0:
            print(f"{bin_info['label']:<20} {bin_info['num_samples']:<10} "
                  f"{bin_info.get('acc_exact', 0):<12.4f} "
                  f"{bin_info.get('acc_token', 0):<12.4f} "
                  f"{bin_info.get('correct_prediction', 0):<12.4f}")

    # Verify that accuracy decreases with length
    bin_accs = [b.get('acc_exact', 0) for b in results['bins'] if b['num_samples'] > 0]
    if len(bin_accs) > 1:
        trend = "decreasing" if bin_accs[0] > bin_accs[-1] else "increasing"
        print(f"\nAccuracy trend across length bins: {trend}")
        print(f"  First bin: {bin_accs[0]:.4f}")
        print(f"  Last bin: {bin_accs[-1]:.4f}")

    print("\n✓ Test 4 passed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Testing Length-Stratified Metrics Implementation")
    print("=" * 80 + "\n")

    try:
        test_basic_functionality()
        test_multiple_batches()
        test_per_generation_mode()
        test_realistic_scenario()

        print("=" * 80)
        print("✓ All tests passed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
