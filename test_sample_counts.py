"""Test to verify that sample counts are correctly reported in bins."""

import torch
from length_stratified_metrics import LengthStratifiedMetrics, PerGenerationModeMetrics


def test_sample_count_accuracy():
    """Verify that bin sample counts sum to total and are distributed correctly."""
    print("=" * 80)
    print("Test: Sample Count Accuracy")
    print("=" * 80)

    metrics = LengthStratifiedMetrics(num_bins=4)

    # Simulate validation with known distribution
    # 10 batches of varying sizes
    batch_sizes = [32, 28, 35, 30, 33, 31, 29, 34, 32, 36]
    total_expected = sum(batch_sizes)

    print(f"\nSimulating {len(batch_sizes)} batches:")
    for i, batch_size in enumerate(batch_sizes):
        lengths = torch.randint(20, 100, (batch_size,))
        acc = torch.rand(batch_size)

        metrics.update_per_sample(
            lengths=lengths,
            per_sample_metrics={'acc_exact': acc}
        )
        print(f"  Batch {i+1}: {batch_size} samples")

    print(f"\nTotal samples added: {total_expected}")

    # Compute results
    results = metrics.compute()

    print(f"\nResults:")
    print(f"  Total reported: {results['num_samples']}")
    print(f"  Expected: {total_expected}")

    # Check bin counts
    print(f"\nBin distribution:")
    bin_counts = []
    for bin_info in results['bins']:
        count = bin_info['num_samples']
        bin_counts.append(count)
        print(f"  {bin_info['label']}: {count} samples")

    bin_sum = sum(bin_counts)
    print(f"\nSum of bin counts: {bin_sum}")

    # Verify
    success = True
    if results['num_samples'] != total_expected:
        print(f"\n✗ ERROR: Total samples mismatch!")
        print(f"  Expected: {total_expected}")
        print(f"  Got: {results['num_samples']}")
        success = False

    if bin_sum != total_expected:
        print(f"\n✗ ERROR: Bin counts don't sum to total!")
        print(f"  Expected sum: {total_expected}")
        print(f"  Got sum: {bin_sum}")
        success = False

    if bin_sum != results['num_samples']:
        print(f"\n✗ ERROR: Bin sum doesn't match total!")
        print(f"  Bin sum: {bin_sum}")
        print(f"  Total: {results['num_samples']}")
        success = False

    # Check that bins are roughly equal (for quartiles with uniform distribution)
    expected_per_bin = total_expected / 4
    tolerance = 0.15  # Allow 15% deviation
    for i, count in enumerate(bin_counts):
        deviation = abs(count - expected_per_bin) / expected_per_bin
        if deviation > tolerance:
            print(f"\nWarning: Bin {i} has {count} samples, expected ~{expected_per_bin:.0f} (deviation: {deviation:.1%})")

    if success:
        print(f"\n✓ All sample counts are correct!")
        print(f"✓ {total_expected} samples properly distributed across {len(bin_counts)} bins")
    else:
        print(f"\n✗ Sample count verification FAILED")

    return success


def test_multiple_modes_sample_counts():
    """Test sample counts with multiple generation modes."""
    print("\n" + "=" * 80)
    print("Test: Multiple Modes Sample Counts")
    print("=" * 80)

    multi_metrics = PerGenerationModeMetrics(num_bins=4)

    modes = ['default', 'top_k', 'random']
    total_samples = 200

    print(f"\nAdding {total_samples} samples for each of {len(modes)} modes:")

    for mode in modes:
        # Add samples in multiple batches
        batches = [50, 50, 50, 50]  # 4 batches of 50 each
        for batch_size in batches:
            lengths = torch.randint(20, 100, (batch_size,))
            acc = torch.rand(batch_size)

            multi_metrics.update(
                mode=mode,
                lengths=lengths,
                per_sample_metrics={'acc_exact': acc}
            )

        print(f"  {mode}: {total_samples} samples")

    # Compute all
    all_results = multi_metrics.compute_all()

    print(f"\nVerifying sample counts per mode:")
    all_success = True

    for mode, results in all_results.items():
        total = results['num_samples']
        bin_sum = sum(b['num_samples'] for b in results['bins'])

        print(f"\n  {mode}:")
        print(f"    Total: {total}")
        print(f"    Bin sum: {bin_sum}")
        print(f"    Expected: {total_samples}")

        if total != total_samples:
            print(f"    ✗ Total mismatch!")
            all_success = False
        elif bin_sum != total_samples:
            print(f"    ✗ Bin sum mismatch!")
            all_success = False
        else:
            print(f"    ✓ Correct")

    if all_success:
        print(f"\n✓ All modes have correct sample counts!")
    else:
        print(f"\n✗ Some modes have incorrect sample counts")

    return all_success


def test_edge_case_empty_bins():
    """Test that empty bins report 0 samples correctly."""
    print("\n" + "=" * 80)
    print("Test: Empty Bins Sample Counts")
    print("=" * 80)

    metrics = LengthStratifiedMetrics(num_bins=4)

    # Create data where some bins might be empty
    # All samples have length ~50
    lengths = torch.full((50,), 50) + torch.randint(-2, 3, (50,))
    acc = torch.rand(50)

    metrics.update_per_sample(
        lengths=lengths,
        per_sample_metrics={'acc_exact': acc}
    )

    results = metrics.compute()

    print(f"\nTotal samples: {results['num_samples']}")
    print(f"\nBin distribution (with concentrated lengths):")

    non_empty_bins = 0
    total_in_bins = 0

    for bin_info in results['bins']:
        count = bin_info['num_samples']
        total_in_bins += count
        if count > 0:
            non_empty_bins += 1
        status = "✓" if count >= 0 else "✗"
        print(f"  {status} {bin_info['label']}: {count} samples")

    print(f"\nNon-empty bins: {non_empty_bins}/{len(results['bins'])}")
    print(f"Total in bins: {total_in_bins}")
    print(f"Expected total: {results['num_samples']}")

    success = (total_in_bins == results['num_samples'] == 50)

    if success:
        print(f"\n✓ Empty bins handled correctly!")
    else:
        print(f"\n✗ Empty bin handling has issues")

    return success


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SAMPLE COUNT VERIFICATION TESTS")
    print("=" * 80 + "\n")

    try:
        test1 = test_sample_count_accuracy()
        test2 = test_multiple_modes_sample_counts()
        test3 = test_edge_case_empty_bins()

        print("\n" + "=" * 80)
        if test1 and test2 and test3:
            print("✓ ALL SAMPLE COUNT TESTS PASSED!")
            print("\nThe issue with reporting all samples for all lengths is FIXED.")
            print("Each bin now correctly reports only its own sample count.")
        else:
            print("✗ SOME TESTS FAILED")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
