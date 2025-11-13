"""
Test script for curriculum learning callback.
"""

import sys

import torch
from omegaconf import OmegaConf

from curriculum_callback import CurriculumLearningCallback


def test_bin_boundaries():
    """Test that bin boundaries are computed correctly."""
    print("Testing bin boundary computation...")

    # Test case 1: 4 bins, no overlap
    callback = CurriculumLearningCallback(
        enabled=True,
        num_bins=4,
        epochs_per_bin=5,
        overlap=0.0,
        min_train_len=16,
        max_train_len=64,
    )
    callback._compute_bin_boundaries()

    print("\n4 bins, no overlap, range [16, 64]:")
    for i, (start, end) in enumerate(callback.bin_boundaries):
        print(f"  Bin {i+1}: [{start}, {end}] (size: {end-start})")

    expected = [(16, 28), (28, 40), (40, 52), (52, 64)]
    for i, (expected_range, actual_range) in enumerate(
        zip(expected, callback.bin_boundaries)
    ):
        assert (
            abs(actual_range[0] - expected_range[0]) <= 1
        ), f"Bin {i} start mismatch: {actual_range[0]} vs {expected_range[0]}"

    # Test case 2: 4 bins, 20% overlap
    callback = CurriculumLearningCallback(
        enabled=True,
        num_bins=4,
        epochs_per_bin=5,
        overlap=0.2,
        min_train_len=16,
        max_train_len=64,
    )
    callback._compute_bin_boundaries()

    print("\n4 bins, 20% overlap, range [16, 64]:")
    for i, (start, end) in enumerate(callback.bin_boundaries):
        print(f"  Bin {i+1}: [{start}, {end}] (size: {end-start})")

    # Verify overlap exists between consecutive bins
    for i in range(len(callback.bin_boundaries) - 1):
        curr_end = callback.bin_boundaries[i][1]
        next_start = callback.bin_boundaries[i + 1][0]
        assert (
            curr_end > next_start
        ), f"No overlap between bin {i} and {i+1}: {curr_end} <= {next_start}"

    # Test case 3: Single bin (should cover full range)
    callback = CurriculumLearningCallback(
        enabled=True,
        num_bins=1,
        epochs_per_bin=10,
        overlap=0.0,
        min_train_len=16,
        max_train_len=64,
    )
    callback._compute_bin_boundaries()

    print("\n1 bin, range [16, 64]:")
    for i, (start, end) in enumerate(callback.bin_boundaries):
        print(f"  Bin {i+1}: [{start}, {end}]")

    assert callback.bin_boundaries[0] == (16, 64), "Single bin should cover full range"

    print("\n✓ All bin boundary tests passed!")


def test_state_dict():
    """Test that callback state can be saved and loaded."""
    print("\nTesting state dict save/load...")

    callback = CurriculumLearningCallback(
        enabled=True,
        num_bins=4,
        epochs_per_bin=5,
        overlap=0.2,
        min_train_len=16,
        max_train_len=64,
    )
    callback._compute_bin_boundaries()

    # Simulate some progress
    callback.current_bin = 2
    callback.epochs_in_current_bin = 3

    # Save state
    state = callback.state_dict()

    # Create new callback and load state
    new_callback = CurriculumLearningCallback(
        enabled=True,
        num_bins=4,
        epochs_per_bin=5,
        overlap=0.2,
        min_train_len=16,
        max_train_len=64,
    )
    new_callback.load_state_dict(state)

    # Verify state was restored
    assert new_callback.current_bin == 2, "Current bin not restored"
    assert new_callback.epochs_in_current_bin == 3, "Epochs in bin not restored"
    assert (
        new_callback.bin_boundaries == callback.bin_boundaries
    ), "Bin boundaries not restored"

    print("✓ State dict test passed!")


def test_configuration():
    """Test that configuration is properly parsed."""
    print("\nTesting configuration parsing...")

    # Simulate a config
    config = OmegaConf.create(
        {
            "curriculum": {
                "enabled": True,
                "num_bins": 3,
                "epochs_per_bin": 10,
                "overlap": 0.1,
            },
            "data": {"properties": {"min_train_len": 20, "max_train_len": 100}},
        }
    )

    callback = CurriculumLearningCallback(
        enabled=config.curriculum.enabled,
        num_bins=config.curriculum.num_bins,
        epochs_per_bin=config.curriculum.epochs_per_bin,
        overlap=config.curriculum.overlap,
        min_train_len=config.data.properties.min_train_len,
        max_train_len=config.data.properties.max_train_len,
    )

    assert callback.enabled == True
    assert callback.num_bins == 3
    assert callback.epochs_per_bin == 10
    assert callback.overlap == 0.1
    assert callback.min_train_len == 20
    assert callback.max_train_len == 100

    print("✓ Configuration test passed!")


if __name__ == "__main__":
    try:
        test_bin_boundaries()
        test_state_dict()
        test_configuration()
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
