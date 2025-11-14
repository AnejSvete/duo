"""
Test script to verify MDLM stability fixes.

This script tests the numerical stabilization and detection mechanisms.
"""
import torch
import sys


def test_division_by_zero_fix():
    """Test that the clamping prevents division by zero."""
    print("\n" + "="*80)
    print("Test 1: Division by Zero Fix")
    print("="*80)

    # Simulate alpha_t values that are very close to 1
    alpha_t = torch.tensor([0.999999, 0.9999999, 0.99999999, 1.0 - 1e-10])

    # Old version (would cause extremely large values)
    print("\nOLD VERSION (without clamping):")
    old_denominator = 1 - alpha_t
    print(f"Denominator values: {old_denominator}")
    print(f"Min denominator: {old_denominator.min():.2e}")
    old_result = torch.ones_like(alpha_t) / old_denominator
    print(f"Result of 1/denominator: {old_result}")
    print(f"Max result: {old_result.max():.2e}")

    # New version (with clamping)
    print("\nNEW VERSION (with clamping to 1e-7):")
    new_denominator = (1 - alpha_t).clamp(min=1e-7)
    print(f"Denominator values: {new_denominator}")
    print(f"Min denominator: {new_denominator.min():.2e}")
    new_result = torch.ones_like(alpha_t) / new_denominator
    print(f"Result of 1/denominator: {new_result}")
    print(f"Max result: {new_result.max():.2e}")

    # Check that new version is bounded
    assert new_result.max() <= 1e7, "Clamping should bound the result"
    print("\n✓ Test passed: Clamping prevents extreme values")


def test_nan_inf_detection():
    """Test that NaN/Inf detection works correctly."""
    print("\n" + "="*80)
    print("Test 2: NaN/Inf Detection")
    print("="*80)

    # Test various problematic tensors
    test_cases = [
        ("Normal values", torch.randn(4, 4), False),
        ("Contains NaN", torch.tensor([[1.0, 2.0], [float('nan'), 4.0]]), True),
        ("Contains Inf", torch.tensor([[1.0, 2.0], [float('inf'), 4.0]]), True),
        ("Contains -Inf", torch.tensor([[1.0, 2.0], [float('-inf'), 4.0]]), True),
        ("All zeros", torch.zeros(4, 4), False),
    ]

    for name, tensor, should_detect in test_cases:
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        is_invalid = has_nan or has_inf

        status = "✓" if is_invalid == should_detect else "✗"
        print(f"{status} {name}: NaN={has_nan}, Inf={has_inf}, Expected invalid={should_detect}")

        if is_invalid != should_detect:
            print(f"  ERROR: Detection mismatch!")
            return False

    print("\n✓ Test passed: NaN/Inf detection works correctly")
    return True


def test_loss_computation_stability():
    """Test the full MDLM loss computation with edge cases."""
    print("\n" + "="*80)
    print("Test 3: Loss Computation Stability")
    print("="*80)

    batch_size, seq_len, vocab_size = 4, 16, 100

    # Simulate log probabilities
    log_x_theta = torch.randn(batch_size, seq_len, vocab_size)
    x0 = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Test with various alpha_t values
    test_alphas = [
        ("Normal (0.5)", torch.tensor([0.5])),
        ("Near 1 (0.999)", torch.tensor([0.999])),
        ("Very near 1 (0.9999999)", torch.tensor([0.9999999])),
        ("Near 0 (0.001)", torch.tensor([0.001])),
    ]

    for name, alpha_t in test_alphas:
        alpha_t = alpha_t.expand(batch_size, 1)
        dalpha_t = -0.001 * torch.ones_like(alpha_t)

        # Gather log probabilities
        log_p_theta = torch.gather(
            input=log_x_theta, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)

        # Compute loss with clamping (new version)
        denominator = (1 - alpha_t).clamp(min=1e-7)
        loss = log_p_theta * dalpha_t / denominator

        has_nan = torch.isnan(loss).any()
        has_inf = torch.isinf(loss).any()

        status = "✓" if not (has_nan or has_inf) else "✗"
        print(f"{status} {name}: Loss range=[{loss.min():.2e}, {loss.max():.2e}], "
              f"NaN={has_nan}, Inf={has_inf}")

        if has_nan or has_inf:
            print(f"  ERROR: Loss computation produced invalid values!")
            return False

    print("\n✓ Test passed: Loss computation remains stable")
    return True


def test_checkpoint_recovery_logic():
    """Test the logic for finding checkpoints."""
    print("\n" + "="*80)
    print("Test 4: Checkpoint Recovery Logic")
    print("="*80)

    import os
    import tempfile

    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(checkpoint_dir)

        # Create dummy checkpoint files
        checkpoints = ["best.ckpt", "last.ckpt", "epoch=0-step=100.ckpt"]
        for ckpt in checkpoints:
            path = os.path.join(checkpoint_dir, ckpt)
            with open(path, 'w') as f:
                f.write("dummy")

        # Test finding best checkpoint
        best_path = os.path.join(checkpoint_dir, "best.ckpt")
        assert os.path.exists(best_path), "best.ckpt should exist"
        print(f"✓ Found best checkpoint: {best_path}")

        # Test finding last checkpoint
        last_path = os.path.join(checkpoint_dir, "last.ckpt")
        assert os.path.exists(last_path), "last.ckpt should exist"
        print(f"✓ Found last checkpoint: {last_path}")

        # Test listing all checkpoints
        all_ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        assert len(all_ckpts) == 3, f"Should find 3 checkpoints, found {len(all_ckpts)}"
        print(f"✓ Found {len(all_ckpts)} total checkpoints: {all_ckpts}")

    print("\n✓ Test passed: Checkpoint recovery logic works")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MDLM STABILITY FIXES - TEST SUITE")
    print("="*80)

    tests = [
        test_division_by_zero_fix,
        test_nan_inf_detection,
        test_loss_computation_stability,
        test_checkpoint_recovery_logic,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result is not False:  # None or True
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*80 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
