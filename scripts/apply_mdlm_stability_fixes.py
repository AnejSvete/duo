#!/usr/bin/env python3
"""
Apply critical stability fixes to MDLM implementation.

This script applies the following fixes:
1. Clamp per-token losses to prevent explosions
2. Increase denominator clamping from 1e-7 to 1e-3
3. Avoid extreme mask ratios that lead to unstable timesteps
4. Add safe numerical operations for log(expm1())

Usage:
    python scripts/apply_mdlm_stability_fixes.py
    python scripts/apply_mdlm_stability_fixes.py --dry-run  # Preview changes only
"""

import argparse
import re
from pathlib import Path


def apply_fix_to_file(file_path, old_pattern, new_content, description):
    """Apply a single fix to a file."""
    with open(file_path, 'r') as f:
        content = f.read()

    if isinstance(old_pattern, str):
        # Simple string replacement
        if old_pattern in content:
            new_file_content = content.replace(old_pattern, new_content)
            return new_file_content, True, description
        else:
            return content, False, f"Pattern not found: {old_pattern[:50]}..."
    else:
        # Regex replacement
        match = old_pattern.search(content)
        if match:
            new_file_content = old_pattern.sub(new_content, content)
            return new_file_content, True, description
        else:
            return content, False, f"Pattern not found: {old_pattern.pattern[:50]}..."


def apply_fixes(dry_run=False):
    """Apply all critical fixes."""
    fixes_applied = []
    fixes_failed = []

    print("="*80)
    print("MDLM Stability Fixes")
    print("="*80)
    print()

    # Fix 1: Clamp per-token losses in MDLM
    print("Fix 1: Clamping per-token losses in MDLM...")
    algo_path = Path("algo.py")

    old_code = "        return log_p_theta * dalpha_t / denominator"
    new_code = """        # Clamp per-token loss to prevent extreme values
        loss_per_token = log_p_theta * dalpha_t / denominator
        return loss_per_token.clamp(min=-20.0, max=20.0)"""

    new_content, success, msg = apply_fix_to_file(algo_path, old_code, new_code, "Clamp MDLM per-token loss")

    if success:
        if not dry_run:
            with open(algo_path, 'w') as f:
                f.write(new_content)
        fixes_applied.append(f"✓ {msg}")
        print(f"  ✓ {msg}")
    else:
        fixes_failed.append(f"✗ {msg}")
        print(f"  ✗ {msg}")

    # Fix 2: Increase denominator clamping
    print("\nFix 2: Increasing denominator clamping...")

    old_code = "        denominator = (1 - alpha_t).clamp(min=1e-7)"
    new_code = "        denominator = (1 - alpha_t).clamp(min=1e-3)  # Increased from 1e-7 for stability"

    new_content, success, msg = apply_fix_to_file(algo_path, old_code, new_code, "Increase denominator clamping")

    if success:
        if not dry_run:
            with open(algo_path, 'w') as f:
                f.write(new_content)
        fixes_applied.append(f"✓ {msg}")
        print(f"  ✓ {msg}")
    else:
        fixes_failed.append(f"✗ {msg}")
        print(f"  ✗ {msg}")

    # Fix 3: Clamp mask ratios in diffusion.py
    print("\nFix 3: Clamping mask ratios to avoid extreme timesteps...")
    diffusion_path = Path("diffusion.py")

    old_code = "            mask_ratio = (mask_counts / num_maskable_tokens).clamp(0.0, 1.0)"
    new_code = "            mask_ratio = (mask_counts / num_maskable_tokens).clamp(0.05, 0.95)  # Avoid extreme timesteps"

    new_content, success, msg = apply_fix_to_file(diffusion_path, old_code, new_code, "Clamp mask ratios")

    if success:
        if not dry_run:
            with open(diffusion_path, 'w') as f:
                f.write(new_content)
        fixes_applied.append(f"✓ {msg}")
        print(f"  ✓ {msg}")
    else:
        fixes_failed.append(f"✗ {msg}")
        print(f"  ✗ {msg}")

    # Fix 4: Clamp model outputs before normalization
    print("\nFix 4: Clamping model outputs before normalization...")

    # Find the line that does normalization in _process_model_output
    old_pattern = re.compile(
        r"(def _process_model_output\(self, model_output, xt, sigma\):.*?)"
        r"(model_output\[:, :, self\.mask_index\] \+= self\.neg_infinity\s+# Normalize to log-probabilities)",
        re.DOTALL
    )

    new_code = r"""\1model_output[:, :, self.mask_index] += self.neg_infinity
        # Clamp logits before normalization to prevent extreme values
        model_output = model_output.clamp(min=-50.0, max=50.0)
        # Normalize to log-probabilities"""

    new_content, success, msg = apply_fix_to_file(algo_path, old_pattern, new_code, "Clamp model outputs")

    if success:
        if not dry_run:
            with open(algo_path, 'w') as f:
                f.write(new_content)
        fixes_applied.append(f"✓ {msg}")
        print(f"  ✓ {msg}")
    else:
        # Try simpler replacement
        with open(algo_path, 'r') as f:
            content = f.read()

        if "model_output = model_output - torch.logsumexp(" in content:
            old_simple = "        model_output[:, :, self.mask_index] += self.neg_infinity\n        # Normalize to log-probabilities\n        model_output = model_output - torch.logsumexp("
            new_simple = "        model_output[:, :, self.mask_index] += self.neg_infinity\n        # Clamp logits before normalization\n        model_output = model_output.clamp(min=-50.0, max=50.0)\n        # Normalize to log-probabilities\n        model_output = model_output - torch.logsumexp("

            if old_simple in content:
                new_content = content.replace(old_simple, new_simple)
                if not dry_run:
                    with open(algo_path, 'w') as f:
                        f.write(new_content)
                fixes_applied.append(f"✓ Clamp model outputs (simple replacement)")
                print(f"  ✓ Clamp model outputs (simple replacement)")
            else:
                fixes_failed.append(f"✗ {msg}")
                print(f"  ✗ {msg}")
        else:
            fixes_failed.append(f"✗ {msg}")
            print(f"  ✗ {msg}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if dry_run:
        print("\n🔍 DRY RUN MODE - No files were modified")

    print(f"\n✅ Fixes Applied: {len(fixes_applied)}")
    for fix in fixes_applied:
        print(f"   {fix}")

    if fixes_failed:
        print(f"\n❌ Fixes Failed: {len(fixes_failed)}")
        for fix in fixes_failed:
            print(f"   {fix}")
        print("\n⚠️  Some fixes could not be applied automatically.")
        print("   Please review MDLM_STABILITY_ISSUES.md and apply manually.")

    if not dry_run and fixes_applied:
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Review the changes:")
        print("   git diff algo.py diffusion.py")
        print("\n2. Test the fixes:")
        print("   python main.py data=bfvp algo=mdlm")
        print("\n3. Monitor training:")
        print("   - Watch for loss spikes")
        print("   - Check auto-recovery logs")
        print("   - Verify losses stay < 20.0 per token")
        print("\n4. If training is stable, run grid search:")
        print("   ./scripts/quick_grid_search.sh bfvp recommended")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Apply MDLM stability fixes")
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview changes without modifying files")
    args = parser.parse_args()

    apply_fixes(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
