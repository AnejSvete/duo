#!/usr/bin/env python3
"""
Test noise schedules for correctness and numerical stability.

This script verifies:
1. All schedules satisfy boundary conditions
2. Derivatives are computed correctly (via numerical differentiation)
3. No NaN or Inf values
4. Monotonicity of alpha(t)
"""

import torch


# Copy schedule implementations to avoid import issues
class LogLinear(torch.nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, t):
        t = (1 - self.eps) * t + self.eps
        alpha_t = 1 - t
        dalpha_t = -(1 - self.eps) * torch.ones_like(t)
        return dalpha_t, alpha_t


class Cosine(torch.nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.pi_over_2 = torch.pi / 2

    def forward(self, t):
        t = (1 - self.eps) * t + self.eps
        alpha_t = torch.cos(self.pi_over_2 * t) ** 2
        dalpha_t = -(1 - self.eps) * self.pi_over_2 * torch.sin(torch.pi * t)
        return dalpha_t, alpha_t


class Linear(torch.nn.Module):
    def __init__(self, beta_min=0.1, beta_max=20.0, eps=1e-3):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.eps = eps

    def forward(self, t):
        t = (1 - self.eps) * t + self.eps
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        log_alpha_t = -0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2)
        alpha_t = torch.exp(log_alpha_t)
        dalpha_t = -(1 - self.eps) * 0.5 * alpha_t * beta_t
        return dalpha_t, alpha_t


class Polynomial(torch.nn.Module):
    def __init__(self, power=2.0, eps=1e-3):
        super().__init__()
        self.power = power
        self.eps = eps

    def forward(self, t):
        t = (1 - self.eps) * t + self.eps
        alpha_t = 1 - t ** self.power
        dalpha_t = -(1 - self.eps) * self.power * t ** (self.power - 1)
        return dalpha_t, alpha_t


class Sigmoid(torch.nn.Module):
    def __init__(self, scale=6.0, shift=3.0, eps=1e-3):
        super().__init__()
        self.scale = scale
        self.shift = shift
        self.eps = eps

    def forward(self, t):
        t = (1 - self.eps) * t + self.eps
        x = (1 - t) * self.scale - self.shift
        alpha_t = torch.sigmoid(x)
        dalpha_t = -(1 - self.eps) * self.scale * alpha_t * (1 - alpha_t)
        return dalpha_t, alpha_t


class SquaredCosine(torch.nn.Module):
    def __init__(self, s=0.008, eps=1e-3):
        super().__init__()
        self.s = s
        self.eps = eps
        self.pi_over_2 = torch.pi / 2

    def forward(self, t):
        t = (1 - self.eps) * t + self.eps
        arg = self.pi_over_2 * (t + self.s) / (1 + self.s)
        alpha_t = torch.cos(arg) ** 2
        dalpha_t = -(1 - self.eps) * (torch.pi / (2 * (1 + self.s))) * torch.sin(2 * arg)
        return dalpha_t, alpha_t


def numerical_derivative(schedule, t, h=1e-5):
    """Compute numerical derivative using central difference."""
    t_plus = torch.clamp(t + h, 0, 1)
    t_minus = torch.clamp(t - h, 0, 1)

    with torch.no_grad():
        _, alpha_plus = schedule(t_plus)
        _, alpha_minus = schedule(t_minus)

    return (alpha_plus - alpha_minus) / (2 * h)


def test_schedule(name, schedule, eps=1e-3):
    """Run comprehensive tests on a noise schedule."""
    print(f"\nTesting {name}...")
    print("-" * 60)

    errors = []

    # Test 1: Boundary conditions
    with torch.no_grad():
        t_start = torch.tensor([eps])
        t_end = torch.tensor([1.0 - eps])

        _, alpha_start = schedule(t_start)
        _, alpha_end = schedule(t_end)

    if alpha_start.item() > 0.95:
        print(f"✓ α({eps:.3f}) = {alpha_start.item():.6f} (close to 1)")
    else:
        errors.append(f"α({eps}) = {alpha_start.item():.6f}, expected ≈ 1")

    if alpha_end.item() < 0.1:
        print(f"✓ α({1-eps:.3f}) = {alpha_end.item():.6f} (close to 0)")
    else:
        errors.append(f"α({1-eps}) = {alpha_end.item():.6f}, expected ≈ 0")

    # Test 2: No NaN or Inf
    t_test = torch.linspace(eps, 1 - eps, 100)
    with torch.no_grad():
        dalpha_t, alpha_t = schedule(t_test)

    if not torch.isnan(alpha_t).any() and not torch.isinf(alpha_t).any():
        print(f"✓ No NaN/Inf in α(t)")
    else:
        errors.append("NaN or Inf detected in α(t)")

    if not torch.isnan(dalpha_t).any() and not torch.isinf(dalpha_t).any():
        print(f"✓ No NaN/Inf in dα/dt")
    else:
        errors.append("NaN or Inf detected in dα/dt")

    # Test 3: Monotonicity (alpha should decrease)
    diffs = alpha_t[1:] - alpha_t[:-1]
    if (diffs <= 0).all():
        print(f"✓ α(t) is monotonically decreasing")
    else:
        errors.append("α(t) is not monotonic")

    # Test 4: Derivative correctness (analytical vs numerical)
    t_sample = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    with torch.no_grad():
        dalpha_analytical, _ = schedule(t_sample)
    dalpha_numerical = numerical_derivative(schedule, t_sample)

    relative_error = torch.abs(
        (dalpha_analytical - dalpha_numerical) / (dalpha_analytical + 1e-10)
    )
    max_error = relative_error.max().item()

    if max_error < 0.01:  # 1% tolerance
        print(f"✓ Derivative matches numerical derivative (max error: {max_error:.6f})")
    else:
        errors.append(f"Derivative error too large: {max_error:.6f}")
        print(f"  Analytical: {dalpha_analytical.tolist()}")
        print(f"  Numerical:  {dalpha_numerical.tolist()}")

    # Test 5: dalpha/dt should be negative (signal decays)
    with torch.no_grad():
        dalpha_t, _ = schedule(t_test)

    if (dalpha_t < 0).all():
        print(f"✓ dα/dt < 0 everywhere (signal decays)")
    else:
        errors.append("dα/dt is not always negative")

    # Summary
    if errors:
        print("\n❌ FAILED with errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ ALL TESTS PASSED")
        return True


def main():
    """Run tests on all schedules."""
    print("=" * 80)
    print("NOISE SCHEDULE VALIDATION")
    print("=" * 80)

    schedules = {
        'Log-Linear': LogLinear(eps=1e-3),
        'Cosine': Cosine(eps=1e-3),
        'Linear (DDPM)': Linear(beta_min=0.1, beta_max=20.0, eps=1e-3),
        'Polynomial (p=2)': Polynomial(power=2.0, eps=1e-3),
        'Polynomial (p=0.5)': Polynomial(power=0.5, eps=1e-3),
        'Sigmoid': Sigmoid(scale=6.0, shift=3.0, eps=1e-3),
        'Squared Cosine': SquaredCosine(s=0.008, eps=1e-3),
    }

    results = {}
    for name, schedule in schedules.items():
        results[name] = test_schedule(name, schedule)

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\n{passed}/{total} schedules passed all tests")

    if passed == total:
        print("\n🎉 All schedules are working correctly!")
        return 0
    else:
        print("\n⚠️  Some schedules have issues - please review above")
        return 1


if __name__ == '__main__':
    exit(main())
