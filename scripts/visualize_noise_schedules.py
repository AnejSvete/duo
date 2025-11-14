#!/usr/bin/env python3
"""
Visualize different noise schedules to understand their behavior.

This script plots alpha(t), dalpha/dt, and the masking probability (1 - alpha)
for all available noise schedules.

Usage:
    python scripts/visualize_noise_schedules.py
"""

import matplotlib.pyplot as plt
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer_base import LogLinear, Cosine, Linear, Polynomial, Sigmoid, SquaredCosine


def visualize_schedules():
    """Create comparison plots of all noise schedules."""

    # Define schedules to compare
    schedules = {
        'Log-Linear': LogLinear(eps=1e-3),
        'Cosine': Cosine(eps=1e-3),
        'Linear (DDPM)': Linear(beta_min=0.1, beta_max=20.0, eps=1e-3),
        'Polynomial (p=2)': Polynomial(power=2.0, eps=1e-3),
        'Polynomial (p=0.5)': Polynomial(power=0.5, eps=1e-3),
        'Polynomial (p=3)': Polynomial(power=3.0, eps=1e-3),
        'Sigmoid': Sigmoid(scale=6.0, shift=3.0, eps=1e-3),
        'Squared Cosine': SquaredCosine(s=0.008, eps=1e-3),
    }

    # Time points
    t = torch.linspace(0.001, 0.999, 1000)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Noise Schedule Comparison', fontsize=16, fontweight='bold')

    # Plot 1: alpha(t) - Signal level
    ax1 = axes[0, 0]
    ax1.set_title('Signal Level α(t)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('α(t)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: 1 - alpha(t) - Noise/Masking level
    ax2 = axes[0, 1]
    ax2.set_title('Noise Level (1 - α(t))', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('1 - α(t)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: dalpha/dt - Rate of change
    ax3 = axes[1, 0]
    ax3.set_title('Rate of Change dα/dt', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('dα/dt')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Effective noise rate
    ax4 = axes[1, 1]
    ax4.set_title('Effective Noise Rate |dα/dt| / α(t)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time t')
    ax4.set_ylabel('|dα/dt| / α(t)')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    # Plot each schedule
    for name, schedule in schedules.items():
        with torch.no_grad():
            dalpha_t, alpha_t = schedule(t)

        alpha_t_np = alpha_t.cpu().numpy()
        dalpha_t_np = dalpha_t.cpu().numpy()
        t_np = t.cpu().numpy()

        # Plot alpha(t)
        ax1.plot(t_np, alpha_t_np, label=name, linewidth=2)

        # Plot 1 - alpha(t) (noise level / masking probability)
        ax2.plot(t_np, 1 - alpha_t_np, label=name, linewidth=2)

        # Plot dalpha/dt
        ax3.plot(t_np, dalpha_t_np, label=name, linewidth=2)

        # Plot effective noise rate (avoid division by zero)
        effective_rate = abs(dalpha_t_np) / (alpha_t_np + 1e-10)
        ax4.plot(t_np, effective_rate, label=name, linewidth=2)

    # Add legends
    ax1.legend(loc='best', fontsize=8)
    ax2.legend(loc='best', fontsize=8)
    ax3.legend(loc='best', fontsize=8)
    ax4.legend(loc='best', fontsize=8)

    plt.tight_layout()

    # Save figure
    output_path = 'noise_schedules_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")

    # Also create individual plots for key schedules
    create_individual_plots()

    plt.show()


def create_individual_plots():
    """Create detailed individual plots for recommended schedules."""

    recommended = {
        'Log-Linear (Current)': LogLinear(eps=1e-3),
        'Cosine (Recommended)': Cosine(eps=1e-3),
        'Squared Cosine (DDPM)': SquaredCosine(s=0.008, eps=1e-3),
    }

    t = torch.linspace(0.001, 0.999, 1000)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Recommended Noise Schedules for Discrete Diffusion', fontsize=14, fontweight='bold')

    for idx, (name, schedule) in enumerate(recommended.items()):
        ax = axes[idx]

        with torch.no_grad():
            dalpha_t, alpha_t = schedule(t)

        alpha_t_np = alpha_t.cpu().numpy()
        t_np = t.cpu().numpy()
        noise_level = 1 - alpha_t_np

        # Plot both alpha and noise on same axis
        ax.plot(t_np, alpha_t_np, label='α(t) (signal)', linewidth=2.5, color='blue')
        ax.plot(t_np, noise_level, label='1-α(t) (noise)', linewidth=2.5,
                linestyle='--', color='red')

        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time t', fontsize=10)
        ax.set_ylabel('Level', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()

    output_path = 'recommended_schedules.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved recommended schedules to {output_path}")


def print_schedule_properties():
    """Print numerical properties of each schedule at key time points."""

    print("\n" + "="*80)
    print("NOISE SCHEDULE PROPERTIES AT KEY TIME POINTS")
    print("="*80 + "\n")

    schedules = {
        'Log-Linear': LogLinear(eps=1e-3),
        'Cosine': Cosine(eps=1e-3),
        'Linear (DDPM)': Linear(beta_min=0.1, beta_max=20.0, eps=1e-3),
        'Squared Cosine': SquaredCosine(s=0.008, eps=1e-3),
    }

    time_points = [0.1, 0.25, 0.5, 0.75, 0.9]

    for name, schedule in schedules.items():
        print(f"{name}:")
        print("-" * 60)
        print(f"{'t':<8} {'α(t)':<12} {'1-α(t)':<12} {'|dα/dt|':<12} {'rate':<12}")
        print("-" * 60)

        for t_val in time_points:
            t = torch.tensor([t_val])
            with torch.no_grad():
                dalpha_t, alpha_t = schedule(t)

            rate = abs(dalpha_t.item()) / (alpha_t.item() + 1e-10)

            print(f"{t_val:<8.2f} {alpha_t.item():<12.6f} {1-alpha_t.item():<12.6f} "
                  f"{abs(dalpha_t.item()):<12.6f} {rate:<12.6f}")

        print("\n")


if __name__ == '__main__':
    print("Visualizing noise schedules...")
    print_schedule_properties()
    visualize_schedules()
