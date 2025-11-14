# Noise Schedules for Discrete Diffusion

This document describes the available noise schedules and provides guidance on choosing the right one for your task.

## Overview

In discrete diffusion models, the noise schedule `α(t)` controls how the signal decays over time:
- At `t=0`: `α(0) ≈ 1` (clean data)
- At `t=1`: `α(1) ≈ 0` (fully noised/masked)

The masking probability at time `t` is `1 - α(t)`.

## Available Schedules

### 1. Log-Linear (Current Default)
**Config:** `noise=log-linear`

```yaml
type: log-linear
eps: 1e-3
```

**Formula:** `α(t) = 1 - t`

**Properties:**
- ✓ Simple and interpretable
- ✓ Linear masking rate
- ⚠️ Constant derivative (may not be optimal)

**When to use:** Baseline experiments, simple tasks

---

### 2. Cosine (Recommended) ⭐
**Config:** `noise=cosine`

```yaml
type: cosine
eps: 1e-3
```

**Formula:** `α(t) = cos²(πt/2)`

**Properties:**
- ✓ Smooth transitions
- ✓ Less aggressive at extremes (t→0, t→1)
- ✓ Better gradient flow
- ✓ Proven to work well in practice

**When to use:**
- General-purpose, recommended starting point
- Tasks requiring smooth diffusion process
- When log-linear gives unstable training

**Reference:** *Improved Denoising Diffusion Probabilistic Models* (Nichol & Dhariwal, 2021)

---

### 3. Squared Cosine (DDPM Variant)
**Config:** `noise=squared-cosine`

```yaml
type: squared-cosine
eps: 1e-3
s: 0.008  # offset parameter
```

**Formula:** `α(t) = cos²(π(t+s)/(2(1+s)))`

**Properties:**
- ✓ Prevents α from reaching 0 too quickly
- ✓ More stable than basic cosine
- ✓ Used in original DDPM

**When to use:**
- When basic cosine has issues near t=1
- Long sequences where you need more steps
- Following DDPM-style training

**Reference:** *Improved Denoising Diffusion Probabilistic Models* (Nichol & Dhariwal, 2021)

---

### 4. Linear (DDPM Style)
**Config:** `noise=linear`

```yaml
type: linear
eps: 1e-3
beta_min: 0.1
beta_max: 20.0
```

**Formula:** Linear in variance space
- `β(t) = β_min + t(β_max - β_min)`
- `α(t) = exp(-0.5 ∫β(τ)dτ)`

**Properties:**
- ✓ Original DDPM schedule
- ✓ Theoretically motivated (variance schedule)
- ⚠️ Can be aggressive with default parameters

**When to use:**
- Replicating DDPM experiments
- When you want to control variance explicitly

**Tuning:**
- Decrease `beta_max` for gentler diffusion
- Increase `beta_min` for stronger early noise

**Reference:** *Denoising Diffusion Probabilistic Models* (Ho et al., 2020)

---

### 5. Polynomial
**Config:** `noise=polynomial`

```yaml
type: polynomial
eps: 1e-3
power: 2.0
```

**Formula:** `α(t) = 1 - t^p`

**Properties:**
- ✓ Flexible shape control via power parameter
- `p < 1`: More noise early, slower at end
- `p = 1`: Same as log-linear
- `p > 1`: Less noise early, faster at end

**When to use:**
- When you want to emphasize early vs late diffusion
- Hierarchical tasks (use p>1 for coarse-to-fine)

**Tuning:**
- `power=0.5`: Good for coarse-to-fine generation
- `power=2.0`: Good for fine-to-coarse generation
- `power=3.0`: Very aggressive late diffusion

---

### 6. Sigmoid
**Config:** `noise=sigmoid`

```yaml
type: sigmoid
eps: 1e-3
scale: 6.0
shift: 3.0
```

**Formula:** `α(t) = sigmoid((1-t) × scale - shift)`

**Properties:**
- ✓ Concentrates diffusion in a specific time region
- ✓ Smooth transitions
- ✓ Flexible via scale/shift parameters

**When to use:**
- When you want most diffusion to happen in a specific time window
- Experimenting with non-standard schedules

**Tuning:**
- `scale`: Controls steepness (higher = sharper transition)
- `shift`: Controls center point (higher = later transition)

---

## Choosing a Schedule

### Quick Start Recommendations

| Task Type | Recommended Schedule | Alternative |
|-----------|---------------------|-------------|
| **General tasks** | `cosine` | `squared-cosine` |
| **Boolean formulas (BFVP)** | `cosine` | `log-linear` |
| **Arithmetic** | `cosine` | `polynomial (p=2)` |
| **FSA/Parity** | `cosine` | `squared-cosine` |
| **Hierarchical tasks** | `polynomial (p>1)` | `sigmoid` |
| **Long sequences** | `squared-cosine` | `linear` |

### Decision Tree

```
Start with: cosine (eps=1e-3)
│
├─ If training is unstable
│  └─ Try: squared-cosine or linear with lower beta_max
│
├─ If accuracy plateaus
│  └─ Try: polynomial (vary power)
│
├─ If need more early diffusion
│  └─ Try: polynomial (p < 1)
│
└─ If need more late diffusion
   └─ Try: polynomial (p > 1)
```

## Using Different Schedules

### Via Command Line
```bash
# Use cosine schedule
python main.py data=bfvp algo=mdlm noise=cosine

# Use polynomial with custom power
python main.py data=parity algo=mdlm noise=polynomial noise.power=2.5

# Use linear with custom beta range
python main.py data=arithmetic algo=mdlm noise=linear noise.beta_min=0.05 noise.beta_max=15.0
```

### Via Config Override
Edit your config file:
```yaml
defaults:
  - /noise: cosine  # or linear, polynomial, etc.

# Optional: override parameters
noise:
  eps: 1e-3
  # schedule-specific params
```

### Visualizing Schedules
```bash
python scripts/visualize_noise_schedules.py
```

This will generate comparison plots showing α(t), noise levels, and derivatives for all schedules.

## Monitoring Schedule Behavior

The following metrics are logged to WandB:

- `diffusion/alpha_t_mean`: Average signal level (should vary with timestep)
- `diffusion/mask_ratio`: Actual masking ratio (should ≈ 1 - α_t)
- `diffusion/t_mean`, `diffusion/t_std`: Timestep sampling distribution

**What to look for:**
- `mask_ratio ≈ 1 - alpha_t_mean`: Schedule is working correctly
- `alpha_t_std > 0.2`: Good timestep diversity (antithetic sampling working)
- Stable `grad_norm`: Schedule isn't causing training instability

## Implementation Details

All schedules implement the same interface:

```python
class NoiseSchedule(torch.nn.Module):
    def forward(self, t):
        """
        Args:
            t: timestep in [eps, 1], shape (batch,)

        Returns:
            dalpha_t: dα/dt at time t, shape (batch,)
            alpha_t: signal level at time t, shape (batch,)
        """
        ...
```

The derivative `dalpha_t` is used in the ELBO loss computation for MDLM and other models.

## References

1. **DDPM**: Ho et al. "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
2. **Improved DDPM**: Nichol & Dhariwal "Improved Denoising Diffusion Probabilistic Models" (ICML 2021)
3. **MDLM**: Sahoo et al. "Simple and Effective Masked Diffusion Language Models" (2024)
4. **SEDD**: Lou et al. "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" (ICML 2024)

## Troubleshooting

### Training Loss is NaN
- Try `squared-cosine` with larger `s` parameter (e.g., 0.01)
- Reduce `beta_max` if using `linear` schedule
- Check `diffusion/alpha_t_min` - should stay > 1e-5

### Poor Generation Quality
- Try `cosine` if using `log-linear`
- Increase number of sampling steps
- Check if `mask_ratio` is too aggressive (> 0.8) early in training

### Slow Convergence
- Try `polynomial` with `power < 1` for more early diffusion
- Adjust learning rate scheduler
- Check if timestep sampling is uniform (`diffusion/t_std` ≈ 0.29)

## Contributing New Schedules

To add a new schedule:

1. Implement the schedule class in `trainer_base.py`:
```python
class YourSchedule(torch.nn.Module):
    def __init__(self, param1=default, eps=1e-3):
        super().__init__()
        self.param1 = param1
        self.eps = eps

    def forward(self, t):
        t = (1 - self.eps) * t + self.eps
        alpha_t = ...  # your formula
        dalpha_t = ...  # derivative
        return dalpha_t, alpha_t
```

2. Add config file in `configs/noise/your_schedule.yaml`

3. Register in `TrainerBase.__init__()` noise schedule selection

4. Test with visualization script

5. Document in this README
