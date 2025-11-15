# MDLM-Specific Stability Issues

## Critical Issues Identified

### 1. **Division by Near-Zero in Loss Computation** ⚠️⚠️⚠️

**Location**: [algo.py:291](algo.py:291)

```python
def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
    # ...
    denominator = (1 - alpha_t).clamp(min=1e-7)
    return log_p_theta * dalpha_t / denominator
```

**Problem**:
- When `alpha_t` approaches 1 (high signal, low noise), denominator `(1 - alpha_t)` approaches 0
- Even with clamping at `1e-7`, the division can produce values up to `1e7` or higher
- If `log_p_theta` or `dalpha_t` are large, the loss can explode to infinity
- This is especially problematic at the **start of training** when alpha_t is sampled uniformly and can be very close to 1

**Why It Causes Explosions**:
```python
# Example scenario:
alpha_t = 0.9999  # Very close to 1 (clean data)
denominator = (1 - 0.9999) = 0.0001
log_p_theta = -5.0  # Reasonable log probability
dalpha_t = -0.999  # From log-linear schedule

loss = (-5.0) * (-0.999) / 0.0001 = 4.995 / 0.0001 = 49,950!
# Single token can contribute ~50K to the loss!
```

**Fix Needed**:
```python
# Option 1: More aggressive clamping
denominator = (1 - alpha_t).clamp(min=1e-3)  # 1000x safer

# Option 2: Clamp the entire loss
denominator = (1 - alpha_t).clamp(min=1e-7)
loss = (log_p_theta * dalpha_t / denominator).clamp(max=10.0)

# Option 3: Better time sampling (avoid extreme alpha_t values)
# Don't sample alpha_t > 0.999 during training
```

---

### 2. **Numerical Instability in `log(exp(x) - 1)`** ⚠️⚠️

**Location**: [algo.py:315](algo.py:315)

```python
log_k = -torch.log(torch.expm1(sigma)).squeeze(-1)
```

**Problem**:
- `torch.expm1(sigma)` computes `exp(sigma) - 1`
- For small `sigma` (< 1e-3), `exp(sigma) - 1 ≈ sigma`, and `torch.expm1` helps
- But `log(expm1(sigma))` can still be numerically unstable
- When `sigma` is very small: `log(expm1(sigma)) → log(sigma) → -∞`
- When `sigma` is very large: `log(expm1(sigma)) → sigma`, but computation can overflow

**Example**:
```python
sigma = 1e-5
expm1_sigma = torch.expm1(torch.tensor(1e-5))  # ≈ 1e-5
log_k = -torch.log(expm1_sigma)  # ≈ -log(1e-5) = 11.5

sigma = 10.0
expm1_sigma = torch.expm1(torch.tensor(10.0))  # ≈ 22025
log_k = -torch.log(expm1_sigma)  # ≈ -10.0
```

**Fix Needed**:
```python
# Better numerically stable version
def safe_log_expm1(sigma):
    # For small sigma: log(exp(sigma) - 1) ≈ log(sigma)
    # For large sigma: log(exp(sigma) - 1) ≈ sigma
    return torch.where(
        sigma < 0.5,
        torch.log(torch.expm1(sigma)),  # Accurate for small values
        sigma - torch.log1p(torch.exp(-sigma))  # Stable for large values
    )

log_k = -safe_log_expm1(sigma).squeeze(-1)
```

---

### 3. **No Gradient Clipping on Per-Token Losses** ⚠️

**Location**: [algo.py:282-292](algo.py:282-292)

**Problem**:
- MDLM computes per-token NLL which can be arbitrarily large
- Gradient clipping in the config only clips the **total gradient norm**, not individual losses
- A single token with loss = 50,000 will contribute massive gradients
- Even if total gradient is clipped, the damage is done in the backward pass

**Current Flow**:
```python
# Per-token loss (can be huge!)
loss_per_token = log_p_theta * dalpha_t / (1 - alpha_t)

# Summed over batch
total_loss = loss_per_token.sum()

# Backward (individual tokens contribute huge gradients)
total_loss.backward()

# Gradient clipping happens AFTER backward
# trainer.gradient_clip_val = 10.0  # Too late!
```

**Fix Needed**:
```python
def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
    log_p_theta = torch.gather(log_x_theta, dim=-1, index=x0[:, :, None]).squeeze(-1)
    denominator = (1 - alpha_t).clamp(min=1e-3)  # More conservative

    # Clamp per-token loss to prevent explosions
    loss_per_token = log_p_theta * dalpha_t / denominator
    loss_per_token = loss_per_token.clamp(min=-20.0, max=20.0)  # NEW!

    return loss_per_token
```

---

### 4. **Extreme Timestep Sampling** ⚠️

**Location**: [diffusion.py:140](diffusion.py:140)

```python
t = mask_ratio.clamp(min=1.0 / self.T if self.T > 0 else 1e-6)
```

**Problem**:
- Timesteps `t` are derived from mask ratios
- When `mask_ratio ≈ 0` (almost no masks), `t` is clamped to a minimum
- For continuous time (`T=0`), `t_min = 1e-6`, which means `alpha_t ≈ 1 - 1e-6 = 0.999999`
- This puts denominator `(1 - alpha_t) = 1e-6`, leading to million-fold amplification of loss!

**Fix Needed**:
```python
# More reasonable minimum timestep
eps = 1e-3  # Instead of 1e-6
t = mask_ratio.clamp(min=eps, max=1.0 - eps)

# Or avoid extreme timesteps entirely
min_t = 0.01  # Don't sample too close to t=0 or t=1
max_t = 0.99
t = mask_ratio.clamp(min=min_t, max=max_t)
```

---

### 5. **Softmax + Log Normalization Instability** ⚠️

**Location**: [algo.py:336-340](algo.py:336-340)

```python
def _process_model_output(self, model_output, xt, sigma):
    model_output[:, :, self.mask_index] += self.neg_infinity
    # Normalize to log-probabilities
    model_output = model_output - torch.logsumexp(model_output, dim=-1, keepdim=True)
    # ...
```

**Problem**:
- When model outputs are extreme (e.g., all logits are -1000), logsumexp can underflow
- `self.neg_infinity = -1e10` might not be negative enough, causing numerical issues
- If model outputs unstable logits, normalization can fail

**Fix Needed**:
```python
def _process_model_output(self, model_output, xt, sigma):
    # More aggressive masking
    model_output[:, :, self.mask_index] = -1e10

    # Clamp logits before normalization to prevent extreme values
    model_output = model_output.clamp(min=-50.0, max=50.0)

    # Normalize to log-probabilities
    model_output = model_output.log_softmax(dim=-1)  # Use built-in for stability

    # ... rest of the function
```

---

### 6. **No Loss Monitoring for Individual Tokens** ⚠️

**Problem**:
- Current auto-recovery only monitors **batch-level loss**
- MDLM loss is summed over all tokens
- A single bad token can spike to 50K, but if there are 100 tokens, average loss is "only" 500
- Loss threshold of 5.0 won't catch this!

**Fix Needed**:
Add per-token loss monitoring to auto-recovery:

```python
# In auto_recovery_callback.py
def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    # Existing loss check
    loss = outputs['loss']

    # NEW: Check for per-token loss spikes
    if 'loss_per_token' in outputs:
        loss_per_token = outputs['loss_per_token']
        max_token_loss = loss_per_token.abs().max().item()

        if max_token_loss > 100.0:  # Token-level threshold
            is_unstable = True
            reason = f"Per-token loss spike (max={max_token_loss:.2f})"
```

---

### 7. **Curriculum Learning + Extreme Timesteps** ⚠️

**Problem**:
- Curriculum learning starts with short sequences
- Short sequences have fewer tokens to mask
- Fewer masks → lower mask_ratio → more extreme timesteps
- This creates a **perfect storm** at the start of training:
  1. Short sequences (curriculum bin 1)
  2. Extreme timesteps (mask_ratio ≈ 0)
  3. Division by near-zero in loss
  4. **BOOM!** 💥

**Example**:
```python
# Curriculum bin 1: length = 16-32 tokens
# Prompt: "x1 x2 x3 #"  (4 tokens)
# Completion: " y"  (1 token)
# Total masks: 1 out of 32 → mask_ratio = 1/32 = 0.03

# Extreme timestep
t = 0.03  # Very close to 0
alpha_t = 1 - 0.03 = 0.97
denominator = (1 - 0.97) = 0.03  # Not clamped enough!

# Loss explosion!
```

**Fix Needed**:
```python
# In diffusion.py, adjust mask_ratio calculation
mask_ratio = (mask_counts / num_maskable_tokens).clamp(0.1, 0.9)  # NEW: Don't allow extreme ratios

# Or adjust minimum timestep based on sequence length
min_t = max(0.01, 1.0 / seq_len)  # At least 1/length
t = mask_ratio.clamp(min=min_t, max=1.0 - min_t)
```

---

## Recommended Fixes (Priority Order)

### **🔴 CRITICAL (Implement Immediately)**

1. **Clamp per-token losses** in `MDLM.nll_per_token()`:
   ```python
   return (log_p_theta * dalpha_t / denominator).clamp(min=-20.0, max=20.0)
   ```

2. **Increase denominator clamping** from `1e-7` to `1e-3`:
   ```python
   denominator = (1 - alpha_t).clamp(min=1e-3)
   ```

3. **Avoid extreme timesteps** in mask_ratio:
   ```python
   mask_ratio = (mask_counts / num_maskable_tokens).clamp(0.05, 0.95)
   ```

### **🟡 HIGH PRIORITY (Next)**

4. **Fix `log(expm1())` numerical stability**:
   ```python
   def safe_log_expm1(sigma):
       return torch.where(
           sigma < 0.5,
           torch.log(torch.expm1(sigma)),
           sigma - torch.log1p(torch.exp(-sigma))
       )
   ```

5. **Clamp model outputs before normalization**:
   ```python
   model_output = model_output.clamp(min=-50.0, max=50.0)
   ```

### **🟢 MEDIUM PRIORITY (After Testing)**

6. **Add per-token loss monitoring** to auto-recovery

7. **Adjust minimum timestep** based on curriculum:
   ```python
   # Longer min_t for early curriculum bins
   min_t = max(0.01, 0.1 / current_bin)
   ```

---

## Testing the Fixes

### Test 1: Check Current Loss Distribution

```python
# Add to training_step in trainer_base.py
def training_step(self, batch, batch_idx):
    losses = self._loss(...)

    # Log loss statistics
    if batch_idx % 100 == 0:
        print(f"Loss stats:")
        print(f"  Mean: {losses.mean():.4f}")
        print(f"  Max: {losses.max():.4f}")
        print(f"  Min: {losses.min():.4f}")
        print(f"  Std: {losses.std():.4f}")

    return {"loss": losses.mean()}
```

### Test 2: Monitor Denominator Values

```python
# In MDLM.nll_per_token
denominator = (1 - alpha_t).clamp(min=1e-7)

# Add logging
if denominator.min() < 1e-5:
    print(f"WARNING: Small denominator detected!")
    print(f"  Min: {denominator.min():.2e}")
    print(f"  alpha_t range: [{alpha_t.min():.6f}, {alpha_t.max():.6f}]")
```

### Test 3: Verify Timestep Distribution

```python
# Add to diffusion.py after computing t
import matplotlib.pyplot as plt

all_t_values = []
# ... collect during training ...

plt.hist(all_t_values, bins=50)
plt.xlabel("Timestep t")
plt.ylabel("Frequency")
plt.title("Distribution of Sampled Timesteps")
plt.savefig("timestep_distribution.png")
```

---

## Why These Issues Are MDLM-Specific

1. **AR models** don't have timesteps or diffusion dynamics → no division by `(1 - alpha_t)`
2. **LT models** don't use continuous time → no extreme alpha_t sampling
3. **D3PM/SEDD** use different parameterizations that avoid the `1/(1-alpha_t)` term

MDLM is uniquely vulnerable because it combines:
- Continuous-time diffusion (`alpha_t` can be arbitrarily close to 1)
- Division by `(1 - alpha_t)` in the loss
- Curriculum learning (which creates extreme mask ratios)
- Log-space computations (`log(expm1())`)

---

## Expected Impact

After implementing these fixes:
- ✅ Loss should stay in reasonable range (< 20 per token)
- ✅ No more sudden spikes to 100+
- ✅ Training should be much more stable, especially in early curriculum bins
- ✅ Recovery callback should rarely need to trigger
- ✅ Can potentially use higher learning rates safely

---

## Implementation Script

Create `scripts/apply_mdlm_fixes.py`:

```python
#!/usr/bin/env python3
"""Apply critical stability fixes to MDLM."""

import sys

def apply_fixes():
    # Fix 1: Update algo.py
    with open('algo.py', 'r') as f:
        content = f.read()

    # Replace denominator clamping
    content = content.replace(
        'denominator = (1 - alpha_t).clamp(min=1e-7)',
        'denominator = (1 - alpha_t).clamp(min=1e-3)'
    )

    # Add loss clamping
    content = content.replace(
        'return log_p_theta * dalpha_t / denominator',
        'return (log_p_theta * dalpha_t / denominator).clamp(min=-20.0, max=20.0)'
    )

    with open('algo.py', 'w') as f:
        f.write(content)

    print("✓ Applied fixes to algo.py")

    # Fix 2: Update diffusion.py
    with open('diffusion.py', 'r') as f:
        content = f.read()

    # Adjust mask_ratio clamping
    content = content.replace(
        'mask_ratio = (mask_counts / num_maskable_tokens).clamp(0.0, 1.0)',
        'mask_ratio = (mask_counts / num_maskable_tokens).clamp(0.05, 0.95)'
    )

    with open('diffusion.py', 'w') as f:
        f.write(content)

    print("✓ Applied fixes to diffusion.py")
    print("\nAll critical fixes applied!")
    print("Please run tests and restart training.")

if __name__ == "__main__":
    apply_fixes()
```

---

## Summary

**Root Cause**: MDLM's loss formula `loss = log_p * dalpha / (1 - alpha)` creates **extreme amplification** when `alpha ≈ 1`, which happens frequently due to:
1. Continuous-time sampling
2. Short sequences in curriculum learning
3. Extreme mask ratios

**Solution**: Multi-layered protection:
- Clamp denominators
- Clamp per-token losses
- Avoid extreme timesteps
- Fix numerical instabilities

These fixes address the **structural instability** in MDLM, making the auto-recovery + LR reduction + data reshuffling much more effective.
