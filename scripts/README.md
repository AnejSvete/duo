# Scripts Directory

## Main Launch Script

### `launch_smart.sh` ⭐
**One-command experiment launcher with automatic data preparation**

```bash
./scripts/launch_smart.sh EXPERIMENT_NAME LANGUAGE1 [LANGUAGE2 ...]
```

**What it does:**
1. Prepares all data formats in parallel (3 formats per language)
2. Launches all 8 training jobs with dependencies
3. Creates analysis and monitoring scripts

**Example:**
```bash
./scripts/launch_smart.sh my_experiment bfvp parity arithmetic
```

This submits:
- 9 data prep jobs (3 languages × 3 formats)
- 24 training jobs (3 languages × 8 algorithms)

---

## Low-Level Scripts

### `prepare_data.sh`
Prepare data for a specific language and format (used by `launch_smart.sh`)

```bash
sbatch --export=ALL,LANGUAGE=bfvp,FORMAT=trace,MODEL_LENGTH=256 scripts/prepare_data.sh
```

### `run_single.sh`
Run a single training job (used by `launch_smart.sh`)

```bash
sbatch --export=ALL,TASK=bfvp,ALGO=cot,MODEL_LENGTH=256,OUTPUT_DIR=path scripts/run_single.sh
```

---

## Individual Training Scripts

These are called by the main launchers:

- `train_cot.sh` - Chain-of-thought (AR with trace)
- `train_mdm.sh` - Masked diffusion (MDLM)
- `train_classifier.sh` - Single-pass classifier (constant depth LT)
- `train_looping.sh` - Looping transformer (log depth)
- `train_padding.sh` - Padding baseline
- `train_padded_looping.sh` - Padded looping transformer
- `train_empty_padding.sh` - Empty padding baseline
- `train_empty_padded_looping.sh` - Empty padded looping

**Usage:**
```bash
sbatch scripts/train_cot.sh LANGUAGE MODEL_LENGTH
```

---

## Recommended Workflow

**Just use the main launcher:**
```bash
./scripts/launch_smart.sh my_experiment bfvp
```

Everything else is handled automatically!

See [../QUICK_START.md](../QUICK_START.md) for complete usage guide.
