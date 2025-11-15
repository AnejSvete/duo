#!/usr/bin/env python3
"""
Grid search over MDM hyperparameters.

This script generates SLURM jobs for a comprehensive hyperparameter search over:
- Learning rates
- Weight decay
- Gradient clipping values
- Noise schedules
- Loss types
- Warmup steps
- Batch sizes

Usage:
    python scripts/grid_search_mdm.py --task bfvp --output_dir grid_search_results
    python scripts/grid_search_mdm.py --task parity --output_dir grid_search_results --dry_run
"""

import argparse
import itertools
import os
import subprocess
from pathlib import Path
from datetime import datetime


# Grid search parameter space
HYPERPARAMETER_GRID = {
    # Learning rate is the most critical hyperparameter for diffusion models
    "lr": [1e-3, 3e-3, 5e-3, 1e-2],

    # Weight decay for regularization
    "weight_decay": [0.0, 0.05, 0.1, 0.2],

    # Gradient clipping to prevent explosions
    "gradient_clip_val": [1.0, 5.0, 10.0, 20.0],

    # Noise schedules - different schedules can significantly affect training
    "noise_schedule": ["log-linear", "linear", "cosine"],

    # Warmup steps - important for stable training
    "warmup_steps": [100, 250, 500, 1000],

    # Batch size - affects gradient variance
    "batch_size": [1024, 2048, 4096],
}

# Reduced grid for quick testing
QUICK_GRID = {
    "lr": [1e-3, 5e-3, 1e-2],
    "weight_decay": [0.0, 0.1],
    "gradient_clip_val": [5.0, 10.0],
    "noise_schedule": ["log-linear", "linear"],
    "warmup_steps": [250, 500],
    "batch_size": [2048],
}

# Best practices based on diffusion literature
RECOMMENDED_GRID = {
    "lr": [3e-3, 5e-3, 8e-3],  # Around 5e-3 often works well
    "weight_decay": [0.05, 0.1, 0.15],  # Moderate regularization
    "gradient_clip_val": [5.0, 10.0],  # Prevent explosions but not too aggressive
    "noise_schedule": ["log-linear", "cosine"],  # These tend to work best
    "warmup_steps": [250, 500],  # Enough warmup but not too long
    "batch_size": [2048],  # Reasonable batch size for stability
}


def generate_slurm_script(job_name, command, output_dir, time="04:00:00", mem="64G", gpus=1):
    """Generate SLURM job script."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --gpus={gpus}
#SBATCH --cpus-per-task=8

# Activate environment
source ~/.bashrc
conda activate duo

# Run command
{command}
"""
    return script


def create_job_name(params, task, run_id):
    """Create a descriptive job name from parameters."""
    # Shorten noise schedule name
    noise_short = params["noise_schedule"].replace("log-", "l").replace("linear", "lin").replace("cosine", "cos")

    name_parts = [
        f"mdm_{task}",
        f"lr{params['lr']:.0e}".replace("e-0", "e-"),
        f"wd{params['weight_decay']:.2f}",
        f"gc{params['gradient_clip_val']:.0f}",
        f"ns{noise_short}",
        f"ws{params['warmup_steps']}",
        f"bs{params['batch_size']}",
        f"r{run_id}",
    ]
    return "_".join(name_parts)


def build_command(params, task, output_base_dir):
    """Build the training command with hyperparameters."""
    # Create unique output directory for this configuration
    job_name = create_job_name(params, task, 0)  # 0 for directory name
    output_dir = Path(output_base_dir) / job_name

    # Build command
    cmd_parts = [
        "python main.py",
        f"data={task}",
        "algo=mdlm",
        "model=nano",

        # Hyperparameters
        f"optim.lr={params['lr']}",
        f"optim.weight_decay={params['weight_decay']}",
        f"trainer.gradient_clip_val={params['gradient_clip_val']}",
        f"noise={params['noise_schedule']}",
        f"lr_scheduler.lr_lambda.warmup_steps={params['warmup_steps']}",
        f"loader.batch_size={params['batch_size']}",
        f"loader.global_batch_size={params['batch_size']}",

        # Output configuration
        f"hydra.run.dir={output_dir}",

        # W&B configuration
        f"wandb.name={job_name}",
        f"wandb.tags=[grid_search,mdm,{task}]",
        f"wandb.group=grid_search_{task}_{datetime.now().strftime('%Y%m%d')}",

        # Training configuration
        "trainer.max_steps=10000",
        "curriculum.enabled=true",
        "checkpointing.resume_from_ckpt=true",
    ]

    return " ".join(cmd_parts)


def main():
    parser = argparse.ArgumentParser(description="Grid search for MDM hyperparameters")
    parser.add_argument("--task", type=str, default="bfvp",
                       choices=["bfvp", "parity", "arithmetic", "contains_a", "ab_star", "mod_3"],
                       help="Task to run grid search on")
    parser.add_argument("--output_dir", type=str, default="grid_search_results",
                       help="Base directory for outputs")
    parser.add_argument("--grid", type=str, default="recommended",
                       choices=["full", "quick", "recommended"],
                       help="Which grid to use")
    parser.add_argument("--num_seeds", type=int, default=3,
                       help="Number of random seeds per configuration")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print commands without submitting jobs")
    parser.add_argument("--sequential", action="store_true",
                       help="Run jobs sequentially instead of submitting to SLURM")
    parser.add_argument("--create_scripts_only", action="store_true",
                       help="Only create job scripts without submitting")
    args = parser.parse_args()

    # Select grid
    if args.grid == "full":
        grid = HYPERPARAMETER_GRID
    elif args.grid == "quick":
        grid = QUICK_GRID
    else:
        grid = RECOMMENDED_GRID

    # Generate all parameter combinations
    param_names = sorted(grid.keys())
    param_values = [grid[name] for name in param_names]

    all_configs = []
    for values in itertools.product(*param_values):
        config = dict(zip(param_names, values))
        all_configs.append(config)

    print(f"Grid search configuration:")
    print(f"  Task: {args.task}")
    print(f"  Grid size: {len(all_configs)} configurations")
    print(f"  Seeds per config: {args.num_seeds}")
    print(f"  Total jobs: {len(all_configs) * args.num_seeds}")
    print(f"  Output directory: {args.output_dir}")
    print()

    # Print parameter ranges
    print("Parameter grid:")
    for name, values in grid.items():
        print(f"  {name}: {values}")
    print()

    if args.dry_run:
        print("DRY RUN - showing first 5 configurations:")
        for i, config in enumerate(all_configs[:5]):
            print(f"\nConfiguration {i+1}:")
            for k, v in config.items():
                print(f"  {k}: {v}")
            cmd = build_command(config, args.task, args.output_dir)
            print(f"  Command: {cmd}")
        print(f"\n... and {len(all_configs) - 5} more configurations")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Directory for SLURM scripts
    scripts_dir = output_dir / "slurm_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Submit jobs
    submitted_jobs = []
    job_scripts = []

    for config_id, config in enumerate(all_configs):
        for seed in range(args.num_seeds):
            # Create job name
            job_name = create_job_name(config, args.task, seed)

            # Build command with seed
            cmd = build_command(config, args.task, args.output_dir)
            cmd += f" seed={seed}"

            # Generate SLURM script
            slurm_script = generate_slurm_script(
                job_name=job_name,
                command=cmd,
                output_dir=args.output_dir,
                time="04:00:00",
                mem="64G",
                gpus=1,
            )

            # Save script
            script_path = scripts_dir / f"{job_name}.sh"
            with open(script_path, "w") as f:
                f.write(slurm_script)
            script_path.chmod(0o755)
            job_scripts.append(script_path)

            if not args.create_scripts_only:
                if args.sequential:
                    # Run sequentially
                    print(f"Running job {len(submitted_jobs)+1}/{len(all_configs)*args.num_seeds}: {job_name}")
                    subprocess.run(["bash", str(script_path)], check=True)
                    submitted_jobs.append(job_name)
                else:
                    # Submit to SLURM
                    result = subprocess.run(
                        ["sbatch", str(script_path)],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        job_id = result.stdout.strip().split()[-1]
                        submitted_jobs.append((job_name, job_id))
                        print(f"Submitted job {len(submitted_jobs)}/{len(all_configs)*args.num_seeds}: {job_name} (ID: {job_id})")
                    else:
                        print(f"Failed to submit job {job_name}: {result.stderr}")

    # Save summary
    summary_file = output_dir / "grid_search_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Grid Search Summary\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Grid: {args.grid}\n")
        f.write(f"Number of configurations: {len(all_configs)}\n")
        f.write(f"Seeds per configuration: {args.num_seeds}\n")
        f.write(f"Total jobs: {len(all_configs) * args.num_seeds}\n")
        f.write(f"Submitted jobs: {len(submitted_jobs)}\n")
        f.write(f"Output directory: {args.output_dir}\n")
        f.write(f"\n{'='*80}\n\n")
        f.write("Parameter Grid:\n")
        for name, values in grid.items():
            f.write(f"  {name}: {values}\n")
        f.write(f"\n{'='*80}\n\n")
        f.write("Job Scripts:\n")
        for script_path in job_scripts:
            f.write(f"  {script_path}\n")
        if not args.create_scripts_only and not args.sequential:
            f.write(f"\n{'='*80}\n\n")
            f.write("Submitted Jobs:\n")
            for job_name, job_id in submitted_jobs:
                f.write(f"  {job_name}: {job_id}\n")

    print(f"\n{'='*80}")
    if args.create_scripts_only:
        print(f"Created {len(job_scripts)} job scripts in {scripts_dir}")
        print(f"To submit jobs, run: for f in {scripts_dir}/*.sh; do sbatch $f; done")
    elif args.sequential:
        print(f"Completed {len(submitted_jobs)} jobs sequentially")
    else:
        print(f"Submitted {len(submitted_jobs)} jobs to SLURM")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
