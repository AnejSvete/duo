#!/bin/bash
#SBATCH --job-name=mdm_a5_lr3e-3_wd0.05_gc5_nsllin_ws250_bs2048_r0
#SBATCH --output=grid_search_a5_recommended_20251115_163219/logs/mdm_a5_lr3e-3_wd0.05_gc5_nsllin_ws250_bs2048_r0_%j.out
#SBATCH --error=grid_search_a5_recommended_20251115_163219/logs/mdm_a5_lr3e-3_wd0.05_gc5_nsllin_ws250_bs2048_r0_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

# Activate environment
source ~/.bashrc
conda activate duo

# Run command
python main.py data=a5 algo=mdlm model=nano optim.lr=0.003 optim.weight_decay=0.05 trainer.gradient_clip_val=5.0 noise=log-linear lr_scheduler.lr_lambda.warmup_steps=250 loader.batch_size=2048 loader.global_batch_size=2048 hydra.run.dir=grid_search_a5_recommended_20251115_163219/mdm_a5_lr3e-3_wd0.05_gc5_nsllin_ws250_bs2048_r0 wandb.name=mdm_a5_lr3e-3_wd0.05_gc5_nsllin_ws250_bs2048_r0 wandb.tags=[grid_search,mdm,a5] wandb.group=grid_search_a5_20251115 trainer.max_steps=10000 curriculum.enabled=true checkpointing.resume_from_ckpt=true seed=0
