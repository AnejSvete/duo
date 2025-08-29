#!/bin/bash
# Usage: bash launch_job.sh <TASK> [--prepare|--rest]
TASK=$1
FLAG=$2

if [ "$FLAG" = "--prepare" ]; then
    sbatch scripts/train_classifier.sh $TASK
    sbatch scripts/train_empty_padding.sh $TASK
    sbatch scripts/train_cot.sh $TASK
elif [ "$FLAG" = "--rest" ]; then
    sbatch scripts/train_empty_padded_looping.sh $TASK
    sbatch scripts/train_looping.sh $TASK
    sbatch scripts/train_mdm.sh $TASK
    sbatch scripts/train_padded_looping.sh $TASK
    sbatch scripts/train_padding.sh $TASK
else
    sbatch scripts/train_classifier.sh $TASK
    sbatch scripts/train_cot.sh $TASK
    sbatch scripts/train_empty_padded_looping.sh $TASK
    sbatch scripts/train_empty_padding.sh $TASK
    sbatch scripts/train_looping.sh $TASK
    sbatch scripts/train_mdm.sh $TASK
    sbatch scripts/train_padded_looping.sh $TASK
    sbatch scripts/train_padding.sh $TASK
fi

