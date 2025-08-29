#!/bin/bash

# Usage: bash launch_job.sh <TASK> [--prepare|--rest] [SHORT_LEN] [LONG_LEN]


TASK=$1
FLAG=$2
SHORT_LEN=${3:-48}
LONG_LEN=${4:-96}

if [ "$FLAG" = "--prepare" ]; then
    sbatch scripts/train_classifier.sh $TASK $SHORT_LEN
    sbatch scripts/train_empty_padding.sh $TASK $LONG_LEN
    sbatch scripts/train_cot.sh $TASK $LONG_LEN
elif [ "$FLAG" = "--rest" ]; then
    sbatch scripts/train_empty_padded_looping.sh $TASK $LONG_LEN
    sbatch scripts/train_looping.sh $TASK $SHORT_LEN
    sbatch scripts/train_mdm.sh $TASK $LONG_LEN
    sbatch scripts/train_padded_looping.sh $TASK $LONG_LEN
    sbatch scripts/train_padding.sh $TASK $LONG_LEN
else
    sbatch scripts/train_classifier.sh $TASK $SHORT_LEN
    sbatch scripts/train_cot.sh $TASK $LONG_LEN
    sbatch scripts/train_empty_padded_looping.sh $TASK $LONG_LEN
    sbatch scripts/train_empty_padding.sh $TASK $LONG_LEN
    sbatch scripts/train_looping.sh $TASK $SHORT_LEN
    sbatch scripts/train_mdm.sh $TASK $LONG_LEN
    sbatch scripts/train_padded_looping.sh $TASK $LONG_LEN
    sbatch scripts/train_padding.sh $TASK $LONG_LEN
fi

