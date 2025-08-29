#!/bin/bash
# Usage: bash launch_job.sh <TASK> [--prepare|--rest]
TASK=$1
FLAG=$2

if [ "$FLAG" = "--prepare" ]; then
    bash scripts/train_classifier.sh $TASK
    bash scripts/train_empty_padding.sh $TASK
    bash scripts/train_cot.sh $TASK
elif [ "$FLAG" = "--rest" ]; then
    bash scripts/train_empty_padded_looping.sh $TASK
    bash scripts/train_looping.sh $TASK
    bash scripts/train_mdm.sh $TASK
    bash scripts/train_padded_looping.sh $TASK
    bash scripts/train_padding.sh $TASK
else
    bash scripts/train_classifier.sh $TASK
    bash scripts/train_cot.sh $TASK
    bash scripts/train_empty_padded_looping.sh $TASK
    bash scripts/train_empty_padding.sh $TASK
    bash scripts/train_looping.sh $TASK
    bash scripts/train_mdm.sh $TASK
    bash scripts/train_padded_looping.sh $TASK
    bash scripts/train_padding.sh $TASK
fi

