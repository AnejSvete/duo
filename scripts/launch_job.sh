#!/bin/bash

# Usage: bash launch_job.sh <LANGUAGE> [--prepare|--rest]

LANGUAGE=$1
FLAG=$2
# Set model lengths based on language and max train length
if [ "$LANGUAGE" = "arithmetic" ]; then
    SHORT_MODEL_LENGTH=256
    LONG_MODEL_LENGTH=512
elif [ "$LANGUAGE" = "bfvp" ]; then
    SHORT_MODEL_LENGTH=256
    LONG_MODEL_LENGTH=512
else
    SHORT_MODEL_LENGTH=96
    LONG_MODEL_LENGTH=192
fi

if [ "$FLAG" = "--prepare" ]; then
    sbatch scripts/train_classifier.sh $LANGUAGE $SHORT_MODEL_LENGTH
    sbatch scripts/train_empty_padding.sh $LANGUAGE $LONG_MODEL_LENGTH
    sbatch scripts/train_cot.sh $LANGUAGE $LONG_MODEL_LENGTH
elif [ "$FLAG" = "--rest" ]; then
    sbatch scripts/train_empty_padded_looping.sh $LANGUAGE $LONG_MODEL_LENGTH
    sbatch scripts/train_looping.sh $LANGUAGE $SHORT_MODEL_LENGTH
    sbatch scripts/train_mdm.sh $LANGUAGE $LONG_MODEL_LENGTH
    sbatch scripts/train_padded_looping.sh $LANGUAGE $LONG_MODEL_LENGTH
    sbatch scripts/train_padding.sh $LANGUAGE $LONG_MODEL_LENGTH
else
    sbatch scripts/train_classifier.sh $LANGUAGE $SHORT_MODEL_LENGTH
    sbatch scripts/train_cot.sh $LANGUAGE $LONG_MODEL_LENGTH
    sbatch scripts/train_empty_padded_looping.sh $LANGUAGE $LONG_MODEL_LENGTH
    sbatch scripts/train_empty_padding.sh $LANGUAGE $LONG_MODEL_LENGTH
    sbatch scripts/train_looping.sh $LANGUAGE $SHORT_MODEL_LENGTH
    sbatch scripts/train_mdm.sh $LANGUAGE $LONG_MODEL_LENGTH
    sbatch scripts/train_padded_looping.sh $LANGUAGE $LONG_MODEL_LENGTH
    sbatch scripts/train_padding.sh $LANGUAGE $LONG_MODEL_LENGTH
fi

