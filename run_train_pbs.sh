#!/bin/bash

conda activate focalpose

export PYTHONPATH=$(pwd)
export JOB_DIR=$$PBS_O_WORKDIR/local_data/run_$1

if [ -d "$JOB_DIR" ]; then
  rm $JOB_DIR/*
else
  mkdir $JOB_DIR
fi

python -m focalpose.scripts.run_pose_training --config $2