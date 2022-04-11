#!/bin/bash
#SBATCH --job-name=focalpose
#SBATCH --nodes=5
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --output=pose_%j.out
#SBATCH --error=pose_%j.err

module purge
module load Anaconda3/5.0.1 NCCL

. /opt/apps/software/Anaconda3/5.0.1/etc/profile.d/conda.sh

conda deactivate
conda activate focalpose

export PYTHONPATH=$(pwd)
export JOB_DIR=local_data/run_$1

if [ -d "$JOB_DIR" ]; then
  rm $JOB_DIR/*
else
  mkdir $JOB_DIR
fi

srun python -m focalpose.scripts.run_pose_training --config $2