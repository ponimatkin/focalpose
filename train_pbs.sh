#!/bin/bash
#PBS -N focalpose
#PBS -q gpu
#PBS -l select=5
#PBS -l walltime=8:00:00
#PBS -lplace=scatter:exclhost
#PBS -j oe
#PBS -k o

cd $PBS_O_WORKDIR

module load CUDA
module load MPICH

mpirun -np 40 $PBS_O_WORKDIR/run_train_pbs.sh ${arg1} ${arg2}