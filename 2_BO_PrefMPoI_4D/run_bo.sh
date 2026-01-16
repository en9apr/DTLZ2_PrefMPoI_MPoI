#!/bin/bash

#PBS -N CFD_lamella

# Select 1 nodes (maximum of 64 cores)
#PBS -l select=1:ncpus=1

# Select wall time to 16 hours (as we have doubled number of iterations: 8000 to 15000)
#PBS -l walltime=03:59:59

# Use the arm nodes
#PBS -q arm

# Load Python
module load cray-python/3.8.5.1
source ~/vEnPy38/bin/activate

# Load modules for currently recommended OpenFOAM v2306 build
module unload PrgEnv-cray
module load PrgEnv-gnu

# Change to directory that script was submitted from
export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)
export OMP_NUM_THREADS=1
cd $PBS_O_WORKDIR

python -u BO_Sampling_and_Optimisation.py > log.python 2>&1
echo "Python running in background with nohup"

