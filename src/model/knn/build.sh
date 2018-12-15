#!/bin/bash
#
#SBATCH --job-name=build_knn_model_rds
#SBATCH --output=model_knn_rds.log
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --time=0-12:30
#SBATCH --mem-per-cpu=1000
export PYTHONPATH=/home/012413011/expression-recognition/src:/opt/ohpc/pub/libs/gnu7/openmpi3/mpi4py/3.0.0/lib64/python3.4/site-packages:/home/012413011/lib/python3.6/site-packages:$PYTHONPATH
python build_model_knn.py icv_mefed
