#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=0
#SBATCH --exclusive

#SBATCH --time=23:50:00
#SBATCH --partition=amdrtx
#SBATCH --mail-type=ALL
#SBATCH --mail-user=damianc@student.ethz.ch



rsync -av --exclude='cifar-10-batches-py' --exclude='*.gz' --exclude='*.ncu-rep' --exclude='*.so' --exclude='*.pyc' "$HOME/code/" "$SCRATCH/code/"

cd $SCRATCH
module load libffi
srun bash -c 'source $SCRATCH/.venv/bin/activate ; julia $SCRATCH/code/vits2.jl'

rsync -av "$SCRATCH/measurements/" "$HOME/measurements/"