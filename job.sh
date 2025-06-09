#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

#SBATCH --time=23:50:00
#SBATCH --partition=amdrtx
#SBATCH --mail-type=ALL
#SBATCH --mail-user=damianc@student.ethz.ch


rsync -av --exclude='*.pyc' "$HOME/code/" "$SCRATCH/code/"
rsync -av --exclude='*.pyc' "$HOME/measurements/" "$SCRATCH/measurements/"

cd $SCRATCH
module load libffi
srun bash -c 'source $SCRATCH/.venv/bin/activate ; julia $SCRATCH/code/vits.jl'

rsync -av "$SCRATCH/measurements/" "$HOME/measurements/"