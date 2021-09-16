#!/bin/sh

#SBATCH --account=mics-lg
#SBATCH --partition=mics-lg
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=0

source ~/.bashrc
conda activate calm

python dataset_utils/generate_discriminative_dataset.py
