#!/bin/bash

# The interpreter used to execute the script


#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=foldingzero_cuda_small
#SBATCH --mail-user=heyifei@umich.edu
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --account=tewaria1
#SBATCH --partition=gpu
#SBATCH --mem-per-gpu=8g
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --get-user-env
# The application(s) to execute along with its input arguments and options:
conda activate base
module load cuda
module load gcc
python -u run.py
