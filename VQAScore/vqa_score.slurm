#!/bin/bash
#SBATCH --job-name=vqa_score
#SBATCH --output=vqa_output.log
#SBATCH --error=vqa_error.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# Initialize Conda
source ~/.bashrc
conda activate t2v

# Move to the correct directory
cd /home/ljwilliams/Attributes-Effect-on-OOD-Performance/VQAScore/t2v_metrics

# Run your Python script
python VQAScore.py
