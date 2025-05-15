#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH -p gpu
#SBATCH -G 3
#SBATCH -t 01:00:00
#SBATCH -o SLURM-%J.OUT

module load cuda/11.8
source ../../test_LLM/bin/activate

python chatbot.py
