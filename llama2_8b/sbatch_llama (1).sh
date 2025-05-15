#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH -p gpu
#SBATCH --gpus=1                      # Request 5 GPUs (corrected)
#SBATCH -t 40:59:59
#SBATCH -o SLURM-%J.OUT
#SBATCH --nodes=1
#SBATCH  --constraint=a100

module load cuda/11.8

source ../../test_LLM/bin/activate



# Launch with torchrun for multi-GPU support

#torchrun --nproc_per_node=5 train_bot.py
#python train_bot.py
#python tb_partial.py
python runTrained.py
#python chatbot.py

