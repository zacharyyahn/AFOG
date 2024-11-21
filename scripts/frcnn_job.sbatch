#!/bin/bash
#SBATCH -JAFOG
#SBATCH -N1 --gres=gpu:A100:1 --ntasks-per-node=1 --mem-per-gpu=8G
#SBATCH -t8:00:00
#SBATCH -ooutput/none/frcnn_afog_all.out

module load anaconda3
conda deactivate
conda activate AFOG

srun python attack_frcnn.py
