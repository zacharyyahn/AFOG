#!/bin/bash
#SBATCH -JAFOG
#SBATCH -N1 --ntasks-per-node=1
#SBATCH -t18:00:00
#SBATCH -ooutput/baseline/ssd_afog_all.out

echo $CONDA_PREFIX
module load anaconda3
#conda init
conda deactivate
conda activate AFOG_CNN
echo $CONDA_PREFIX
srun python attack_ssd.py
