#!/bin/bash
#SBATCH -JAFOG_Deformable_DETR
#SBATCH -N1 --gres=gpu:A100:1 --ntasks-per-node=1 --mem-per-gpu=32G
#SBATCH -t16:00:00
#SBATCH -ooutput/baseline/deformable_detr_afog_all.out

module load anaconda3
#conda init
conda deactivate
conda activate AFOG

# ---- PARAMETERS ----
# RESUME: the saved model checkpoint, typically: ".../model.pth"
# COCO_PATH: path to coco, typically: ".../coco"
# ATTACK: "afog" or "none"
# MODE: AFOG variant selection, either: "baseline", "vanishing", or "fabrication"
# LOAD_DIR: directory from which to load saved attacked images.
# LOAD_ATTACK: 1.0 if want to load from LOAD_DIR, else 0.0.
# --------------------

BATCH_SIZE=1
RESUME="model_files/r50_deformable_detr-checkpoint.pth"
COCO_PATH="datasets/coco"
ATTACK="afog"
MODE="baseline"
LOAD_DIR="datasets/blackbox/internimage/"
LOAD_ATTACK=0.0

srun python scripts/attack_deformable_detr.py --batch_size $BATCH_SIZE --no_aux_loss --eval --resume $RESUME --coco_path $COCO_PATH --attack $ATTACK --attack_mode $MODE --load_attack $LOAD_ATTACK --load_dir $LOAD_DIR
