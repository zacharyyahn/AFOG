#!/bin/bash
#SBATCH -JAFOG_DETR
#SBATCH -N1 --gres=gpu:A100:1 --ntasks-per-node=1 --mem-per-gpu=32G
#SBATCH -t16:00:00
#SBATCH -ooutput/baseline/detr_r50_afo_all.out

module load anaconda3
#conda init
conda deactivate
conda activate AFOG

# ---- PARAMETERS ----
# RESUME: the saved model checkpoint, typically: ".../model.pth"
# COCO_PATH: path to coco, typically: ".../coco"
# BACKBONE: "resnet50" or "resnet101"
# ATTACK: "afog" or "none"
# MODE: AFOG variant selection, either: "baseline", "vanishing", or "fabrication"
# LOAD_DIR: directory from which to load saved attacked images.
# LOAD_ATTACK: 1.0 if want to load from LOAD_DIR, else 0.0.
# --------------------

BATCH_SIZE=1
RESUME="model_files/detr-r50.pth"
COCO_PATH="datasets/coco"
BACKBONE="resnet50"
ATTACK="afog"
MODE="baseline"
LOAD_DIR="datasets/blackbox/internimage/"
LOAD_ATTACK="0.0"

srun python scripts/attack_detr.py --batch_size $BATCH_SIZE --no_aux_loss --eval --resume $RESUME --backbone $BACKBONE --coco_path $COCO_PATH --attack $ATTACK --attack_mode $MODE --load_dir $LOAD_DIR --load_attack $LOAD_ATTACK
