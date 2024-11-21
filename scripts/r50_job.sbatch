#!/bin/bash
#SBATCH -JAFOG_R50
#SBATCH -N1 --gres=gpu:A100:1 --ntasks-per-node=1 --mem-per-gpu=128G
#SBATCH -t16:00:00
#SBATCH -ooutput/baseline/dino_r50_afog_all.out

module load anaconda3
#conda init
conda deactivate
conda activate AFOG

# ---- PARAMETERS ----
# ATTACK: "afog" or "none"
# MODE: AFOG variant selection, either: "baseline", "vanishing", or "fabrication"
# CONFIG: the path to the Detrex config, typically: ".../dino_r50_4scale_12ep.py"
# CHECKPOINT: the path to the mode file, typically: ".../model.pth"
# SAMPLE: sample of dataset, usually 1.0
# SAVE_DIR: directory to save attacked images.
# SAVE_ATTACK: 1.0 if want to save to SAVE_DIR, else 0.0
# LOAD_DIR: directory from which to load saved attacked images.
# LOAD_ATTACK: 1.0 if want to load from LOAD_DIR, else 0.0.
# ---------------------

ATTACK="afog"
MODE="baseline"
CONFIG="models/detrex/projects/dino/configs/dino-resnet/dino_r50_4scale_12ep.py"
CHECKPOINT="model_files/dino_r50_4scale.pth"
SAMPLE=1.0
SAVE_ATTACK=0.0
SAVE_DIR="datasets/blackbox/dino_r50_vanishing/"
LOAD_ATTACK=0.0
LOAD_DIR="datasets/blackbox/internimage/"

python scripts/attack_detrex.py --config-file $CONFIG --eval-only  train.init_checkpoint=$CHECKPOINT attack=$ATTACK attack_mode=$MODE sample=$SAMPLE save_attack=$SAVE_ATTACK save_dir=$SAVE_DIR load_attack=$LOAD_ATTACK load_dir=$LOAD_DIR
