#!/bin/bash
#SBATCH -JAFOG_Align_DETR
#SBATCH -N1 --gres=gpu:A100:1 --ntasks-per-node=1 --mem-per-gpu=128G
#SBATCH -t16:00:00
#SBATCH -ooutput/baseline/align_detr_afog_all.out

module load anaconda3
conda init
conda deactivate
conda activate AFOG

# ---- PARAMETERS ----
# ATTACK: "afog" or "none"
# MODE: AFOG variant selection, either: "baseline", "vanishing", or "fabrication"
# CONFIG: the path to the Detrex config, typically: ".../aligndetr_k=2_r50_4scale_24ep.py"
# CHECKPOINT: the path to the mode file, typically: ".../model.pth"
# SAMPLE: sample of dataset, usually 1.0
# SAVE_DIR: directory to save attacked images.
# SAVE_ATTACK: 1.0 if want to save to SAVE_DIR, else 0.0
# LOAD_DIR: directory from which to load saved attacked images.
# LOAD_ATTACK: 1.0 if want to load from LOAD_DIR, else 0.0.
# ---------------------

ATTACK="afog"
MODE="baseline"
CONFIG="models/detrex/projects/align_detr/configs/aligndetr_k=2_r50_4scale_24ep.py"
CHECKPOINT="model_files/aligndetr_r50_multi_4scale_24ep.pth"
SAMPLE=1.0
SAVE_DIR=""
SAVE_ATTACK=0.0
LOAD_DIR="datasets/blackbox/internimage/"
LOAD_ATTACK=0.0

python scripts/attack_detrex.py --config-file $CONFIG --eval-only  train.init_checkpoint=$CHECKPOINT attack=$ATTACK attack_mode=$MODE sample=$SAMPLE save_dir=$SAVE_DIR save_attack=$SAVE_ATTACK load_dir=$LOAD_DIR load_attack=$LOAD_ATTACK

