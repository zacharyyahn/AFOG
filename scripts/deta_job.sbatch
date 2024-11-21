#!/bin/bash
#SBATCH -JAFOG_DETA
#SBATCH -N1 --gres=gpu:A100:1 --ntasks-per-node=1 --mem-per-gpu=128G
#SBATCH -t24:00:00
#SBATCH -ooutput/baseline/deta_afog_all.out

module load anaconda3
#conda init
conda deactivate
conda activate AFOG

# ---- PARAMETERS ----
# ATTACK: "afog" or "none"
# MODE: AFOG variant selection, either: "baseline", "vanishing", or "fabrication"
# CONFIG: the path to the Detrex config, typically: ".../deta_swin_large_finetune_24ep.py"
# CHECKPOINT: the path to the mode file, typically: ".../model.pth"
# SAMPLE: sample of dataset, usually 1.0
# SAVE_DIR: directory to save attacked images.
# SAVE_ATTACK: 1.0 if want to save to SAVE_DIR, else 0.0
# LOAD_DIR: directory from which to load saved attacked images.
# LOAD_ATTACK: 1.0 if want to load from LOAD_DIR, else 0.0.
# ---------------------

ATTACK="afog"
MODE="baseline"
CONFIG="models/detrex/projects/deta/configs/deta_swin_large_finetune_24ep.py"
CHECKPOINT="model_files/deta.pth"
SAMPLE=1.0
SAVE_ATTACK=0.0
SAVE_DIR=""
LOAD_ATTACK=0.0
LOAD_DIR="datasets/blackbox/internimage/"

python scripts/attack_detrex.py --config-file $CONFIG --eval-only  train.init_checkpoint=$CHECKPOINT attack=$ATTACK attack_mode=$MODE sample=$SAMPLE save_attack=$SAVE_ATTACK save_dir=$SAVE_DIR save_attack=$SAVE_ATTACK save_dir=$SAVE_DIR load_attack=$LOAD_ATTACK load_dir=$LOAD_DIR

