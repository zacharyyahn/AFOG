#!/bin/bash
#SBATCH -JAFOG_DINO
#SBATCH -N1 --gres=gpu:A100:1 --ntasks-per-node=1 --mem-per-gpu=64G
#SBATCH -t16:00:00
#SBATCH -ooutput/baseline/swin_afog_all.out

module load anaconda3
#conda init
conda deactivate
conda activate AFOG

# ---- PARAMETERS ----
# RESUME: the saved model checkpoint, typically: ".../model.pth"
# COCO_PATH: path to coco, typically: ".../coco"
# CONFIG: path to config file, typically: ".../DINO_4scale_swin.py"
# ATTACK: "afog" or "none"
# MODE: AFOG variant selection, either: "baseline", "vanishing", or "fabrication"
# SAMPLE_RATE: sampling rate for coco, usually 1.0
# LOAD_DIR: directory from which to load saved attacked images.
# LOAD_ATTACK: 1.0 if want to load from LOAD_DIR, else 0.0.
# --------------------

BATCH_SIZE=1
RESUME="model_files/checkpoint0011_4scale_swin.pth"
COCO_PATH="datasets/mini_coco/coco"
CONFIG="utils/dino_utils/config/DINO/DINO_4scale_swin.py"
ATTACK="afog"
MODE="baseline"
SAMPLE_RATE=1.0
LOAD_DIR="datasets/blackbox/internimage/"
LOAD_ATTACK=0.0

srun python scripts/attack_dino.py --output_dir logs/DINO/R50-MS4-%j -c $CONFIG --num_workers 8 --coco_path $COCO_PATH  \
	--eval --resume $RESUME --attack $ATTACK --attack_mode $MODE --sample_rate $SAMPLE_RATE --load_dir $LOAD_DIR --load_attack $LOAD_ATTACK --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 \
    use_ema=False dn_box_noise_scale=1.0 
