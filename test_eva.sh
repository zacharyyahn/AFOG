export DETECTRON2_DATASETS=/home/hice1/zyahn3/scratch/TOG_plus/dataset/

ATTACK="attention"
MODE="vanishing"
CONFIG="detrex/projects/dino_eva/configs/dino-eva-01/dino_eva_01_1280_4scale_12ep.py"
CHECKPOINT="model_files/dino_eva.pth"
SAMPLE=1.0

python attack_detrex.py --config-file $CONFIG --eval-only  train.init_checkpoint=$CHECKPOINT attack=$ATTACK attack_mode=$MODE sample=$SAMPLE
