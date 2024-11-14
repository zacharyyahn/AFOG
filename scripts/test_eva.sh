ATTACK="none"
MODE="vanishing"
CONFIG="detrex/projects/dino_eva/configs/dino-eva-01/dino_eva_01_1280_4scale_12ep.py"
CHECKPOINT="model_files/dino_eva.pth"
SAMPLE=0.001

python attack_detrex.py --config-file $CONFIG --eval-only  train.init_checkpoint=$CHECKPOINT attack=$ATTACK attack_mode=$MODE sample=$SAMPLE
