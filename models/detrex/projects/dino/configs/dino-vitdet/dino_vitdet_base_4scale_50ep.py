from detrex.config import get_config

from .dino_vitdet_base_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    model,
)

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep

ATTACK=None
ATTACK_MODE=None
SAMPLE=1.0
SAVE_ATTACK=None
SAVE_DIR=None

# modify training config
train.max_iter = 375000
train.init_checkpoint = "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth"
train.output_dir = "./output/dino_vitdet_base_50ep"
