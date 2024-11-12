lr=1e-4
lr_backbone=1e-5
batch_size=2
weight_decay=1e-4
epochs=300
lr_drop=200
clip_max_norm=0.1

frozen_weights=None

backbone="resnet50"
dilation=True
position_embedding="sine"

enc_layers=6
dec_layers=6
dim_feedforward=2048
hidden_dim=256
dropout=0.1
nheads=8
num_queries=100
pre_norm=False
masks=True

aux_loss=False
set_cost_class=1.0
set_cost_bbox=5.0
set_cost_giou=2.0

mask_loss_coef=1.0
dice_loss_coef=1.0
bbox_loss_coef=5.0
giou_loss_coef=2.0
eos_coef=0.1

dataset_file="coco"
coco_path=None
coco_panoptic_path=None
remove_difficult=True

output_dir=None
device="cuda"
seed=42
resume=None
start_epoch=0
eval=True
num_workers=1

world_size=1
dist_url=None
distributed=False

num_feature_levels=4
dec_n_points=4
enc_n_points=4
two_stage=False
with_box_refine=False
cls_loss_coef=2
focal_alpha=0.25
