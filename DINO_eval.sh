coco_path=dataset/coco
checkpoint=model_files/checkpoint0011_4scale.pth
attack="attention"
python attack_dino.py \
  --output_dir logs/DINO/R50-MS4-%j \
	-c dino_utils/config/DINO/DINO_4scale.py --coco_path $coco_path  \
	--eval --resume $checkpoint --attack $attack \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
