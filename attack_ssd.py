from dataset_utils.voc.preprocessing import letterbox_image_padded
from dataset_utils.voc.eval import evaluate_dataset, evaluate_dataset_with_torch, evaluate_dataset_with_builtin
from misc_utils.visualization import visualize_detections
#from models.frcnn import FRCNN
from PIL import Image
from models.ssd import SSD300
from afog.attacks import *
from tqdm import tqdm
import os
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

np.random.seed(42)

#weights = 'model_files/FRCNN.pth'  # TODO: Change this path to the victim model's weights
weights = 'model_files/SSD300.h5'

#detector = FRCNN().cuda(device=0).load(weights)
detector = SSD300(weights=weights)

eps = 8 / 255.       # Hyperparameter: epsilon in L-inf norm
eps_iter = 2 / 255.  # Hyperparameter: attack learning rate
n_iter = 10          # Hyperparameter: number of attack iterations
mini_path = "datasets/MiniVOC/"
im_path = "datasets/VOCdevkit/VOC2007/JPEGImages/"
annot_path = "datasets/VOCdevkit/VOC2007/Annotations/"
fail_path = "datasets/AttackFails/FailImages/"
attack = afog_cnn

#scores = evaluate_dataset(detector, path, num_examples=500, attack=None)
#print("(benign) mAP is:", scores["map"])

# scores = evaluate_dataset(detector, path, attack=tog_attention, attack_params={"n_iter": n_iter, "eps": eps, "eps_iter":eps_iter})
# print("(attention) mAP is:", scores["map"])

# print(device_lib.list_local_devices())
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    
scores = evaluate_dataset_with_builtin(detector, im_path, annot_path, num_examples=1500, attack=attack, attack_params={"n_iter": n_iter, "eps": eps, "eps_iter":eps_iter}, flag_attack_fail=False)

print("scores are:", scores)

#    scores = evaluate_dataset(detector, path, num_examples=500, attack=tog_fabrication, attack_params={"n_iter": n_iter, "eps": eps, "eps_iter":eps_iter})
#    print("(fabrication) mAP is:", scores["map"])

#    scores = evaluate_dataset(detector, path, num_examples=500, attack=tog_vanishing, attack_params={"n_iter": n_iter, "eps": eps, "eps_iter":eps_iter})
#    print("(vanishing) mAP is:", scores["map"])


