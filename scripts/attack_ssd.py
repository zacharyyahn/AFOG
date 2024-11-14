import os
import sys

sys.path.append(os.getcwd()) 
from dataset_utils.voc.preprocessing import letterbox_image_padded
from dataset_utils.voc.eval import evaluate_dataset, evaluate_dataset_with_torch, evaluate_dataset_with_builtin
from misc_utils.visualization import visualize_detections
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

weights = 'model_files/SSD300.h5'
detector = SSD300(weights=weights)

eps = 8 / 255.       # Hyperparameter: epsilon in L-inf norm
eps_iter = 2 / 255.  # Hyperparameter: attack learning rate
n_iter = 10          # Hyperparameter: number of attack iterations
mini_path = "datasets/MiniVOC/"
im_path = "datasets/VOCdevkit/VOC2007/JPEGImages/"
annot_path = "datasets/VOCdevkit/VOC2007/Annotations/"
fail_path = "datasets/AttackFails/FailImages/"
attack = afog_cnn
    
scores = evaluate_dataset_with_builtin(detector, im_path, annot_path, num_examples=1500, attack=attack, attack_params={"n_iter": n_iter, "eps": eps, "eps_iter":eps_iter}, flag_attack_fail=False)

print("scores are:", scores)


