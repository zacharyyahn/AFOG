from dataset_utils.preprocessing import letterbox_image_padded
from dataset_utils.eval import evaluate_image
from misc_utils.visualization import visualize_detections
from models.frcnn import FRCNN
from PIL import Image
from tog.attacks import *
from tqdm import tqdm
import os
import xml.etree.ElementTree as ET

weights = 'model_files/FRCNN.pth'  # TODO: Change this path to the victim model's weights

detector = FRCNN().cuda(device=0).load(weights)

eps = 8 / 255.       # Hyperparameter: epsilon in L-inf norm
eps_iter = 2 / 255.  # Hyperparameter: attack learning rate
n_iter = 10          # Hyperparameter: number of attack iterations

aps = np.zeros(20)
cts = np.zeros(20)
for path in tqdm(os.listdir("dataset/VOCdevkit/VOC2007/JPEGImages/")[:50]):
    this_aps, this_cts = evaluate_image(detector, path[:-4], attack=None)
    aps += this_aps
    cts += this_cts
aps = aps/cts
print("APs are", aps)
print("mAP is", np.mean(aps))
