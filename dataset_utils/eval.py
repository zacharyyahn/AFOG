from PIL import Image
from .preprocessing import letterbox_image_padded
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os

IOU_START = 0.5
IOU_END = 0.6
IOU_ITER = 0.1
IOU_RANGE = np.arange(IOU_START, IOU_END, IOU_ITER)

class_index_to_name = {
    0:  "background",
    1:  "aeroplane",
    2:  "bicycle",
    3:  "bird",
    4:  "boat",
    5:  "bottle",
    6:  "bus",
    7:  "car",
    8:  "cat",
    9:  "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor"
  }

class_name_to_index = {name:idx for idx, name in class_index_to_name.items()}

# Accumulate TP and FP numbers for all classes for all images in a dataset to calculate mAP
def evaluate_dataset(detector, path, attack=None, attack_params={"n_iter": 10, "eps": 8/255., "eps_iter":2/255.}):    
    tpfp = np.zeros((len(class_index_to_name.keys())-1, len(IOU_RANGE), 2))
    
    for path in tqdm(os.listdir(path)[:250]):
        this_tpfp = evaluate_image(detector, path[:-4], attack, attack_params)
        tpfp += this_tpfp
    
    def prec(a):
        if a[0] == 0:
            return 0.0
        return (1. * a[0]) / (a[0] + a[1])
    
    precs = np.apply_along_axis(prec, 2, tpfp)
    aps = np.mean(precs, axis=1)
    print("aps:", aps)
    the_map = np.mean(aps)    
    
    scores = {
        "map":the_map
    }
    return scores

# Get tp and fp numbers for every class 
def evaluate_image(detector, path, attack=None, attack_params={"n_iter": 10, "eps": 8/255., "eps_iter":2/255.}):
    base_image_path = "dataset/VOCdevkit/VOC2007/JPEGImages/"
    base_annotation_path = "dataset/VOCdevkit/VOC2007/Annotations/"
        
    # Preprocess the image
    im = Image.open(base_image_path + path + ".jpg")
    im, meta = letterbox_image_padded(im, size=detector.model_img_size)
        
    # Run the attack, if any
    if attack is not None:
        im = attack(
            victim=detector, 
            x_query=im, 
            n_iter=attack_params["n_iter"], 
            eps=attack_params["eps"], 
            eps_iter=attack_params["eps_iter"]
        )
    
    # Make detections on the image and get the ground-truth labels
    detections = detector.detect(im, conf_threshold=detector.confidence_thresh_default)
    
    pred_dict = defaultdict(lambda: [])
    gt_dict = defaultdict(lambda: [])
    score_dict = defaultdict(lambda: [])
    
    # for every detection, store a dictionary where the class maps to all boxes of that class [2:-1] and its confidence score [1]
    #print("Detections is", detections)
    
    for det in detections:    
        # get the box and make sure we rescale predictions to what the annotation is expecting
        cls = int(det[0])
        score = det[1]
        xmin = int(max(int(det[-4] * im.shape[2] / detector.model_img_size[1]), 0))
        ymin = int(max(int(det[-3] * im.shape[1] / detector.model_img_size[0]), 0))
        xmax = int(min(int(det[-2] * im.shape[2] / detector.model_img_size[1]), im.shape[2]))
        ymax = int(min(int(det[-1] * im.shape[1] / detector.model_img_size[0]), im.shape[1]))
       
        # insert into dictionary
        pred_dict[cls].append([xmin, ymin, xmax, ymax])
        score_dict[cls].append(score)
    
    # parse the annotations and shift them according to how the input image was padded and scaled
    tree = ET.parse(base_annotation_path + path + ".xml")
    root = tree.getroot()
    for object in root.findall('object'):
        cls = class_name_to_index[object.findall('name')[0].text] - 1
        box = [int(object.findall('bndbox/xmin')[0].text) + meta[0], 
               int(object.findall('bndbox/ymin')[0].text) + meta[1], 
               int(object.findall('bndbox/xmax')[0].text) * meta[4] + meta[0], 
               int(object.findall('bndbox/ymax')[0].text) * meta[4] + meta[1]]
        
        # insert into dictionary 
        gt_dict[cls].append(box)
    
    # Record the true positives and false positives
    tpfp = np.zeros((len(class_index_to_name.keys())-1, len(IOU_RANGE), 2))
    for cls in pred_dict.keys():
        if len(pred_dict[cls]) > 0 and len(gt_dict[cls]) > 0:
            # items in pred dict are [conf box box box box]
            tpfp[cls] += calculate_tpfp(pred_dict[cls], gt_dict[cls], score_dict[cls])
    return tpfp

# Calculate the IOU between two boxes
def calculate_iou(box1, box2):
    #find where the boxes intersect
    min_x = box1[0] if box1[0] > box2[0] else box2[0]
    min_y = box1[1] if box1[1] > box2[1] else box2[1]
    max_x = box1[2] if box1[2] < box2[2] else box2[2]
    max_y = box1[3] if box1[3] < box2[3] else box2[3]
    
    #check to see if they overlap at all
    if min_x > max_x or min_y > max_y:
        return 0

    intersect_area = (max_x - min_x) * (max_y - min_y)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersect_area
    return float(intersect_area) / union_area

# Calculate an array of tp and fp values for each IoU threshold for just one class
def calculate_tpfp(pred_boxes, gt_boxes, pred_scores):
    this_tpfp = np.zeros((len(IOU_RANGE), 2)) #records TPs and FPs for each IOU value

    # sort by scores in descending order
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = np.array(pred_boxes)[sorted_indices]
    pred_scores = np.array(pred_scores)[sorted_indices]

    # pre-compute IoUs because we loop over them many times
    ious = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            ious[i, j] = calculate_iou(gt_box, pred_box)
    for iou_thresh in IOU_RANGE:
        num_tp = 0
        num_fp = 0
        num_gt_boxes = len(gt_boxes)
        matched_gt_boxes = [False] * num_gt_boxes

        # for every pred box, find the gt box that best aligns with it. If the IoU is greater than the threshold and the box has not
        # yet been matched, then match them and give a TP. If the best one is already matched or the IoU is not greater than the threshold
        # then it is a FP.
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_idx = -1
    
            for j, gt_box in enumerate(gt_boxes):
                iou = ious[j, i] # use pre-computed IoUs
                
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_iou > iou_thresh and not matched_gt_boxes[best_idx]:
                matched_gt_boxes[best_idx] = True
                #print("For IoU thresh", iou_thresh, "pred box", i, "matched gt box", best_idx, "and got a TP")
                this_tpfp[int((iou_thresh - IOU_START) / IOU_ITER)][0] += 1
            else:
                #print("For IoU thresh", iou_thresh, "pred box", i, "got a FP")
                this_tpfp[int((iou_thresh - IOU_START) / IOU_ITER)][1] += 1
    return this_tpfp
