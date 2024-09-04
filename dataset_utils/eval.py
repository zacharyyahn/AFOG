from PIL import Image
from .preprocessing import letterbox_image_padded
import xml.etree.ElementTree as ET
import numpy as np

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

def evaluate_image(detector, path, attack, attack_params={"n_iter": 10, "eps": 8/255., "eps_iter":2/255.}):
    base_image_path = "dataset/VOCdevkit/VOC2007/JPEGImages/"
    base_annotation_path = "dataset/VOCdevkit/VOC2007/Annotations/"
        
    #Preprocess the image
    im = Image.open(base_image_path + path + ".jpg")
    im, meta = letterbox_image_padded(im, size=detector.model_img_size)
    
    #Run the attack, if any
    if attack is not None:
        im = attack(victim=detector, x_query=im, n_iter=attack_params["n_iter"], eps=attack_params["eps"], eps_iter=attack_params["eps_iter"])
    
    #Make detections on the image and get the ground-truth labels
    detections = detector.detect(im, conf_threshold=detector.confidence_thresh_default)
    pred_labels = detections[:, 0].flatten()
    pred_scores = detections[:, 1].flatten()
    gt_labels = np.zeros(20)
    tree = ET.parse(base_annotation_path + path + ".xml")
    root = tree.getroot()
    for object in root.findall('object/name'):
        gt_labels[class_name_to_index[object.text]-1] += 1
    
    #Record the true positives and false positives
    fps = 0
    tps = 0
    for item in pred_labels:
        if gt_labels[int(item)] == 0:
            fps += 1
        else:
            gt_labels[int(item)] -= 1
            tps += 1
    return tps, fps

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

def calculate_ap(pred_boxes, gt_boxes, pred_scores):

    total_precision = 0

    # sort by scores in descending order
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = np.array(pred_boxes)[sorted_indices]
    pred_scores = np.array(pred_scores)[sorted_indices]

    # pre-compute IoUs because we loop over them many times
    ious = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            ious[i, j] = calculate_iou(gt_box, pred_box)

    for iou_thresh in np.arange(0.1, 1.0, 0.1):
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
                iou = ious[i, j] # use pre-computed IoUs
                
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
                    #print("Best IoU for pred box", i, "is gt box", j)

            if best_iou > iou_thresh and not matched_gt_boxes[best_idx]:
                matched_gt_boxes[best_idx] = True
                #print("For IoU thresh", iou_thresh, "pred box", i, "matched gt box", best_idx, "and got a TP")
                num_tp += 1
            else:
                #print("For IoU thresh", iou_thresh, "pred box", i, "got a FP")
                num_fp += 1

        precision = num_tp / (num_tp + num_fp)
        total_precision += precision
    
    ap = total_precision / 9.0 #there are 9 thresholds between [0.1, 0.9]
    return ap
