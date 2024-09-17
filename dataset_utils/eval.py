from PIL import Image
from .preprocessing import letterbox_image_padded
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torchmetrics
from frcnn_utils.eval_tool import eval_detection_voc
import torch
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

def evaluate_dataset_with_builtin(detector, im_path, annot_path, num_examples=-1, attack=None, attack_params={"n_iter": 10, "eps": 8/255., "eps_iter":2/255.}, flag_attack_fail=False):
    
    pred_bboxes_total = []
    pred_scores_total = []
    pred_labels_total = []
    gt_bboxes_total = []
    gt_labels_total = []
    
    for path in tqdm(os.listdir(im_path)[:num_examples]):
        
        # Preprocess the image
        try:
            im = Image.open(im_path + path)
        except:
            print(path, "is not an image.")
            continue
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
               
        pred_boxes = np.zeros((len(detections), 4))
        pred_scores = np.zeros(len(detections))
        pred_labels = np.zeros(len(detections))
                                  

        # for every detection, extract the four bbox coords, the score, and the conf label    
        for i, det in enumerate(detections):    
            # get the box and make sure we rescale predictions to what the annotation is expecting
            cls = int(det[0])
            score = det[1]
            xmin = max(det[-4] * im.shape[2] / detector.model_img_size[1], 0)
            ymin = max(det[-3] * im.shape[1] / detector.model_img_size[0], 0)
            xmax = min(det[-2] * im.shape[2] / detector.model_img_size[1], im.shape[2])
            ymax = min(det[-1] * im.shape[1] / detector.model_img_size[0], im.shape[1])
                         
            pred_boxes[i, 0] = xmin
            pred_boxes[i, 1] = ymin
            pred_boxes[i, 2] = xmax
            pred_boxes[i, 3] = ymax
            pred_scores[i] = score
            pred_labels[i] = int(cls)
                                
        # parse the annotations and shift them according to how the input image was padded and scaled
        tree = ET.parse(annot_path + path[:-4] + ".xml")
        root = tree.getroot()
        
        gt_boxes = np.zeros((len(root.findall('object')), 4))
        gt_labels = np.zeros(len(root.findall('object')))
        
        for i, object in enumerate(root.findall('object')):
            cls = class_name_to_index[object.findall('name')[0].text] - 1
            xmin = float(object.findall('bndbox/xmin')[0].text) + meta[0] 
            ymin = float(object.findall('bndbox/ymin')[0].text) + meta[1] 
            xmax = float(object.findall('bndbox/xmax')[0].text) * meta[4] + meta[0]
            y_max = float(object.findall('bndbox/ymax')[0].text) * meta[4] + meta[1]

            gt_boxes[i, 0] = xmin
            gt_boxes[i, 1] = ymin
            gt_boxes[i, 2] = xmax
            gt_boxes[i, 3] = ymax
            gt_labels[i] = int(cls)
        pred_bboxes_total.append(pred_boxes)
        pred_labels_total.append(pred_labels)
        pred_scores_total.append(pred_scores)
        gt_bboxes_total.append(gt_boxes)
        gt_labels_total.append(gt_labels)
        
    scores = eval_detection_voc(pred_bboxes=pred_bboxes_total, pred_labels=pred_labels_total, pred_scores=pred_scores_total, gt_bboxes=gt_bboxes_total, gt_labels=gt_labels_total, iou_thresh=0.5, use_07_metric=True)
    return scores

def evaluate_dataset_with_torch(detector, im_path, annot_path, num_examples=-1, attack=None, attack_params={"n_iter": 10, "eps": 8/255., "eps_iter":2/255.}, flag_attack_fail=False):
    
    map_metric = torchmetrics.detection.MeanAveragePrecision(iou_type='bbox', extended_summary=True, class_metrics=True)
    
    for path in tqdm(os.listdir(im_path)[:num_examples]):
        
        # Preprocess the image
        im = Image.open(im_path + path)
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
        
        preds = []
        
        pred_boxes = torch.empty((len(detections), 4))
        pred_scores = torch.empty(len(detections))
        pred_labels = torch.empty(len(detections), dtype=torch.int64)
                                  

        # for every detection, extract the four bbox coords, the score, and the conf label    
        for i, det in enumerate(detections):    
            # get the box and make sure we rescale predictions to what the annotation is expecting
            cls = int(det[0])
            score = det[1]
            xmin = max(det[-4] * im.shape[2] / detector.model_img_size[1], 0)
            ymin = max(det[-3] * im.shape[1] / detector.model_img_size[0], 0)
            xmax = min(det[-2] * im.shape[2] / detector.model_img_size[1], im.shape[2])
            ymax = min(det[-1] * im.shape[1] / detector.model_img_size[0], im.shape[1])
                         
            pred_boxes[i, 0] = xmin
            pred_boxes[i, 1] = ymin
            pred_boxes[i, 2] = xmax
            pred_boxes[i, 3] = ymax
            pred_scores[i] = score
            pred_labels[i] = int(cls)
                                
        # parse the annotations and shift them according to how the input image was padded and scaled
        tree = ET.parse(annot_path + path[:-4] + ".xml")
        root = tree.getroot()
        gt_boxes = torch.empty((len(root.findall('object')), 4))
        gt_labels = torch.empty(len(root.findall('object')), dtype=torch.int64)
        
        for i, object in enumerate(root.findall('object')):
            cls = class_name_to_index[object.findall('name')[0].text] - 1
            xmin = float(object.findall('bndbox/xmin')[0].text) + meta[0] 
            ymin = float(object.findall('bndbox/ymin')[0].text) + meta[1] 
            xmax = float(object.findall('bndbox/xmax')[0].text) * meta[4] + meta[0]
            y_max = float(object.findall('bndbox/ymax')[0].text) * meta[4] + meta[1]

            gt_boxes[i, 0] = xmin
            gt_boxes[i, 1] = ymin
            gt_boxes[i, 2] = xmax
            gt_boxes[i, 3] = ymax
            gt_labels[i] = int(cls)

        preds = [dict(boxes=pred_boxes, scores=pred_scores, labels=pred_labels)]
        gts = [dict(boxes=gt_boxes, labels=gt_labels)]
        print(preds)
        print(gts)
        map_metric.update(preds, gts)
    the_map = map_metric.compute()
    return the_map
    
        
    

"""
Accumulate TP and FP numbers for all classes for all images in a dataset to calculate mAP by calling evaluate_image() on every image. 

Returns a dictionary of metrics containing aps for each class and map.

- detector: model object with a model_img_size attribute and a detect() method.
- im_path: string path to the directory of images for evaluating. Iterates through these and checks annot_path for corresponding annotation.
- annot_path: string path to the directory of annotations for ground-truths.
- num_examples: int for how many example images to go through. -1 to go through all images.
- attack: function for TOG attack.
- attack_params: dict containing TOG attack parameters.
- flag_attack_fail: bool for whether to save images that the attack failed on in a different file. 
"""
def evaluate_dataset(detector, im_path, annot_path, num_examples=-1, attack=None, attack_params={"n_iter": 10, "eps": 8/255., "eps_iter":2/255.}, flag_attack_fail=False): 
    
    # Array of shape (num_classes, num_iou_iterations, 2)
    tpfp = np.zeros((len(class_index_to_name.keys())-1, len(IOU_RANGE), 2))
    counts = np.zeros(len(class_index_to_name.keys())-1)
    
    # Iterate through each image in im_path up to num_examples and add their TP/FP numbers to the running total
    for path in tqdm(os.listdir(im_path)[:num_examples]):
        this_tpfp, this_counts = evaluate_image(detector, im_path, annot_path, path, attack, attack_params, flag_attack_fail=flag_attack_fail)
        tpfp += this_tpfp
        counts += this_counts
    
    # Helper function that calculates precision along an axis
    def prec(a):
        if a[0] == 0:
            return 0.0
        return (1. * a[0]) / (a[0] + a[1])
    
    # Apply precision function along axis of TP/FP accumulation array
    precs = np.apply_along_axis(prec, 2, tpfp)
    aps = np.mean(precs, axis=1)
    the_map = np.sum(np.multiply(counts, aps)) / np.sum(counts)  
    
    scores = {
        "aps":aps,
        "map":the_map
    }
    return scores

"""
Accumulate TP and FP numbers for a single image. 

Returns a numpy array of shape (num_classes, num_iou_iterations, 2)

- detector: model object with a model_img_size attribute and a detect() method.
- im_path: string path to the directory of images for evaluating. Iterates through these and checks annot_path for corresponding annotation.
- annot_path: string path to the directory of annotations for ground-truths.
- im_num: string name of image to evaluate, including extension such as .jpg
- attack: function for TOG attack
- attack_params: dict containing TOG attack parameters
- flag_attack_fail: bool for whether to save images that the attack failed on in a different file. 
""" 
def evaluate_image(detector, im_path, annot_path, im_num, attack=None, attack_params={"n_iter": 10, "eps": 8/255., "eps_iter":2/255.}, flag_attack_fail=False):
        
    # Preprocess the image
    try:
        im = Image.open(im_path + im_num)
    except:
        print(im_num, "is not an image.")
        return np.zeros((len(class_index_to_name.keys())-1, len(IOU_RANGE), 2))
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
    for det in detections:    
        # get the box and make sure we rescale predictions to what the annotation is expecting
        cls = int(det[0])
        score = det[1]
        xmin = int(max(int(det[-4] * im.shape[2] / detector.model_img_size[1]), 0))
        ymin = int(max(int(det[-3] * im.shape[1] / detector.model_img_size[0]), 0))
        xmax = int(min(int(det[-2] * im.shape[2] / detector.model_img_size[1]), im.shape[2]))
        ymax = int(min(int(det[-1] * im.shape[1] / detector.model_img_size[0]), im.shape[1]))
       
        # insert into dictionary where key is class number
        pred_dict[cls].append([xmin, ymin, xmax, ymax])
        score_dict[cls].append(score)
    
    # parse the annotations and shift them according to how the input image was padded and scaled
    tree = ET.parse(annot_path + im_num[:-4] + ".xml")
    root = tree.getroot()
    for object in root.findall('object'):
        cls = class_name_to_index[object.findall('name')[0].text] - 1
        box = [int(object.findall('bndbox/xmin')[0].text) + meta[0], 
               int(object.findall('bndbox/ymin')[0].text) + meta[1], 
               int(object.findall('bndbox/xmax')[0].text) * meta[4] + meta[0], 
               int(object.findall('bndbox/ymax')[0].text) * meta[4] + meta[1]]
        
        # insert into dictionary 
        # items in pred dict are [conf box box box box]
        gt_dict[cls].append(box)
    
    # Record the true positives and false positives
    tpfp = np.zeros((len(class_index_to_name.keys())-1, len(IOU_RANGE), 2))
    counts = np.zeros(len(class_index_to_name.keys())-1)
    for cls in pred_dict.keys():
        if len(pred_dict[cls]) == 0 and len(gt_dict[cls]) == 0:
            continue # no TPs or FPs because no preds for this class
        elif len(pred_dict[cls]) == 0:
            continue # we have no preds but there are GTs, all FNs. come back here if FNs matter later
        elif len(gt_dict[cls]) == 0:
            tpfp[cls, :, 1] = len(pred_dict[cls]) * np.ones(len(IOU_RANGE)) # if we have preds but no gt then all FPs
        else: 
            counts[cls] += len(pred_dict[cls])
            tpfp[cls] += calculate_tpfp(pred_dict[cls], gt_dict[cls], score_dict[cls], detector)
    
    
    # if we want to track which images were not corrupted by the attack effectively
    if flag_attack_fail:
        attack_fail_save_dir = "dataset/AttackFails/attack_fails2.txt"
        attack_fail_tp_thresh = 1
        
        tps = np.sum(tpfp[:, tpfp.shape[1] // 2, 0]) #look at the middle of the IoU threshold range

        if tps >= attack_fail_tp_thresh: # if we find even a single TP at the middle of the threshold
            f = open(attack_fail_save_dir, 'a')
            print("Image", im_num, "has tps", tpfp[:, :, 0].flatten())
            f.write(im_num + "\n")
            f.close()
    return tpfp, counts

"""
Calculate the IoU between two boxes

returns a float iou

- box1: array of length 4 containing a box with [minx, miny, maxx, maxy]
- box2: array of length 4 containing a box of same shape as box1
- model_img_size: the size image the model is expecting to receive
"""
def calculate_iou(box1, box2, model_img_size):
    
    # find where the boxes intersect. If they go over the edge of the image then set that value to the edge of the image
    min_x = max(0, box1[0] if box1[0] > box2[0] else box2[0])
    min_y = max(0, box1[1] if box1[1] > box2[1] else box2[1])
    max_x = min(model_img_size[0], box1[2] if box1[2] < box2[2] else box2[2])
    max_y = min(model_img_size[1], box1[3] if box1[3] < box2[3] else box2[3])
    
    #check to see if they overlap at all
    if min_x > max_x or min_y > max_y:
        return 0.0

    intersect_area = (max_x - min_x) * (max_y - min_y)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersect_area
    return float(intersect_area) / union_area

""" 
Calculate an array of tp and fp values for each IoU threshold for just one class.

Returns numpy array of shape (1, num_iou_iter, 2)

- pred_boxes: list of bboxes of length num_classes, where each bbox is a list of shape [minx, miny, maxx, maxy]
- gt_boxes: list of gt_boxes of length num_classes, where each bbox is a list of the same shape as pred_boxes
- pred_scores: list of predicted objectness scores, which are floats
- detector: detector model object
"""
def calculate_tpfp(pred_boxes, gt_boxes, pred_scores, detector):
    this_tpfp = np.zeros((len(IOU_RANGE), 2)) #records TPs and FPs for each IOU value

    # sort by scores in descending order
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = np.array(pred_boxes)[sorted_indices]
    pred_scores = np.array(pred_scores)[sorted_indices]

    # pre-compute IoUs because we loop over them many times
    ious = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            ious[i, j] = calculate_iou(gt_box, pred_box, detector.model_img_size)
    
    for iou_thresh in IOU_RANGE:
        num_tp = 0
        num_fp = 0
        num_gt_boxes = len(gt_boxes)
        matched_gt_boxes = [False] * num_gt_boxes

        # for every pred box, in descending objectness score order, find the gt box that best aligns with it based on IoU. 
        # If the IoU is greater than the threshold and the box has not yet been matched, 
        # then match them and give a TP. If the best one is already matched or the IoU is not greater than the threshold
        # then it is a FP.
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_idx = -1
    
            for j, gt_box in enumerate(gt_boxes):
                iou = ious[j, i] # use pre-computed IoUs
                
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_iou >= iou_thresh and not matched_gt_boxes[best_idx]:
                matched_gt_boxes[best_idx] = True
                #print("For IoU thresh", iou_thresh, "pred box", i, "matched gt box", best_idx, "and got a TP")
                this_tpfp[int((iou_thresh - IOU_START) / IOU_ITER)][0] += 1
            else:
                #print("For IoU thresh", iou_thresh, "pred box", i, "got a FP")
                this_tpfp[int((iou_thresh - IOU_START) / IOU_ITER)][1] += 1
    return this_tpfp
    
"""
Loops through every image in the im_path directory and finds images where applying the attack caused the model to recognize a TP
that it did not see before.

Returns none

- detector: model object with a model_img_size attribute and a detect() method.
- im_path: string path to the directory of images for evaluating. Iterates through these and checks annot_path for corresponding annotation.
- annot_path: string path to the directory of annotations for ground-truths.
- num_examples: int for how many example images to go through. -1 to go through all images.
- attack: function for TOG attack.
- attack_params: dict containing TOG attack parameters.
"""
def flag_induced_tp(detector, im_path, annot_path, num_examples=-1, attack=None, attack_params={"n_iter": 10, "eps": 8/255., "eps_iter":2/255.}): 
    
    print("--- Images where attack induced TP ---")
    # Iterate through each image in im_path up to num_examples and add their TP/FP numbers to the running total for respective version
    for path in tqdm(os.listdir(im_path)[:num_examples]):
        tpfp_benign = evaluate_image(detector, im_path, annot_path, path, attack=None, attack_params=attack_params)
        tpfp_adv = evaluate_image(detector, im_path, annot_path, path, attack=attack, attack_params=attack_params)
    
        tps_benign = np.sum(tpfp_benign[:, tpfp_benign.shape[1] // 2, 0]) #look at the middle of the IoU threshold range
        tps_adv = np.sum(tpfp_adv[:, tpfp_adv.shape[1] // 2, 0])
        print("Im:", path, "has Benign TPs and Adv TPs::")
        print(tpfp_benign[:, tpfp_benign.shape[1] // 2, 0])
        print(tpfp_adv[:, tpfp_adv.shape[1] // 2, 0])
        if tps_benign < tps_adv:
            print(path)