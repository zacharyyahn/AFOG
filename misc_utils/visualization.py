import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET


def get_gt_bboxes(annotation_path, meta):
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
    
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    dets = []
    for object in root.findall('object'):
        cls = class_name_to_index[object.findall('name')[0].text] - 1
        box = [int(object.findall('bndbox/xmin')[0].text) + meta[0], 
               int(object.findall('bndbox/ymin')[0].text) + meta[1], 
               int(object.findall('bndbox/xmax')[0].text) * meta[4] + meta[0], 
               int(object.findall('bndbox/ymax')[0].text) * meta[4] + meta[1]]
        conf = 1.0
        dets.append([cls, conf, box[0], box[1], box[2], box[3]])
    return dets

def visualize_detections(detections_dict):
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.clf()
    plt.figure(figsize=(3 * len(detections_dict), 3))
    for pid, title in enumerate(detections_dict.keys()):
        input_img, detections, model_img_size, classes = detections_dict[title]
        if len(input_img.shape) == 4:
            input_img = input_img[0]
        plt.subplot(1, len(detections_dict), pid + 1)
        plt.title(title)
        plt.imshow(input_img)
        current_axis = plt.gca()
        for box in detections:
            xmin = max(int(box[-4] * input_img.shape[1] / model_img_size[1]), 0)
            ymin = max(int(box[-3] * input_img.shape[0] / model_img_size[0]), 0)
            xmax = min(int(box[-2] * input_img.shape[1] / model_img_size[1]), input_img.shape[1])
            ymax = min(int(box[-1] * input_img.shape[0] / model_img_size[0]), input_img.shape[0])
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='small', color='black', bbox={'facecolor': color, 'alpha': 1.0})
        plt.axis('off')
    plt.show()

def graph_aps(aps_dict):
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus","car","cat","chair","cow", "diningtable","dog","horse", "motorbike","person","pottedplant","sheep",
               "sofa","train","tvmonitor"]
    
    for label, aps in aps_dict.items():
        plt.plot(aps)
    plt.legend(aps_dict.keys())
    plt.xticks(range(len(classes)), classes, size='small', rotation=70)
    plt.grid(True)
    plt.show()