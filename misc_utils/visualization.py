import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

"""
Returns a list of gt_boxes of shape [1, 6] where each box is [class_label, conf_score, minx, miny, maxx, maxy]
for use in visualize_detections()

- annotation_path: string path to annotations folder
- im_num: which image to use, including extension such as .jpg
- meta: list return by letterbox_image_padded(), depicts how image was scaled to meet model_img_size

"""

import numpy as np
import matplotlib



# shiftedColorMap code from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap', plt=None):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    #matplotlib.colormaps.register(cmap=newcmap)

    return newcmap


def get_gt_bboxes(annotation_path, im_num, meta):
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
    
    tree = ET.parse(annotation_path + im_num[:im_num.find(".")] + ".xml")
    root = tree.getroot()
    dets = np.zeros((len(root.findall('object')), 6))
    i = 0     
    for object in root.findall('object'):
        cls = class_name_to_index[object.findall('name')[0].text] - 1
        box = [int(object.findall('bndbox/xmin')[0].text) * meta[4] + meta[0], 
               int(object.findall('bndbox/ymin')[0].text) * meta[4] + meta[1], 
               int(object.findall('bndbox/xmax')[0].text) * meta[4] + meta[0], 
               int(object.findall('bndbox/ymax')[0].text) * meta[4] + meta[1]]
        conf = 1.0
        dets[i,0] = cls
        dets[i,1] = conf
        dets[i,2] = box[0]
        dets[i,3] = box[1]
        dets[i,4] = box[2]
        dets[i,5] = box[3]
        i += 1
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
    plt.legend(aps_dict.keys(), loc='upper right')
    plt.xticks(range(len(classes)), classes, size='small', rotation=70)
    plt.grid(True)
    plt.show()