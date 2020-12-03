import json
from collections import OrderedDict
import torch
import torch.nn.functional as F 
import cv2
import numpy as np

## json file IO
def read_json(fname):
    with open(fname, 'rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with open(fname, 'wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def write_json_no_indent(content, fname):
    with open(fname, 'wt') as handle:
        json.dump(content, handle, sort_keys=False)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SimpleLossLogger(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_list = []

    def update(self, loss):
        self.loss_list.append(loss)

    def get_best(self):
        max_val = min(self.loss_list)
        return self.loss_list.index(max_val)



def normalize_data_to_img255(data):
    min_val = np.amin(data)
    max_val = np.amax(data)
    # RESCALING THE DATA to 0 255
    img = (data - min_val) / (max_val+10e-5-min_val) * 255
    img = img.astype(np.uint8)
    return img

def cv_write_numpy_to_image(img, fname, debug=False):
    fname = fname+".png"
    cv2.imwrite(fname, img)
    if debug:
        print(f"draw {fname}")

def resize_image(img, size):
    img_resized = cv2.resize(img, size)
    return img_resized

class Drawer(object):
    @staticmethod
    def draw_heatmap(path, data, src):
        data = resize_image(data, (224,224))
        heatmap = normalize_data_to_img255(data)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img = 0.5*src + 0.3*heatmap
        cv_write_numpy_to_image(img, path)

    