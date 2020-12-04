import json
from collections import OrderedDict
import torch
import torch.nn.functional as F 
import cv2
import numpy as np
import glob
import os
from os.path import isfile, join

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

class SimpleLogger(object):

    def __init__(self, kwarg_list):
        self.kwarg_list = kwarg_list
        self.content = {}
        for kwarg in kwarg_list:
            self.content[kwarg] = []

    def update(self, loss, kwarg):
        assert kwarg in self.kwarg_list
        self.content[kwarg].append(loss)

    def get_best(self, kwarg, best='min'):
        assert kwarg in self.kwarg_list
        if best == 'min':
            best_val = min(self.content[kwarg])
        else:
            best_val = max(self.content[kwarg])

        return self.content[kwarg].index(best_val)



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
    
def write_text_to_img(img, text):
    # Write some Text
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (100, 100)
    fontScale              = 1
    fontColor              = (255, 255, 255)
    lineType               = 2

    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    return img

class Drawer(object):
    @staticmethod
    def draw_heatmap(path, data, src):
        data = resize_image(data, (224,224))
        heatmap = normalize_data_to_img255(data)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img = 0.5*src + 0.3*heatmap
        cv_write_numpy_to_image(img, path)

    # @staticmethod
    # def write_video_from_images(images, videoName):
    #     frame_array = []
    #     files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    #     # for sorting the file names properly
    #     files.sort(key=lambda x: x[5:-4])
    #     files.sort()

    #     size = (0, 0)

    #     for filename in files:
    #         print(filename)
    #         img = cv2.imread(  pathIn+"\\"+filename)
    #         height, width, layers = img.shape
    #         size = (width, height)
    #         #text = filename
    #         #write_text_to_img(img, text)
    #         frame_array.append(img)

    #     print(size)
    #     out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    #     for i in range(len(frame_array)):
    #         out.write(frame_array[i])

    #     out.release()
    #     print("finishing writing the video")

