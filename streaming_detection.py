from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
import time

import os
import argparse
from datetime import datetime
from mynet import StreamingDataloader
from mynet import MyNet
from utils import Drawer
from selective_search import customized_selective_search as ss_box_finder

BOX_SCORE_THRESHOLD = 0.1


def box_cam_intersection(boxes, cams, scale_wh):
    scale_wh = (1, scale_wh[0], scale_wh[1])
    cams_reshape = scipy.ndimage.zoom(cams, scale_wh)  ## [3, 224, 398]
    boxes = np.array(boxes)  ## [N, 4]

    def generate_mask(boxes, shape):
        shape = (boxes.shape[0], shape[0], shape[1])
        tmp = np.zeros(shape)
        for idx, box in enumerate(boxes):
            tmp[idx, box[1]:box[3], box[0]:box[2]] = 1
        return tmp

    H, W = cams_reshape.shape[1:]
    masks = generate_mask(boxes, (H, W))  ## [N, H, W] X [C, H, W]
    box_areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    scores = masks.reshape(-1, 1, H, W) * cams_reshape.reshape(1, -1, H, W)
    scores = scores.sum(axis=(2, 3)) / box_areas.reshape(-1, 1)
    max_vals = np.amax(scores, axis=-1)
    max_idxs = np.argmax(scores, axis=-1)
    return max_vals, max_idxs


class Detector(object):

    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device

    def __call__(self, image_path, draw_path):
        with torch.no_grad():
            batch = self.data_loader(image_path)
            data = batch[0].to(self.device).unsqueeze(0)  ## [N,H,W] --> [B,N,H,W]
            image = batch[-1]

            ## run forward pass
            out = self.model.detection(data)

            N = out.shape[1]
            # print(out.shape)

            if draw_path is not None:
                image_name = image_path.split('/')[-1]
                image_name = image_name.split('.')[0]
                filename = os.path.join(draw_path, "detection_{}".format(image_name))
                cam = out[0]

                ## normalize the cam
                max_val = torch.max(cam)
                min_val = torch.min(cam)
                cam = (cam - min_val) / (max_val - min_val)

                ## find intersected boxes
                if isinstance(image, np.ndarray):  ## this should be true when we use py2.7
                    img_numpy = image.copy()
                else:  ## this should be selected if using py3.+
                    img_numpy = image.permute(1, 2, 0).numpy()
                boxes = np.array(ss_box_finder(img_numpy)).squeeze()
                scores, classes = box_cam_intersection(boxes, cam.numpy(), (32, 30.6))

                ## draw image
                img_numpy = Drawer.draw_boxes_on_image(boxes[scores > BOX_SCORE_THRESHOLD], img_numpy,
                                                       classes[scores > BOX_SCORE_THRESHOLD], filename)


def main(resume, use_cuda=False, use_augment=False):
    ## path
    if True:
        timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join('detection', timestamp)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print('make a test result folder: ', save_path)
    else:
        save_path = None

    ## cuda or cpu
    if use_cuda:
        device = torch.device("cuda:0")
        print("using cuda")
    else:
        device = torch.device("cpu")
        print("using CPU")

    if use_augment:
        print("data are augmented randomly")

    ## dataloader
    data_loader = StreamingDataloader(imwidth=224)

    ## CNN model
    output_dim = 3
    model = MyNet(output_dim)
    ## resume a ckpt
    if resume is not None:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.to(device)
    print(model)

    ## init a detector
    detector = Detector(model, data_loader, device)
    ## perform detection
    """
    real-time feeding the image path to the detector
    """
    data_folder = '.'
    image_path = os.path.join(data_folder, 'test_images/frame0001.jpg')
    detector(image_path, draw_path=None)
    test_n=100
    tic = time.time()
    for _ in range(0, test_n):
        detector(image_path, draw_path=None)
    toc = time.time()
    print("time consumption", (toc - tic)/test_n)


## main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--augment', action='store_true')

    args = parser.parse_args()

    # assert args.resume is not None, "provide ckpt path to try again"
    main(args.resume, use_cuda=args.cuda, use_augment=args.augment)