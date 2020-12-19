from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy

import os
import argparse
from datetime import datetime
from mynet import SeqDataset
from mynet import MyNet
from torch.utils.data import DataLoader
from utils import AverageMeter, Drawer
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


def evaluate(model, data_loader, device, draw_path=None, use_conf=False):
    ## set model
    model.eval()
    model = model.to(device)

    ## loss
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss_avg = AverageMeter()
    acc_avg = AverageMeter()
    drawer = Drawer()

    with torch.no_grad():
        for batch_idx, batch in tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                ncols=80,
                desc='testing',
        ):

            # if batch_idx == 20: break

            data = batch[0].to(device)
            images = batch[-2]
            image_ids = batch[-1]

            ## run forward pass
            batch_size = data.shape[0]
            out = model.detection(data)  ## [B,N,H,W]
            N = out.shape[1]
            # print(out.shape)

            if draw_path is not None:
                filename = os.path.join(draw_path, "test_{}".format(image_ids[0]))
                cam = out[0]
                ## normalize the cam
                max_val = torch.max(cam)
                min_val = torch.min(cam)
                cam = (cam - min_val) / (max_val - min_val)
                ## convert to heatmap image
                img_numpy = images[0].permute(1, 2, 0).numpy()
                boxes = np.array(ss_box_finder(img_numpy)).squeeze()
                scores, classes = box_cam_intersection(boxes, cam.numpy(), (32, 30.6))

                img_numpy = Drawer.draw_boxes_on_image(boxes[scores > BOX_SCORE_THRESHOLD], img_numpy,
                                                       classes[scores > BOX_SCORE_THRESHOLD], filename)

                # cam_total = np.concatenate(list(cam))
                # img_numpy = np.concatenate([img_numpy,img_numpy,img_numpy])
                # drawer.draw_heatmap(cam_total, img_numpy, filename)
                # input()


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
    test_path = './metadata/test_images.json'
    new_test_path = './metadata/new_test_images.json'
    detection_path = './metadata/detection_test_images.json'

    dataset = SeqDataset(
        phase='test',
        do_augmentations=use_augment,
        metafile_path=detection_path,
        return_gt_label=False)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
    )

    ## CNN model
    output_dim = 3
    model = MyNet(output_dim)
    ## resume a ckpt
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])

    print(model)

    ## evaluate
    log = evaluate(model, data_loader, device, draw_path=save_path, use_conf=True)


## main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--augment', action='store_true')

    args = parser.parse_args()

    assert args.resume is not None, "provide ckpt path to try again"
    main(args.resume, use_cuda=args.cuda, use_augment=args.augment)