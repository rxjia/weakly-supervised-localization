from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np


import os
import argparse
from datetime import datetime
from mynet import MyNet, SeqDataset
from torch.utils.data import DataLoader
from utils import AverageMeter, Drawer



def evaluate(model, data_loader, device, draw_path=None, use_conf=False):

    ## set model
    model.eval()
    model = model.to(device)

    fc_weights = model.head.weight.data
    # print(fc_weights) ## [N, C]
    # print(fc_weights.shape)
    # input()

    ## loss
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss_avg = AverageMeter()
    acc_avg = AverageMeter()
    drawer = Drawer()

    with torch.no_grad():
        for batch_idx, batch in tqdm(
            enumerate(data_loader), 
            total=len(data_loader),
            ncols = 80,
            desc= f'testing',
            ):

            data = batch[0].to(device)
            gt_lbls = batch[1].to(device)
            gt_gt_lbls = batch[2].to(device)
            images = batch[-2]
            image_ids = batch[-1]

            ## run forward pass
            batch_size = data.shape[0]
            out, feat = model.forward_eval(data) ## out: [B, N]; feat: [B, C, H, W] 
            preds = torch.max(out, dim=-1)[1]

            ## compute loss
            class_loss = criterion(out, gt_lbls) ## [B, 1]
            if use_conf:
                weights = model.compute_entropy_weight(out)
                loss = (class_loss * (weights**2) + (1-weights)**2).mean()
            else:
                loss = class_loss.mean()
            
            ## record
            loss_avg.update(loss.item(), batch_size)
            positive = ((gt_lbls == preds) + (gt_gt_lbls>2)).sum()
            batch_acc = positive.to(torch.float)/batch_size
            acc_avg.update(batch_acc.item(), batch_size)

            if draw_path is not None:
                ## get cam
                preds = torch.max(out, dim=-1)[1]
                B,C,H,W = feat.shape
                cam = fc_weights[preds,:].unsqueeze(-1) * feat.reshape(B, C, -1) ## [B, C] * [B, C, H, W]
                cam = torch.sum(cam, dim=1).reshape(B, H, W)
                ## normalize the cam
                max_val = torch.max(cam)
                min_val = torch.min(cam)
                cam = (cam - min_val) / (max_val - min_val)
                ## convert to heatmap image
                cam_numpy = cam.permute(1,2,0).numpy()
                img_numpy = images[0].permute(1,2,0).numpy()
                filename = os.path.join(draw_path, f"test_{image_ids[0]}")
                drawer.draw_heatmap(filename, cam_numpy, img_numpy)

            if False:
                # print("class loss: ", class_loss)
                print("gt_labels: ", gt_lbls)
                print("gtgt_lbls: ", gtgt_lbls)
                print("preds: ", preds)

    return {"loss": loss_avg.avg, "acc": acc_avg.avg}
    # print("test loss: ", loss_avg.avg)



def main(resume, use_cuda=False):
    ## path
    timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join('test_result', timestamp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    ## cuda or cpu
    if use_cuda:
        device = torch.device("cuda:0")
        print("using cuda")
    else:
        device = torch.device("cpu")
        print("using CPU")


    ## dataloader
    dataset = SeqDataset(phase='test')
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
    )


    ## CNN model
    output_dim = 6
    model = MyNet(output_dim)
    ## resume a ckpt
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])


    ## evaluate
    evaluate(model, data_loader, device, draw_path=save_path)




## main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    assert args.resume is not None, "provide ckpt path to try again"
    main(args.resume, use_cuda=args.cuda)