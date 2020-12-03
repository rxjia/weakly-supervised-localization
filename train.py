from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.nn.functional as F 

import os
from datetime import datetime
import argparse
from mynet import MyNet, SeqDataset
from torch.utils.data import DataLoader
from utils import AverageMeter, SimpleLossLogger



def _save_checkpoint(path, epoch, model, optimizer=None):
    arch = type(model).__name__
    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }
    if optimizer is not None:
        state = {**state, 'optimizer': self.optimizer.state_dict()}

    filename = os.path.join(
        path,
        'checkpoint-epoch{}.pth'.format(epoch)
    )
    torch.save(state, filename, _use_new_zipfile_serialization=False)
    print("Saving checkpoint: {} ...".format(filename))
    

def main(config, resume):

    # parameters
    batch_size = config.get('batch_size', 32)
    start_epoch = config['epoch']['start']
    max_epoch = config['epoch']['max']

    ## path
    save_path = config['save_path']
    timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_path, timestamp)

    result_path = os.path.join(save_path, 'result')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    model_path = os.path.join(save_path, 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)




    ## cuda or cpu
    if config['n_gpu'] == 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("using CPU")
    else:
        device = torch.device("cuda:0")
    
    ## dataloader
    dataset = SeqDataset(phase='train')
    data_loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=1,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        # **loader_kwargs,
    )

    ## CNN model
    output_dim = 6
    model = MyNet(output_dim)
    model.train()
    model = model.to(device)

    ## loss
    criterion = nn.CrossEntropyLoss(reduction='none')

    ## optimizer
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optim_params={
        'lr': 0.0001,
        'weight_decay': 0,
        'amsgrad': False,
    }
    optimizer = torch.optim.Adam(params, **optim_params)
    lr_params = {
        'milestones':[],
        'gamma':0.1,
    }
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,**lr_params)

    loss_avg = AverageMeter()
    loss_logger = SimpleLossLogger()

    ## loop
    for epoch in range(start_epoch, max_epoch):
        loss_avg.reset()

        for batch_idx, batch in tqdm(
            enumerate(data_loader), 
            total=len(data_loader),
            ncols = 80,
            desc= f'training epoch {epoch}',
            ):
            data = batch[0].to(device)
            gt_lbls = batch[1].to(device)
            gt_gt_lbls = batch[-1].to(device)


            ## set zerograd
            optimizer.zero_grad()

            ## run forward pass
            out = model(data) ## logits: [B, NC]; conf: [B, 1] 
            # print("out shape: ", out.shape)
            
            weights = model.compute_entropy_weight(out)
            # print("weights shape: ", weights.shape)

            ## compute loss
            class_loss = criterion(out, gt_lbls) ## [B, 1]
            # print("class_loss shape: ", class_loss.shape)

            # if batch_idx == 0:
            #     print("weights: ", weights)
            #     print("class loss: ", class_loss)
            #     print("gt_gt_lbls: ", gt_gt_lbls)
            #     print("gt_labels: ", gt_lbls)
            #     print("preds: ", torch.max(out, dim=-1)[1])

            # loss = class_loss * (weights**2) + (1-weights)**2
            loss = class_loss.mean()

            loss_avg.update(loss.item(), batch_size)

            ## run backward pass
            loss.backward()
            optimizer.step() ## update

        ## each epoch
        print("training loss: ", loss_avg.avg)
        loss_logger.update(loss_avg.avg)

        best_idx = loss_logger.get_best()
        if best_idx == epoch:
            print('save ckpt')
            ## save ckpt
            _save_checkpoint(model_path, epoch, model)

        lr_scheduler.step()



##
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='the size of each minibatch')
    parser.add_argument('-g', '--n_gpu', default=None, type=int,
                        help='if given, override the numb')
    parser.add_argument('-e', '--epoch', default=10, type=int,
                        help='if given, override the numb')
    parser.add_argument('-s', '--save_path', default='saved', type=str,
                        help='path to save')

    
    # We allow a small number of cmd-line overrides for fast dev
    args = parser.parse_args()

    config = {}
    config['batch_size'] = args.batch_size
    config['n_gpu'] = args.n_gpu
    config['save_path'] = args.save_path
    config['epoch'] = {
        'start': 0,
        'max': args.epoch
    }

    main(config, args.resume)