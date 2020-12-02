from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.nn.functional as F 

import argparse
from mynet import MyNet, SeqDataset
from torch.utils.data import DataLoader


def main(config, resume):

    # parameters
    batch_size = config.get('batch_size', 32)
    start_epoch = config['epoch']['start']
    max_epoch = config['epoch']['max']




    ## cuda or cpu
    if config['n_gpu'] == 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
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
    criterion = nn.CrossEntropyLoss()

    ## optimizer
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optim_params={
        'lr': 0.001,
        'weight_decay': 0,
        'amsgrad': False,
    }
    optimizer = torch.optim.Adam(params, **optim_params)
    lr_params = {
        'milestones':[],
        'gamma':0.1,
    }
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,**lr_params)

    ## loop
    for epoch in range(start_epoch, max_epoch):
        for batch_idx, batch in tqdm(enumerate(data_loader)):
            data = batch['data'].to(device)
            gt_lbls = batch['gt_label'].to(device)

            ## set zerograd
            optimizer.zero_grad()

            ## run forward pass
            logits, conf = model(data) ## logits: [B, NC]; conf: [B, 1] 
            
            ## compute loss
            class_loss = criterion(logits, gt_lbls, reduction='none') ## [B, 1]
            loss = (conf * class_loss).sum()

            ## run backward pass
            loss.backward()
            optimizer.step() ## update

        ## each epoch
        lr_scheduler.step()






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

    
    # We allow a small number of cmd-line overrides for fast dev
    args = parser.parse_args()

    config = {}
    config['batch_size'] = args.batch_size
    config['n_gpu'] = args.n_gpu

    config['epoch'] = {
        'start': 0,
        'max': args.epoch
    }

    main(config, args.resume)