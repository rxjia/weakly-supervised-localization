import numpy as np
import os,sys,inspect
import glob
import torch
import numpy

from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from utils import *

class SeqDataset(Dataset):
    classes = []
    def __init__(
        self, root, 
        phase='training', imwidth=224, 
        do_augmentations=False, img_ext='.png',
    ):
        self.root = root
        self.imwidth = imwidth
        self.phase = phase
        self.train = True if phase != 'testing' else False
        self.data_root = os.path.join('/home/yanglei/codes/WSOL/seq_data', phase)   ## desktop

        ## read image paths
        self.image_paths = read_json(os.path.join(self.root, f'{phase}_images.json'))

        ## data pre-processing
        augmentations = [
            JPEGNoise(),
            transforms.transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=0.01),
            transforms.ToTensor(),
            PcaAug()
        ] if (self.train and do_augmentations) else [transforms.ToTensor()]
        self.initial_transforms = transforms.Compose(
            [transforms.Resize(self.imwidth)])
        self.transforms = transforms.Compose(augmentations + [normalize])


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_root, image_path))
        image = self.initial_transforms(image.convert("RGB")) ## to PIL.rgb
        image = TF.pil_to_tensor(image) ## to tensor
        data = self.transforms(TF.to_pil_image(image)) ## to pil for transform
        return data, image