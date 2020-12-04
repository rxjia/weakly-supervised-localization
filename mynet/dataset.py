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

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from utils import *

import warnings
warnings.filterwarnings("ignore")


class SeqDataset(Dataset):
    phases = ['train', 'val', 'test']
    classes_str = ['C', 'H', 'P', 'CP', 'PH', 'HC']
    classes = {'C': [0], 'H': [1], 'P': [2], 'CP': [0,2], 'PH': [1,2], 'HC': [0,1]}
    torch.manual_seed(0)

    def __init__(
        self, phase='train', imwidth=224, 
        do_augmentations=False, img_ext='.png',
        ):
        
        assert phase in self.phases

        self.imwidth = imwidth
        self.phase = phase
        self.train = True if phase != 'testing' else False
        self.data_root = os.path.join('/home/yanglei/codes/WSOL/', phase)   ## desktop

        ## read image paths
        metafile_path = f'./metadata/{phase}_images.json'
        files = read_json(metafile_path)
        files.sort(key=lambda x: x[5:-4])
        self.image_paths = files
        
        ## data pre-processing
        normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                         std=[0.2599, 0.2371, 0.2323])
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
        image_path = self.image_paths[index]
        image = Image.open(os.path.join(self.data_root, image_path))
        image = self.initial_transforms(image.convert("RGB")) ## to PIL.rgb
        image = TF.pil_to_tensor(image) ## to tensor
        data = self.transforms(TF.to_pil_image(image)) ## to pil for transform

        # print(image_path)
        label, path = image_path.split('/')[-2:]
        # test0004.png
        image_id = path.split('.')[0] ## str
        assert label in self.classes
        n = len(self.classes[label])
        rand_idx = torch.randperm(n)[0]
        target = self.classes[label][rand_idx]

        return data, target, self.classes_str.index(label), image, image_id



if __name__ == '__main__':
    dataset = SeqDataset(phase='test')
    files = dataset.image_paths
    files.sort(key=lambda x: x[5:-4])
    print(files)
