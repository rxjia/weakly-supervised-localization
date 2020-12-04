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
from io import BytesIO
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from utils import *

import warnings
warnings.filterwarnings("ignore")

class JPEGNoise(object):
    def __init__(self, low=30, high=99):
        self.low = low
        self.high = high


    def __call__(self, im):
        H = im.height
        W = im.width
        rW = max(int(0.8 * W), int(W * (1 + 0.5 * torch.randn([]))))
        im = TF.resize(im, (rW, rW))
        buf = BytesIO()
        im.save(buf, format='JPEG', quality=torch.randint(self.low, self.high,
                                                          []).item())
        im = Image.open(buf)
        im = TF.resize(im, (H, W))
        return im


class PcaAug(object):
    _eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    _eigvec = torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, im):
        alpha = torch.randn(3) * self.alpha
        rgb = (self._eigvec * alpha.expand(3, 3) * self._eigval.expand(3, 3)).sum(1)
        return im + rgb.reshape(3, 1, 1)

class SeqDataset(Dataset):
    phases = ['train', 'val', 'test']
    classes_str = ['C', 'H', 'P', 'CP', 'PH', 'HC']
    classes = {'C': [0], 'H': [1], 'P': [2], 'CP': [0,2], 'PH': [1,2], 'HC': [0,1]}
    torch.manual_seed(0)

    def __init__(
        self, phase='train', imwidth=224, 
        metafile_path = None,
        do_augmentations=False, img_ext='.png',
        ):
        
        ## parameters
        assert phase in self.phases
        self.imwidth = imwidth
        self.phase = phase
        self.train = True if phase != 'test' else False
        self.do_augmentations = do_augmentations

        ## read image paths
        if metafile_path is None:
            metafile_path = f'./metadata/{phase}_images.json'

        files = read_json(metafile_path)
        files.sort(key=lambda x: x[5:-4])
        self.image_paths = files
        
        ## data pre-processing
        self.normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                         std=[0.2599, 0.2371, 0.2323])
        self.initial_transforms = transforms.Compose([transforms.Resize(self.imwidth)])
        self.augmentations = transforms.RandomApply(
                    [
                        transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=0.01),
                        transforms.RandomAffine(degrees=45,translate=(0.1,0.1),scale=(0.9,1.2))
                    ], 
                    p=0.3)
        self.to_tensor = transforms.ToTensor()
        

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = self.initial_transforms(image.convert("RGB")) ## to PIL.rgb
        if self.do_augmentations:
            image = self.augmentations(image)
        image = TF.pil_to_tensor(image) ## save the image tensor for visualization
        data = self.normalize(self.to_tensor(TF.to_pil_image(image)))

        label, path = image_path.split('/')[-2:]
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
