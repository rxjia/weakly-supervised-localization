import torch
import torch.nn as nn
import torch.nn.functional as F 

import torchvision.models as models

## ResNet50
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()        
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

    def forward(self, x):
        return self.features(x)


class MyNet(nn.Module):
    def __init__(self, output_dim):
        self.resnet = ResNet50()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1)) ## average pooling
        self.head = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv2d(128, output_dim, kernel_size=(1,1)),
        )

    def forward(self, x):
        h = self.resnet(x)
        h = self.avg_pool(h)
        return self.head(h)