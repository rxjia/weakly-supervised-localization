import torch
import torch.nn as nn
import torch.nn.functional as F 

import torchvision.models as models


def compute_entropy(scores):
    ## logits: [B, N]
    logits = F.log_softmax(scores, dim=-1)
    probs = F.softmax(scores, dim=-1)
    entropy = (probs * logits).sum(dim=-1).neg()
    return entropy


## ResNet18
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()        
        resnet = models.resnet18(pretrained=True)
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
        super(MyNet, self).__init__()
        self.output_dim = output_dim
        self.resnet = ResNet18()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1)) ## average pooling
        self.dropout = nn.Dropout(p=0.1)
        self.head = nn.Linear(512, 3)

        self.b = torch.ones((1, output_dim))
        self.max_ent = compute_entropy(self.b)

    def forward(self, x):
        B = x.shape[0]
        h = self.resnet(x)
        h = self.avg_pool(h).reshape(B, 512)
        h = self.dropout(h)
        return self.head(h)

    def forward_eval(self, x):
        B = x.shape[0]
        x = self.resnet(x)
        h = self.avg_pool(x).reshape(B, 512)
        return self.head(h), x

    def compute_entropy_weight(self, scores):
        entropy = compute_entropy(scores)
        entropy_weight = 1.0 - entropy/self.max_ent.to(scores.device)
        return entropy_weight