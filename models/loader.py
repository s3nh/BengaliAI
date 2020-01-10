from __future__ import print_function 
from __future__ import division 
import torch 
import torch.nn as nn 
import numpy as np 
import torchvision 
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt 
from models.net_factory import fishnet150
from torch.nn import Flatten
FISHNET_PATH = 'pretrain/fishnet/fishnet150_ckpt_welltrained.tar'
def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint

class MoFishnet150(nn.Module):
    def __init__(self, path =FISHNET_PATH):
        super(MoFishnet150, self).__init__()
        self.fishnet = fishnet150()
        self.path = path 
        checkpoint = load_checkpoint(self.path)

        self.fishnet.load_state_dict(checkpoint['state_dict'], strict = False)
        self.fishnet.cuda()

        for g in self.fishnet.parameters():
            g.requires_grad = False

        self.features = list(self.fishnet.children())[:-1]
        self.features = nn.Sequential(
            *self.features
        )
        print(self.features)
        self.pool = nn.MaxPool2d(2)
        #50176
        self.fc1 = nn.Linear(50176, 11)
        self.fc2 = nn.Linear(50176,168)
        self.fc3 = nn.Linear(50176,7)

    def forward(self, image):
        x = self.features(image)
        print(x.shape)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x) 
        return x1, x2, x3


def main():
    mod = MoFishnet150(path=FISHNET_PATH) 
    mod.cuda()
    x = torch.zeros((1, 3, 224, 224))
    x = x.cuda()
    x1, x2, x3 = mod(x)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)

    """
    checkpoint = load_checkpoint(FISHNET_PATH)
    best_prec1 = checkpoint['best_prec1'] 
    model.load_state_dict(checkpoint['state_dict'], strict = False)
    print(model)
    print(output.shape)

    for p in model.parameters():
        p.requires_grad = False
    backbone = list(model.children())[:-1]
    """
if __name__ == "__main__":
    main()