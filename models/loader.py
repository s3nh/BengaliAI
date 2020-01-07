from __future__ import print_function 
from __future__ import division 
import torch 
import torch.nn as nn 
import numpy as np 
import torchvision 
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt 
from models.net_factory import fishnet150

FISHNET_PATH = 'pretrain/fishnet/fishnet150_ckpt_welltrained.tar'

def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint


def main():
    model = fishnet150()
    checkpoint = load_checkpoint(FISHNET_PATH)
    best_prec1 = checkpoint['best_prec1'] 
    print(best_prec1)
    model.load_state_dict(checkpoint['state_dict'])
if __name__ == "__main__":
    main()