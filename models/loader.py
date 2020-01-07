from __future__ import print_function 
from __future__ import division 
import torch 
import torch.nn as nn 
import numpy as np 
import torchvision 
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt 


FISHNET_PATH = 'pretrain/fishnet/fishnet150_ckpt_welltrained.tar'

def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint


def main():
    checkpoint = load_checkpoint(FISHNET_PATH)
    print(checkpoint)


if __name__ == "__main__":
    main()