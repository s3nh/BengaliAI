import random 
import math 
import torch 

from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms


class Resize(object):

    def __init__(self, size, interpolation = Image.BILINEAR):
        self.size = size
        self.interpolation =  interpolation

    def __call__(self, img):
        ratio = self.size[0]/self.size[1]
        w, h = img.size 
        if w/h < ratio:
            t = int(h*ratio)
            w_padding = (t-w)//2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int( w/ ratio)
            h_padding = (t-h)//2
            img = img.crop((0, -h_padding, w, h+h_padding))

        img = img.resize(self.size, self.interpolation)

        return img