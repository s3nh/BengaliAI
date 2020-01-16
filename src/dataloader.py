from __future__ import print_function, division
import os 
import torch 
import pandas as pd 
from skimage import io, transform 
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



import warnings
warnings.filterwarnings("ignore")



class BengaliDataLoader(Dataset):
    def __init__(self, image_folder, label_file, transform=None):
        self.image_folder = image_folder  
        
        self.label_file = pd.read_csv(label_file)
        self.grapheme  =  self.label_file.grapheme_root
        self.vowel = self.label_file.vowel_diacritic
        self.consonant = self.label_file.consonant_diacritic
        self.label_file = self.label_file.image_id
        self.transform = None
        
    # Magic function 
    def __len__(self):
        return len(self.label_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.image_folder, '{}.png'.format(self.label_file.iloc[idx]) 
                                )
        image = io.imread(img_name)
        label = self.label_file.iloc[idx]
        grapheme = self.grapheme.iloc[idx]
        vowel = self.vowel.iloc[idx]
        consonant = self.consonant[idx]
        sample = {'image' : image, 'label' : label, 
                 'grapheme': grapheme, 
                 'vowel' : vowel, 
                 'consonant' : consonant}
        grapheme  =  label[0]
        vowel  = label[1]
        consonant = label[2] 
        if self.transform:
            sample = self.transform(sample)
        return sample
       
def main():
    train = BengaliDataLoader('data_png/', 'data/train.csv')
    print(train[9])
    
if __name__ == '__main__':
    main()