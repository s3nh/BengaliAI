import gc
from models.fishnet.loader import MoFishnet150, load_checkpoint
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
# FIshnet path from config? rethink
from src.dataloader import BengaliDataLoader
import cv2
import torch
import torch.nn as nn
device = 'cuda'


def resize(df, size=64, need_progress_bar=True):
    resized = {}
    for i in range(df.shape[0]):
        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))
        resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized
    
FISHNET_PATH = 'pretrain/fishnet/fishnet150_ckpt_welltrained.tar'
def main():
    mod = MoFishnet150(path=FISHNET_PATH)
    mod.cuda()
    train = BengaliDataLoader('data_png/', 'data/train.csv')
    train_data_loader = DataLoader(train, batch_size = 32, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    """
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()
    loss3 = nn.CrossEntropyLoss()
    """
    optimizer = torch.optim.Adam(mod.parameters(), lr = 0.01)
    
    for target in train_data_loader:
        _images = target['image']

        _label = target['label']
        _consonant = target['consonant']
        _grapheme = target['grapheme']
        _vowel = target['vowel']
        
        # Switch to device 
        #_label = _label.to(device)
        
        _images = _images.to(device) 
        _consonant = _consonant.to(device)
        _grapheme = _grapheme.to(device)
        _vowel = _vowel.to(device)
        
        optimizer.zero_grad()
       
        output1, output2, output3 = mod(_images)  
        print(output1, output2, output3)
        loss1 = criterion(output1, _consonant)
        loss2 = criterion(output2, _grapheme)
        loss3 = criterion(output3, _vowel)



if __name__ == "__main__":
    main()

