import gc
#from models.fishnet.loader import MoFishnet150, load_checkpoint
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from src.dataloader import BengaliDataLoader
import cv2
import torch
import torch.nn as nn
device = 'cuda'
from pretrain.pretrain_loader import _DenseNet


def resize(df, size=64, need_progress_bar=True):
    resized = {}
    for i in range(df.shape[0]):
        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))
        resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized
    
FISHNET_PATH = 'pretrain/fishnet/fishnet150_ckpt_welltrained.tar'
def main():
    mod =  _DenseNet()
    mod =  mod.cuda()
    train = BengaliDataLoader('data_png/', 'data/train.csv')
    train_data_loader = DataLoader(train, batch_size = 32, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    """
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()
    loss3 = nn.CrossEntropyLoss()
    """
    losses = []
    accs = []
    acc = 0.0
    total = 0.0  
    runing_loss = 0.0 
    runing_acc = 0.0 
    runing_recall = 0.0 
    optimizer = torch.optim.Adam(mod.parameters(), lr = 0.01)
    for target in train_data_loader:

        _images = target['image']
        total += len(_images)
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
       
        output1, output2, output3 = mod(_images)  
        print(output1, output2, output3)
        loss1 = criterion(output1, _consonant)
        loss2 = criterion(output2, _grapheme)
        loss3 = criterion(output3, _vowel)

        running_loss = loss1.item() + loss2.item() + loss3.item() 
        
        running_acc += (output1.argmax(1) == _consonant).float().mean() 
        running_acc += (output2.argmax(1) == _grapheme).float().mean()
        runing_acc  += (output3.argmax(1) == _vowel).float().mean()
        
        
        (loss1+loss2+loss3).backward()        
        optimizer.step()
        optimizer.zero_grad()
        
        acc = running_acc/total 

if __name__ == "__main__":
    main()

