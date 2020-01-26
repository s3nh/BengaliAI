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

def _train(epoch, history, train_data_loader):
    mod =  _DenseNet()
    mod =  mod.cuda()

    criterion = nn.CrossEntropyLoss()
    history = pd.DataFrame()
    valid_recall = 0.0 
    best_valid_recall = 0.0 
    n_epochs = 10
    
    losses = []
    accs = []
    acc = 0.0
    total = 0.0  
    running_loss = 0.0 
    running_acc = 0.0 
    running_recall = 0.0 
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
        loss1 = criterion(output1, _consonant)
        loss2 = criterion(output2, _grapheme)
        loss3 = criterion(output3, _vowel)

        running_loss = loss1.item() + loss2.item() + loss3.item() 
        
        running_acc += (output1.argmax(1) == _consonant).float().mean() 
        running_acc += (output2.argmax(1) == _grapheme).float().mean()
        running_acc  += (output3.argmax(1) == _vowel).float().mean()
        
        (loss1+loss2+loss3).backward()        
        optimizer.step()
        optimizer.zero_grad()
        acc = running_acc/total 
        print(acc)
    losses.append(running_loss/len(train_loader)*3)
    accs.append(running_acc/(len(train_data_loader)*3))
    print(' train : {}\tacc : {:.2f}%'.format(epoch, running_acc/(len(train_data_loader)*3)))
    print('loss : {:/4f}'.format(running_loss/len(train_data_loader)))
    
    torch.cuda.empty_cache() 
    gc.collect()
    
    
    history.loc[epoch, 'train_loss'] = losses[0]
    history.loc[epoch, 'train_acc'] = accs[0].cpu().numpy()
   
FISHNET_PATH = 'pretrain/fishnet/fishnet150_ckpt_welltrained.tar'
def main():
    train = BengaliDataLoader('data_png/', 'data/train.csv')
    train_data_loader = DataLoader(train, batch_size = 32, shuffle=True)
    n_epochs = 10
    history = pd.DataFrame() 
    for epoch in range(n_epochs):
        torch.cuda.empty_cache()
        gc.collect()
        _train(epoch, history, train_data_loader)
        
        torch.save(model.state_dict(), 'densenet201_{}.pth'.format(epoch))      
        
    """
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()
    loss3 = nn.CrossEntropyLoss()
    """
    
    
if __name__ == "__main__":
    main()

