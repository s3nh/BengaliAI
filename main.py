import sklearn
from sklearn.metrics import recall_score
import gc
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from src.dataloader import BengaliDataLoader
import cv2
import torch
import torch.nn as nn
device = 'cuda'
from pretrain.pretrain_loader import _DenseNet
from torch.utils.tensorboard import SummaryWriter


def macro_recall(pred_y, y, n_grapheme = 168, n_vowel = 11, n_consonant = 7):
    pred_y = torch.split(pred_y, [n_grapheme, n_vowel, n_consonant], dim=1)
    pred_labels = [torch.argmax(py, dim=1).cpu().numpy() for py in pred_y]
    
    y = y.cpu().numpy()
    
    recall_grapheme = sklearn.metrics.recall_score(pred_labels[0], y[:, 0], 
                                                   average='macro')
    recall_vowel = sklearn.metrics.recall_score(pred_labels[1], y[:, 1], 
                                                average='macro')
    recall_consonant = sklearn.metrics.recall_score(pred_labels[2], y[:, 2], 
                                                    average='macro')
    
    scores = [recall_grapheme, recall_vowel, recall_consonant]
    final_score = np.average(scores, weights = [2,1,1])
    return final_score 

def dataset_split(df = Dataset, _ratio = 0.2):
    n_images = len(df)
    assert n_images > 0 
    n_train = int(n_images * _ratio)
    n_test = int(n_images * (1-_ratio))
    train_df, test_df = torch.utils.data.random_split(df, [n_train, n_test])
    return train_df, test_df
     
def resize(df, size=64, need_progress_bar=True):
    resized = {}
    for i in range(df.shape[0]):
        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))
        resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized

def _train(history, train_data_loader, n_epochs = 10):
    mod =  _DenseNet()
    mod =  mod.cuda()
    criterion = nn.CrossEntropyLoss()
    history = pd.DataFrame()
    valid_recall = 0.0 
    best_valid_recall = 0.0 
    losses = []
    accs = []
    acc = 0.0
    total = 0.0  
    running_loss = 0.0 
    running_acc = 0.0 
    running_recall = 0.0 
    optimizer = torch.optim.Adam(mod.parameters(), lr = 0.01)

    writer = SummaryWriter()    
    
    for epoch in range(n_epochs):
        print("Processing for epoch {} from {} epochs".format(epoch, n_epochs))
        for _n , target in enumerate(train_data_loader):
            _images = target['image']
            total += len(_images)
            _label = target['label']
            _consonant = target['consonant']
            _grapheme = target['grapheme']
            _vowel = target['vowel']

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
            writer.add_scalar('batch_acc', acc, _n) 
   
        losses.append(running_loss/len(train_data_loader)*3)
        accs.append(running_acc/(len(train_data_loader)*3))
        
        
        writer.add_scalar('Acc', acc, epoch) 
        writer.add_scalar('Loss', running_loss, epoch) 
        torch.cuda.empty_cache() 
        gc.collect()
        history.loc[epoch, 'train_loss'] = losses[0]
        history.loc[epoch, 'train_acc'] = accs[0].cpu().numpy()
        torch.save(mod.state_dict(), 'models/densenet_{}.pth'.format(epoch))
    
def main():
    print("Beginning data loading") 
    data  = BengaliDataLoader('data_png/', 'data/train.csv')
    print("Beginning data split")
    train, test = dataset_split(data, _ratio = 0.2) 
    train_data_loader = DataLoader(train, batch_size = 128, shuffle=True)
    test_data_loader = DataLoader(test, batch_size=128, shuffle=True)
    history = pd.DataFrame() 
    torch.cuda.empty_cache()
    gc.collect()
    _train(history, train_data_loader, n_epochs = 50)
    
if __name__ == "__main__":
    main()