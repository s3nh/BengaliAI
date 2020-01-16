import gc
from models.fishnet.loader import MoFishnet150, load_checkpoint
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
# FIshnet path from config? rethink
from src.dataloader import BengaliDataLoader
import cv2
import torch

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
    
    
    for data, target in train_data_loader:
        print(target)
    
if __name__ == "__main__":
    main()

