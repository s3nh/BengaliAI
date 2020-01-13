import gc
from models.fishnet.loader import MoFishnet150, load_checkpoint
import pandas as pd 
from torch import Dataset
# FIshnet path from config? rethink
from src.data_loader import _convert_data, get_and_split
import cv2

def resize(df, size=64, need_progress_bar=True):
    resized = {}
    for i in range(df.shape[0]):
        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))
        resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized

class GraphemeDataset(Dataset):
    def __init__(self, df, label, type = 'train'):
        self.df = df
        self.label = label


    
FISHNET_PATH = 'pretrain/fishnet/fishnet150_ckpt_welltrained.tar'
def main():
    mod = MoFishnet150(path=FISHNET_PATH)
    mod.cuda()
    train_data = pd.read_csv('data/train.csv') 
    print(train_data.shape)
    b_train_data = pd.merge(pd.read_parquet('data/train_image_data_0.parquet'), 
                       train_data, on = 'image_id').drop(['image_id'], axis=1)
    train_image = b_train_data.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic','grapheme'], axis = 1)
    train_image = resize(train_image)/255
    train_image = train_image.values.reshape(-1, 64, 64, 1) 
    y_train1 =  b_train_data.grapheme_root
    y_train2 = b_train_data.vowel_diacritic
    y_train3 = b_train_data.consonant_diacritic

    train_loader = torch.utils.data.DataLoader([x_train, y_train1], batch_size = 16, 
    shuffle=True)
    
if __name__ == "__main__":
    main()

