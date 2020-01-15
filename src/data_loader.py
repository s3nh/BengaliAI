# Bengali AI data loader helper scripts. 
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data  import Dataset
import gc
import pandas as pd
import numpy as np 
from PIL import Image
import random 
import os 
import cv2 
import gc
import torch.optim as optim
from models.fishnet.loader import MoFishnet150, load_checkpoint

DATA_PATH = 'data/'
FILES = os.listdir(DATA_PATH)
FILES = [os.path.join(DATA_PATH, file) for file in FILES]
assert len(FILES) > 0, 'theres no data'
gc.enable();

def resize(df, size=64, need_progress_bar=True):
    resized = {}
    for i in range(df.shape[0]):
        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))
        resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized

def _convert_data(datafile = 'data/train_image_data_0.parquet', train_data = pd.DataFrame()):
    assert os.path.exists(datafile)
    b_train_data = pd.merge(pd.read_parquet(datafile), 
                       train_data, on = 'image_id').drop(['image_id'], axis=1)
    train_image = b_train_data.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic','grapheme'], axis = 1)
    del b_train_data
    gc.collect()
    train_image = resize(train_image, size = 128)/255
    labels = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
    return train_image

def get_and_split(df = pd.DataFrame(), label = 'grapheme_root'):
    assert df.shape[0] > 0, 'Data is not properly defined' 
    Y_train = b_train_data[label]    

    Y_train = pd.get_dummies(Y_train).values
    x_train, x_test, y_train, y_test =  train_test_split(df, Y_train, test_size = 0.1, random_state = 4321)
    return x_train, x_test, y_train, y_test


class DatasetMixin(Dataset):
    """
    Data transformer wrapper
    """
    def __init__(self, transform = None):
        self.transform = transform

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example_wrapper(i) for i in 
            six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return [self.get_example_wrapper(i) for i in index]         
        else:
            return self.get_example_wrapper(index)

    def __len__(self):
        raise NotImplementedError

    def get_example_wrapper(self, i):
        example = self.get_example(i)
        if self.transform:
            example = self.transform(example)
        return example

    def get_example(self, i):
        raise NotImplementedError  

class BengaliAIDataset(DatasetMixin):
    def __init__(self, images, labels=None, transform=None, indices = None):
        super(BengaliAIDataset, self).__init__(transform=transform)
        self.images = images
        self.labels = labels
        if indices is None:
            indices = np.arange(len(images))
        self.indices = indices 
        self.train = labels is not None

    def __len__(self):
        return len(self.indices)

    def get_example(self, i):
        i = self.indices[i]
        x = self.images[i]
        x = (255-x).astype(np.float32) / 255. 
        if self.train:
            y = self.labels[i]
            return x, y 
        else:
            return x

def prepare_image(datadir, data_type = 'train', 
    submission = True, indices = [0]):
    if submission:
        image_df_list = [pd.read_parquet(os.path.join(datadir, f'{data_type}_image_data_{i}.parquet')) for i in indices]
    print("Number of images {}".format(len(image_df_list)))    
    HEIGHT = 137
    WIDTH = 236 
    images = [df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH) for df in image_df_list]
    del image_df_list
    gc.collect()
    images = np.concatenate(images, axis=0)
    return images

# Based n corochan preprocess 


def _define_criterion(net):
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()
    loss3 = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)
    return loss1, loss2, loss3, optimizer 

def _train_(dataset , net = nn.Module()):
    crit1, crit2, crit3  , optimizer = _define_criterion(net)
    # From tutorial params 
    for epoch in range(2):
        for i, data in enumerate(dataset, 0):
            inputs, labels = data

        label1 = labels[0]
        label2 = labels[1]
        label3 = labels[2]

        optimizer.zero_grad()

        output1, output2, output3 = net(inputs)
        loss1 = crit1(output1, label1) 
        loss2 = crit2(output2, label2)
        loss3 = crit3(output3, label3) 

        loss1.backward()
        loss2.backward()
        loss3.backward()
        optimizer.step()


def main():
    train_data = pd.read_csv('data/train.csv')
    vow_uq = len(np.unique(train_data.vowel_diacritic))
    train_labels = train_data[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values 
    indices =  [0,1,2,3]
    train_images = prepare_image(datadir = 'data', data_type='train', 
    submission=True, indices=indices)

    train_dataset = BengaliAIDataset(train_images, train_labels)
    x, y = train_dataset.get_example(1)
    print(x)
    print(y)
    x=  x.cuda() 
    mod = MoFishnet150(path=FISHNET_PATH)
    mod.cuda()
    output1, output2, output3 = mod(x)
    print(output1, output2, output3)


if __name__ == "__main__":
    main()
