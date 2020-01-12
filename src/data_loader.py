# Bengali AI data loader helper scripts. 
from sklearn.model_selection import train_test_split
import gc
import pandas as pd
import numpy as np 
from PIL import Image
import random 
import os 
import cv2 
import gc

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

def main():
    train_data = pd.read_csv('data/train.csv')
    print("Train dataset observations {}".format(train_data.shape[0]))
    # Grapheme root unique 
    root_uq = len(np.unique(train_data.grapheme_root))
    #Vowel diactric 
    vow_uq = len(np.unique(train_data.vowel_diacritic))
    #Consonant diactritic 
    con_uq = len(np.unique(train_data.consonant_diacritic))

    print("Unique ROOT UQ {}".format(root_uq))
    print("Unique VOW  UQ {}".format(vow_uq))
    print("Unique CON  UQ {}".format(con_uq))

    train_image = _convert_data(train_data = train_data)
    x_train, x_test, y_train, y_test = get_and_split(train_image, label = 'grapheme_root')

if __name__ == "__main__":
    main()
