import torch 
import torch.nn as nn 
import os 
import pandas as pd 
import numpy as np 
import cv2 


def main():

    train = pd.read_csv('data/train.csv')
    os.makedirs('data_png', exist_ok=True)


    for el in range(3):

        train = pd.read_parquet('data/train_image_data_{}.parquet'.format(el))

        # Indices 
        # Get indices, just to know how to name files 

        _ixes = train.image_id
        # Iter image by image 
        for _num,  _name in enumerate(_ixes):
            _img = train.iloc[_num, :].drop('image_id').values.reshape(137, 236)
            _img = _img.astype(np.uint8)
            cv2.resize(_img, (128, 128))
            cv2.imwrite(os.path.join('data_png', '{}.png'.format(_name)), _img)

if __name__ == "__main__":
    main()
