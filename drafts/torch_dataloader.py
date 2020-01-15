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

        train = pd.read_parquest('data/train_image_data_{}.parquet'.format(el))

        # Indices 
        # Get indices, just to know how to name files 

        _ixes = train.image_id
        # Iter image by image 
        for iter in range(train.shape[0]):
            _name =  _ixes[iter]
            _img = train.loc[_name, :].drop('image_id').values.reshape(137, 236)
            # Change type frmo object to uint8 
            _img = _img.astype(np.uint8)
            cv2.resize(_img, (128, 128))
            cv2.imwrite(os.path.join('data_png', '{}.png'.format(_name)), _img)

if __name__ == "__main__":
    main()